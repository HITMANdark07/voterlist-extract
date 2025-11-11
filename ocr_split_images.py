from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import json
import os
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictors (loaded once per process)
_foundation = None
_detection_predictor = None
_recognition_predictor = None

# Try to import nvidia-ml-py for GPU monitoring (replaces deprecated pynvml)
# nvidia-ml-py provides the pynvml module - see https://pypi.org/project/nvidia-ml-py/
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("nvidia-ml-py not available - GPU monitoring disabled")


def load_predictors():
    """Load Surya OCR predictors once (reused for all images in a process)."""
    global _foundation, _detection_predictor, _recognition_predictor
    
    if _foundation is None or _detection_predictor is None or _recognition_predictor is None:
        logger.info("Loading Surya OCR predictors (first time)...")
        _foundation = FoundationPredictor()
        _detection_predictor = DetectionPredictor()
        _recognition_predictor = RecognitionPredictor(_foundation)
        logger.info("Predictors loaded successfully")
    
    return _foundation, _detection_predictor, _recognition_predictor


def ocr_file(file_path, output_text_path, recognition_predictor, detection_predictor, output_json_path=None):
    """
    Run OCR on a single image and extract text.
    
    Args:
        file_path: Path to image file
        output_text_path: Path to save text file
        recognition_predictor: Pre-loaded RecognitionPredictor instance
        detection_predictor: Pre-loaded DetectionPredictor instance
        output_json_path: Optional path to save JSON results
    
    Returns:
        Tuple of (file_path, success, error_message)
    """
    try:
        # Load the image
        image = Image.open(file_path)
        
        # Run the OCR using pre-loaded predictors
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        
        # Extract text from results
        all_text_lines = []
        results = []
        
        # Each prediction is an OCRResult object
        for page_pred in predictions:
            text_lines = []
            for line in page_pred.text_lines:
                text = line.text
                if text:
                    all_text_lines.append(text)
                text_lines.append({
                    "text": text,
                    "confidence": line.confidence,
                    "bbox": line.bbox,
                    "polygon": line.polygon
                })
            results.append({
                "page": getattr(page_pred, "page", 1),
                "text_lines": text_lines,
                "image_bbox": getattr(page_pred, "image_bbox", None)
            })
        
        # Combine all text lines
        full_text = "\n".join(all_text_lines)
        
        # Save JSON if requested
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save text
        os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        return (file_path, True, None)
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return (file_path, False, str(e))


def get_gpu_stats() -> Optional[Dict]:
    """
    Get GPU statistics using NVIDIA Management Library (nvidia-ml-py).
    See https://pypi.org/project/nvidia-ml-py/ for API documentation.
    
    Returns:
        Dictionary with GPU stats or None if not available
    """
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        
        # Get device count
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            pynvml.nvmlShutdown()
            return None
        
        # Get stats from the first NVIDIA GPU (usually the primary one)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get GPU name (returns string directly, no need to decode)
        name = pynvml.nvmlDeviceGetName(handle)
        
        # Get memory info (returns struct with total, free, used attributes)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem_gb = mem_info.total / (1024**3)
        used_mem_gb = mem_info.used / (1024**3)
        free_mem_gb = mem_info.free / (1024**3)
        
        # Get utilization rates (returns struct with gpu and memory attributes)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        
        # Get temperature (may not be available on all systems)
        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            # Temperature not available or error accessing it
            pass
        
        # Shutdown NVML
        pynvml.nvmlShutdown()
        
        return {
            'name': name,
            'total_memory_gb': total_mem_gb,
            'used_memory_gb': used_mem_gb,
            'free_memory_gb': free_mem_gb,
            'utilization_percent': gpu_util,
            'temperature_c': temp
        }
    except Exception as e:
        logger.debug(f"Error getting GPU stats: {e}")
        # Try to shutdown if initialization succeeded
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        return None


def calculate_optimal_workers(gpu_stats: Optional[Dict] = None, 
                              estimated_memory_per_worker_gb: float = 2.0) -> int:
    """
    Calculate optimal number of workers based on GPU memory and CPU count.
    
    Args:
        gpu_stats: GPU statistics dictionary from get_gpu_stats()
        estimated_memory_per_worker_gb: Estimated GPU memory per worker in GB
    
    Returns:
        Recommended number of workers
    """
    max_cpu_workers = cpu_count()
    
    if gpu_stats is None:
        # No GPU info available, use conservative CPU-based estimate
        logger.info("GPU stats not available, using CPU-based worker calculation")
        return min(2, max_cpu_workers)
    
    free_mem_gb = gpu_stats['free_memory_gb']
    gpu_util = gpu_stats['utilization_percent']
    
    # Calculate workers based on available GPU memory
    # Reserve 20% of free memory as buffer
    usable_memory_gb = free_mem_gb * 0.8
    memory_based_workers = int(usable_memory_gb / estimated_memory_per_worker_gb)
    
    # Consider GPU utilization - if already high, reduce workers
    if gpu_util > 80:
        # GPU is heavily utilized, be conservative
        utilization_factor = 0.5
    elif gpu_util > 60:
        # GPU is moderately utilized
        utilization_factor = 0.7
    else:
        # GPU has capacity
        utilization_factor = 1.0
    
    memory_based_workers = max(1, int(memory_based_workers * utilization_factor))
    
    # Take minimum of CPU-based and memory-based workers
    optimal_workers = min(memory_based_workers, max_cpu_workers)
    
    # Ensure at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    return optimal_workers


def process_batch(image_batch: List[Tuple[Path, Path]]) -> Tuple[int, int]:
    """
    Process a batch of images in a single worker process.
    Models are loaded once per worker process and reused for all images in the batch.
    
    Args:
        image_batch: List of tuples (image_path, output_text_path)
    
    Returns:
        Tuple of (success_count, total_count)
    """
    # Load predictors once for this worker process (reused for all images in batch)
    foundation, detection_predictor, recognition_predictor = load_predictors()
    
    success_count = 0
    total_count = len(image_batch)
    
    for image_path, output_text_path in image_batch:
        file_path, success, error = ocr_file(
            str(image_path), 
            str(output_text_path),
            recognition_predictor,
            detection_predictor
        )
        if success:
            success_count += 1
            logger.info(f"Processed: {Path(file_path).name}")
        else:
            logger.error(f"Failed: {Path(file_path).name} - {error}")
    
    return (success_count, total_count)


def process_all_images(split_dir="voter split", output_dir="ocr_results", batch_size=5, num_workers=2):
    """
    Process all images in the voter split directory and save text to .txt files.
    Uses multiprocessing with batch processing.
    
    Args:
        split_dir: Directory containing split images (default: "voter split")
        output_dir: Output directory for text files (default: "ocr_results")
        batch_size: Number of images to process per batch (default: 5, reduced for memory efficiency)
        num_workers: Number of worker processes (default: 2, reduced for memory efficiency)
                     Set to 1 for sequential processing if memory is limited
    """
    # Record start time
    start_time = time.time()
    
    split_path = Path(split_dir)
    
    if not split_path.exists():
        logger.error(f"Split directory does not exist: {split_dir}")
        return
    
    # Find all PDF directories
    pdf_dirs = [d for d in split_path.iterdir() if d.is_dir()]
    
    if not pdf_dirs:
        logger.warning(f"No PDF directories found in {split_dir}")
        return
    
    logger.info(f"Found {len(pdf_dirs)} PDF directory(ies) to process")
    
    # Create output base directory
    output_base_path = Path(output_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Get GPU stats and calculate optimal workers
    gpu_stats = get_gpu_stats()
    
    if gpu_stats:
        logger.info(f"\n{'='*60}")
        logger.info("GPU RESOURCE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"GPU: {gpu_stats['name']}")
        logger.info(f"GPU Utilization: {gpu_stats['utilization_percent']}%")
        logger.info(f"GPU Memory: {gpu_stats['used_memory_gb']:.1f} GB / {gpu_stats['total_memory_gb']:.1f} GB used")
        logger.info(f"GPU Memory Free: {gpu_stats['free_memory_gb']:.1f} GB")
        if gpu_stats['temperature_c']:
            logger.info(f"GPU Temperature: {gpu_stats['temperature_c']}°C")
        
        # Calculate optimal workers based on GPU resources
        optimal_workers = calculate_optimal_workers(gpu_stats)
        logger.info(f"\nRecommended workers based on GPU: {optimal_workers}")
        
        if num_workers == 2:  # Default value
            logger.info(f"Using recommended worker count: {optimal_workers}")
            num_workers = optimal_workers
        else:
            logger.info(f"Using user-specified worker count: {num_workers}")
            if num_workers > optimal_workers:
                logger.warning(f"Warning: Specified workers ({num_workers}) exceeds recommended ({optimal_workers})")
                logger.warning(f"This may cause GPU memory issues. Consider using {optimal_workers} workers.")
        logger.info(f"{'='*60}\n")
    else:
        logger.info("GPU monitoring not available - using CPU-based limits")
    
    # Limit number of workers to avoid memory issues
    # Each worker loads full Surya OCR models which are memory-intensive
    # num_workers is already set with default value, but ensure it's reasonable
    if num_workers > cpu_count():
        logger.warning(f"num_workers ({num_workers}) exceeds CPU count ({cpu_count()}), limiting to {cpu_count()}")
        num_workers = cpu_count()
    
    logger.info(f"Using {num_workers} worker process(es) with batch size of {batch_size}")
    logger.info("Note: Models will be loaded once per worker process (memory-intensive)")
    if num_workers > 1:
        logger.warning(f"Memory usage: ~{num_workers} workers × model size. Consider num_workers=1 if memory issues occur.\n")
    else:
        logger.info("Sequential processing (memory-efficient)\n")
    
    # Collect all images to process
    all_image_tasks = []
    
    for pdf_dir in pdf_dirs:
        pdf_name = pdf_dir.name
        logger.info(f"Collecting images from: {pdf_name}")
        
        # Find all images in this PDF directory
        image_files = sorted(pdf_dir.glob("*.jpg"))
        
        if not image_files:
            logger.warning(f"No images found in {pdf_dir}")
            continue
        
        # Create output directory for this PDF
        pdf_output_dir = output_base_path / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add tasks for each image
        for image_path in image_files:
            text_output_path = pdf_output_dir / f"{image_path.stem}.txt"
            all_image_tasks.append((image_path, text_output_path))
    
    if not all_image_tasks:
        logger.warning("No images found to process")
        return
    
    logger.info(f"\nTotal images to process: {len(all_image_tasks)}")
    logger.info(f"Processing in batches of {batch_size} with {num_workers} workers\n")
    
    # Split tasks into batches
    batches = []
    for i in range(0, len(all_image_tasks), batch_size):
        batch = all_image_tasks[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batch(es)\n")
    
    # Process batches using multiprocessing
    total_processed = 0
    total_successful = 0
    
    if num_workers == 1:
        # Sequential processing (memory-efficient)
        logger.info("Processing batches sequentially (single worker)...")
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)} ({len(batch)} images)...")
            success_count, total_count = process_batch(batch)
            total_processed += total_count
            total_successful += success_count
            logger.info(f"Batch {i} complete: {success_count}/{total_count} successful\n")
    else:
        # Parallel processing with limited workers
        logger.info(f"Processing batches in parallel with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            # Process batches in parallel
            results = pool.map(process_batch, batches)
            
            # Aggregate results
            for success_count, total_count in results:
                total_processed += total_count
                total_successful += success_count
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format time in a human-readable way
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OCR PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total images processed: {total_processed}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_processed - total_successful}")
    logger.info(f"Total time taken: {time_str} ({elapsed_time:.2f} seconds)")
    if total_processed > 0:
        avg_time_per_image = elapsed_time / total_processed
        logger.info(f"Average time per image: {avg_time_per_image:.2f} seconds")
    logger.info(f"Output location: {output_base_path}/")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    split_dir = sys.argv[1] if len(sys.argv) > 1 else "voter split"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ocr_results"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # Reduced default
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # Reduced default
    
    # Special case: if num_workers is 0, use sequential processing
    if num_workers == 0:
        num_workers = 1
    
    # Get GPU stats for display
    gpu_stats = get_gpu_stats()
    
    print("\n" + "="*70)
    print("OCR Processing for Split Voter Images")
    print("Using Surya OCR with Multiprocessing and Batch Processing")
    print("="*70)
    
    if gpu_stats:
        print(f"\nGPU: {gpu_stats['name']}")
        print(f"GPU Utilization: {gpu_stats['utilization_percent']}%")
        print(f"GPU Memory: {gpu_stats['used_memory_gb']:.1f} GB / {gpu_stats['total_memory_gb']:.1f} GB used")
        print(f"GPU Memory Free: {gpu_stats['free_memory_gb']:.1f} GB")
        if gpu_stats['temperature_c']:
            print(f"GPU Temperature: {gpu_stats['temperature_c']}°C")
        
        optimal_workers = calculate_optimal_workers(gpu_stats)
        print(f"\nRecommended workers: {optimal_workers}")
        if num_workers == 2:  # Default
            print(f"Using recommended worker count: {optimal_workers}")
            num_workers = optimal_workers
    else:
        print("\nGPU monitoring not available")
    
    print(f"\nConfiguration: batch_size={batch_size}, num_workers={num_workers}")
    if num_workers == 1:
        print("Mode: Sequential processing (memory-efficient)")
    else:
        print(f"Mode: Parallel processing with {num_workers} workers")
        print("Note: If you encounter memory errors, use num_workers=1 for sequential processing")
    print("="*70 + "\n")
    
    # Process all images
    process_all_images(split_dir, output_dir, batch_size=batch_size, num_workers=num_workers)
