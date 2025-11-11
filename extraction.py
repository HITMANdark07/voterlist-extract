#!/usr/bin/env python3
"""
Unified Extraction Pipeline
===========================
Combined pipeline that:
1. Extracts voter images from PDFs
2. Cleans voter images (removes borders and boxes)
3. Runs OCR on cleaned images
All processing is done in memory without saving intermediate images.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import time
import os
import json

# Import modules
from config import INPUT_DIR, OUTPUT_DIR, USE_MULTIPROCESSING, MAX_WORKERS
from pdf_converter import get_all_pages, get_single_page, get_page_range
from grid_detector import detect_voter_blocks

# Surya OCR imports
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# Try to import nvidia-ml-py for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODULE 1: PDF EXTRACTION
# ============================================================================

def detect_page_contour(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the main content area of a page using contour detection.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (x, y, width, height) bounding box or None if detection fails
    """
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 20
        height, width = image.size[1], image.size[0]
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        return (x, y, w, h)
        
    except Exception as e:
        logger.error(f"Error detecting page contour: {e}")
        return None


def crop_page(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop the page image to the specified bounding box.
    
    Args:
        image: PIL Image object
        bbox: Tuple of (x, y, width, height)
        
    Returns:
        Cropped PIL Image
    """
    x, y, w, h = bbox
    width, height = image.size
    
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    x2 = max(0, min(x + w, width))
    y2 = max(0, min(y + h, height))
    
    return image.crop((x, y, x2, y2))


def extract_voters_from_pdf(pdf_path: str) -> Tuple[List[Image.Image], Optional[Image.Image]]:
    """
    Extract voter images from PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (list of voter images, optional page_1 image)
    """
    try:
        pdf_name = Path(pdf_path).stem
        logger.info(f"Extracting voters from {pdf_name}")
        
        # Get all pages to determine total page count
        all_pages = get_all_pages(pdf_path)
        if not all_pages:
            logger.error(f"Could not load pages from {pdf_path}")
            return [], None
        
        total_pages = len(all_pages)
        logger.info(f"PDF {pdf_name} has {total_pages} pages")
        
        voter_images = []
        page_1_image = None
        
        # Process page 1 (optional - for reference)
        if total_pages > 0:
            page_1 = get_single_page(pdf_path, 1)
            if page_1:
                bbox = detect_page_contour(page_1)
                if bbox:
                    page_1_image = crop_page(page_1, bbox)
                else:
                    page_1_image = page_1
        
        # Process pages 3 to n-1 for voter grids
        if total_pages >= 3:
            start_page = 3
            end_page = total_pages
            
            page_images = get_page_range(pdf_path, start_page, end_page)
            
            for page_num, page_image in page_images:
                logger.info(f"Processing page {page_num}")
                
                grids = detect_voter_blocks(page_image)
                
                if not grids:
                    logger.warning(f"No grids detected on page {page_num}")
                    continue
                
                logger.info(f"Detected {len(grids)} grids on page {page_num}")
                voter_images.extend(grids)
        
        logger.info(f"Extracted {len(voter_images)} voter images from {pdf_name}")
        return voter_images, page_1_image
        
    except Exception as e:
        logger.error(f"Error extracting voters from PDF {pdf_path}: {e}", exc_info=True)
        return [], None


# ============================================================================
# MODULE 2: IMAGE CLEANING
# ============================================================================

def detect_outer_border(image: Image.Image, 
                        border_threshold: float = 0.85) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the outer border rectangle of the image.
    
    Args:
        image: PIL Image object
        border_threshold: Minimum area threshold to consider as outer border
        
    Returns:
        Tuple of (x, y, width, height) of the outer border, or None if not found
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = img.shape
    img_area = img_h * img_w
    
    outer_border = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if area > border_threshold * img_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4 and area > max_area:
                max_area = area
                outer_border = (x, y, w, h)
    
    return outer_border


def remove_outer_border(image: Image.Image, 
                       border_threshold: float = 0.85,
                       padding: int = 5) -> Image.Image:
    """
    Remove the outer border rectangle from the image by cropping inside it.
    
    Args:
        image: PIL Image object
        border_threshold: Minimum area threshold to consider as outer border
        padding: Padding to add inside the border when cropping
    
    Returns:
        Image with outer border removed (cropped)
    """
    outer_border = detect_outer_border(image, border_threshold)
    
    if outer_border is None:
        return image
    
    x, y, w, h = outer_border
    img_width, img_height = image.size
    
    crop_x1 = max(0, x + padding)
    crop_y1 = max(0, y + padding)
    crop_x2 = min(img_width, x + w - padding)
    crop_y2 = min(img_height, y + h - padding)
    
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image
    
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def find_inner_boxes(image: Image.Image, 
                     min_box_width: int = 20, 
                     min_box_height: int = 20, 
                     border_threshold: float = 0.9) -> List[List[int]]:
    """
    Detect all boxes from structured form-like images.
    Automatically ignores the largest outer border box.
    
    Args:
        image: PIL Image object
        min_box_width: Minimum width of valid boxes
        min_box_height: Minimum height of valid boxes
        border_threshold: Ignore boxes covering > threshold * image area
    
    Returns:
        List of detected box coordinates [x1, y1, x2, y2]
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_h, img_w = img.shape
    img_area = img_h * img_w
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if w < min_box_width or h < min_box_height:
            continue
        
        if area > border_threshold * img_area:
            continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            boxes.append([x, y, x + w, y + h])
    
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def remove_boxes_from_image(image: Image.Image, 
                           boxes: List[List[int]],
                           fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Remove detected boxes from the image by filling them with background color.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2] to remove
        fill_color: Color to fill the boxes with (default: white)
    
    Returns:
        Image with boxes removed
    """
    img_array = np.array(image).copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_array.shape[1], x2 + padding)
        y2 = min(img_array.shape[0], y2 + padding)
        
        if len(img_array.shape) == 2:
            img_array[y1:y2, x1:x2] = 255
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 3:
                img_array[y1:y2, x1:x2] = fill_color[:3]
            elif img_array.shape[2] == 4:
                img_array[y1:y2, x1:x2] = (*fill_color[:3], 255)
            else:
                img_array[y1:y2, x1:x2] = 255
    
    return Image.fromarray(img_array)


def clean_voter_image(image: Image.Image,
                     min_box_width: int = 20,
                     min_box_height: int = 20) -> Image.Image:
    """
    Clean a single voter image by removing outer border and inner boxes.
    
    Args:
        image: PIL Image object (voter image)
        min_box_width: Minimum width for boxes to detect
        min_box_height: Minimum height for boxes to detect
    
    Returns:
        Cleaned PIL Image
    """
    # Step 1: Remove outer border
    image = remove_outer_border(image, border_threshold=0.85, padding=5)
    
    # Step 2: Find inner boxes
    inner_boxes = find_inner_boxes(
        image,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        border_threshold=0.9
    )
    
    # Step 3: Remove inner boxes
    cleaned_image = remove_boxes_from_image(image, inner_boxes)
    
    return cleaned_image


def clean_voter_images(voter_images: List[Image.Image]) -> List[Image.Image]:
    """
    Clean a list of voter images.
    
    Args:
        voter_images: List of PIL Image objects
    
    Returns:
        List of cleaned PIL Image objects
    """
    cleaned_images = []
    
    for idx, voter_image in enumerate(voter_images):
        try:
            cleaned = clean_voter_image(voter_image)
            cleaned_images.append(cleaned)
            if (idx + 1) % 10 == 0:
                logger.info(f"Cleaned {idx + 1}/{len(voter_images)} voter images")
        except Exception as e:
            logger.error(f"Error cleaning voter image {idx + 1}: {e}")
            # Keep original if cleaning fails
            cleaned_images.append(voter_image)
    
    logger.info(f"Cleaned {len(cleaned_images)} voter images")
    return cleaned_images


# ============================================================================
# MODULE 3: OCR PROCESSING
# ============================================================================

# Global predictors (loaded once per process in multiprocessing)
_foundation = None
_detection_predictor = None
_recognition_predictor = None


def load_predictors():
    """
    Load Surya OCR predictors once per process (reused for all images in a process).
    In multiprocessing, each worker process gets its own instance.
    """
    global _foundation, _detection_predictor, _recognition_predictor
    
    if _foundation is None or _detection_predictor is None or _recognition_predictor is None:
        logger.info(f"Loading Surya OCR predictors for process {os.getpid()}...")
        try:
            _foundation = FoundationPredictor()
            _detection_predictor = DetectionPredictor()
            _recognition_predictor = RecognitionPredictor(_foundation)
            logger.info(f"Predictors loaded successfully for process {os.getpid()}")
        except Exception as e:
            logger.error(f"Error loading predictors in process {os.getpid()}: {e}", exc_info=True)
            raise
    
    return _foundation, _detection_predictor, _recognition_predictor


def ocr_image(image: Image.Image, 
              recognition_predictor: RecognitionPredictor,
              detection_predictor: DetectionPredictor) -> str:
    """
    Run OCR on a single image and extract text.
    
    Args:
        image: PIL Image object
        recognition_predictor: Pre-loaded RecognitionPredictor instance
        detection_predictor: Pre-loaded DetectionPredictor instance
    
    Returns:
        Extracted text as string
    """
    try:
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        
        all_text_lines = []
        
        for page_pred in predictions:
            for line in page_pred.text_lines:
                text = line.text
                if text:
                    all_text_lines.append(text)
        
        return "\n".join(all_text_lines)
        
    except TypeError as e:
        if "BFloat16" in str(e):
            logger.warning(f"BFloat16 conversion error (this is a known Surya library issue): {e}")
            logger.info("Attempting to work around by converting model outputs...")
            # Try to work around by manually handling the conversion
            try:
                import torch
                # Force float32 for model outputs
                with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
                    predictions = recognition_predictor([image], det_predictor=detection_predictor)
                    all_text_lines = []
                    for page_pred in predictions:
                        for line in page_pred.text_lines:
                            text = line.text
                            if text:
                                all_text_lines.append(text)
                    return "\n".join(all_text_lines)
            except Exception as e2:
                logger.error(f"Workaround failed: {e2}")
                return ""
        else:
            logger.error(f"TypeError processing image with OCR: {e}", exc_info=True)
            return ""
    except Exception as e:
        logger.error(f"Error processing image with OCR: {e}", exc_info=True)
        return ""


def process_ocr_batch(image_batch: List[Tuple[Path, Path]]) -> Tuple[int, int]:
    """
    Process a batch of images with OCR in a single worker process.
    Images are loaded from disk (like ocr_split_images.py) to avoid memory issues.
    
    Args:
        image_batch: List of tuples (image_path, output_text_path)
    
    Returns:
        Tuple of (success_count, total_count)
    """
    foundation, detection_predictor, recognition_predictor = load_predictors()
    
    success_count = 0
    total_count = len(image_batch)
    
    for image_path, output_text_path in image_batch:
        try:
            # Load image from disk
            image = Image.open(image_path)
            
            # Run OCR
            text = ocr_image(image, recognition_predictor, detection_predictor)
            
            # Save text result
            output_text_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            success_count += 1
            logger.info(f"Processed: {image_path.name}")
            
            # Cleanup
            del image
            
        except Exception as e:
            logger.error(f"Failed: {image_path.name} - {e}")
    
    # Force garbage collection and clear GPU cache after batch
    import gc
    gc.collect()
    
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    
    return (success_count, total_count)


def process_ocr_images_from_paths(cleaned_image_paths: List[Path],
                                  output_dir: Path,
                                  batch_size: int = 5,
                                  num_workers: int = 2) -> Tuple[int, int]:
    """
    Process all cleaned images with OCR using multiprocessing (like ocr_split_images.py).
    Images are loaded from disk to avoid memory issues.
    
    Args:
        cleaned_image_paths: List of paths to cleaned image files
        output_dir: Directory to save OCR text results
        batch_size: Number of images to process per batch
        num_workers: Number of worker processes (default: 2, will be optimized based on GPU)
    
    Returns:
        Tuple of (success_count, total_count)
    """
    if not cleaned_image_paths:
        return (0, 0)
    
    logger.info(f"Processing {len(cleaned_image_paths)} images with OCR")
    
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
        optimal_workers = calculate_optimal_workers(gpu_stats, estimated_memory_per_worker_gb=2.0)
        logger.info(f"\nRecommended workers based on GPU: {optimal_workers}")
        
        if num_workers == 2:  # Default value
            logger.info(f"Using recommended worker count: {optimal_workers}")
            num_workers = optimal_workers
        else:
            logger.info(f"Using user-specified worker count: {num_workers}")
            if num_workers > optimal_workers:
                logger.warning(f"Warning: Specified workers ({num_workers}) exceeds recommended ({optimal_workers})")
                logger.warning(f"This may cause GPU memory issues. Limiting to recommended: {optimal_workers}")
                num_workers = optimal_workers
        logger.info(f"{'='*60}\n")
    else:
        logger.info("GPU monitoring not available - using CPU-based limits")
        if num_workers > cpu_count():
            num_workers = cpu_count()
    
    # Limit to CPU count only
    if num_workers > cpu_count():
        logger.warning(f"num_workers ({num_workers}) exceeds CPU count ({cpu_count()}), limiting to {cpu_count()}")
        num_workers = cpu_count()
    
    logger.info(f"Using {num_workers} worker process(es) with batch size of {batch_size}")
    logger.info("Note: Models will be loaded once per worker process (memory-intensive)")
    if num_workers > 1:
        estimated_memory = num_workers * 2.0  # 2.0 GB per worker estimate
        logger.info(f"Estimated GPU memory usage: ~{estimated_memory:.1f} GB ({num_workers} workers × 2.0 GB)")
        logger.info(f"Images loaded from disk to avoid memory buildup.\n")
    else:
        logger.info("Sequential processing (memory-efficient)\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare image tasks: (image_path, output_text_path)
    all_image_tasks = []
    for idx, image_path in enumerate(cleaned_image_paths, 1):
        text_output_path = output_dir / f"{image_path.stem}.txt"
        all_image_tasks.append((image_path, text_output_path))
    
    # Split tasks into batches
    batches = []
    for i in range(0, len(all_image_tasks), batch_size):
        batch = all_image_tasks[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batch(es) for OCR processing")
    
    total_processed = 0
    total_successful = 0
    
    if num_workers == 1:
        # Sequential processing (memory-efficient)
        logger.info("Processing batches sequentially (single worker)...")
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)} ({len(batch)} images)...")
            success_count, total_count = process_ocr_batch(batch)
            total_processed += total_count
            total_successful += success_count
            logger.info(f"Batch {i} complete: {success_count}/{total_count} successful\n")
    else:
        # Parallel processing with multiprocessing Pool
        # Reuse Pool across all batches to avoid reloading models
        logger.info(f"Processing batches in parallel with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            # Process batches in parallel
            results = pool.map(process_ocr_batch, batches)
            
            # Aggregate results
            for success_count, total_count in results:
                total_processed += total_count
                total_successful += success_count
    
    logger.info(f"OCR processing complete: {total_successful}/{total_processed} successful")
    return (total_successful, total_processed)


# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_gpu_stats() -> Optional[Dict]:
    """Get GPU statistics using NVIDIA Management Library."""
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            pynvml.nvmlShutdown()
            return None
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem_gb = mem_info.total / (1024**3)
        used_mem_gb = mem_info.used / (1024**3)
        free_mem_gb = mem_info.free / (1024**3)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        
        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        
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
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        return None


def calculate_optimal_workers(gpu_stats: Optional[Dict] = None, 
                              estimated_memory_per_worker_gb: float = 2.0) -> int:
    """
    Calculate optimal number of workers based on GPU memory and CPU count.
    Optimized for better GPU utilization while maintaining safety.
    
    Args:
        gpu_stats: GPU statistics dictionary from get_gpu_stats()
        estimated_memory_per_worker_gb: Estimated GPU memory per worker in GB (default: 2.0 GB)
    
    Returns:
        Recommended number of workers (safely capped)
    """
    max_cpu_workers = cpu_count()
    
    # No hard cap - let GPU memory determine optimal workers
    
    if gpu_stats is None:
        # No GPU info available, use conservative CPU-based estimate
        logger.info("GPU stats not available, using CPU-based worker calculation")
        return min(2, max_cpu_workers)
    
    free_mem_gb = gpu_stats['free_memory_gb']
    gpu_util = gpu_stats['utilization_percent']
    total_mem_gb = gpu_stats['total_memory_gb']
    
    # Safety check: Need at least 2 GB free memory to use multiple workers
    if free_mem_gb < 2.0:
        logger.warning(f"Low GPU memory ({free_mem_gb:.1f} GB free). Using single worker for safety.")
        return 1
    
    # Calculate workers based on available GPU memory
    # Reserve 20% of free memory as buffer
    usable_memory_gb = free_mem_gb * 0.8
    memory_based_workers = int(usable_memory_gb / estimated_memory_per_worker_gb)
    
    # Consider GPU utilization - allow more workers when GPU is underutilized
    if gpu_util > 80:
        # GPU is heavily utilized, be conservative
        utilization_factor = 0.5
    elif gpu_util > 60:
        # GPU is moderately utilized
        utilization_factor = 0.7
    else:
        # GPU has capacity - use full potential
        utilization_factor = 1.0  # Use full capacity when GPU is available
    
    memory_based_workers = max(1, int(memory_based_workers * utilization_factor))
    
    # Take minimum of CPU-based and memory-based (no hard cap)
    optimal_workers = min(memory_based_workers, max_cpu_workers)
    
    # Ensure at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    return optimal_workers


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_pdf(pdf_path: str) -> Dict:
    """
    Phase 1: Extract, clean, and save images (NO OCR).
    This separates preprocessing from OCR to prevent memory issues.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary with preprocessing statistics
    """
    start_time = time.time()
    pdf_name = Path(pdf_path).stem
    
    logger.info(f"\n[PREPROCESS] Processing PDF: {pdf_name}")
    
    # Step 1: Extract voters from PDF
    voter_images, page_1_image = extract_voters_from_pdf(pdf_path)
    
    # Step 2: Clean voter images
    cleaned_images = []
    if voter_images:
        logger.info(f"[PREPROCESS] Extracted {len(voter_images)} voter images")
        cleaned_images = clean_voter_images(voter_images)
        logger.info(f"[PREPROCESS] Cleaned {len(cleaned_images)} voter images")
    
    # Step 3: Save cleaned images and page 1 to clean_preprocessed directory
    cleaned_dir = Path("clean_preprocessed") / pdf_name
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Save page 1 to clean_preprocessed directory
    page_1_saved = False
    if page_1_image:
        page_1_path = cleaned_dir / "page_1.jpg"
        page_1_image.save(page_1_path, "JPEG", quality=95)
        page_1_saved = True
        logger.info(f"[PREPROCESS] Saved page 1 to: {page_1_path}")
    
    if not cleaned_images:
        logger.warning(f"[PREPROCESS] No voter images extracted from {pdf_name}")
        return {
            'pdf_name': pdf_name,
            'pdf_path': pdf_path,
            'total_voters': 0,
            'cleaned': 0,
            'page_1_saved': page_1_saved,
            'preprocess_success': page_1_saved,  # Success if at least page 1 was saved
            'cleaned_dir': str(cleaned_dir),
            'time_taken': time.time() - start_time
        }
    
    cleaned_image_paths = []
    for idx, cleaned_img in enumerate(cleaned_images, 1):
        img_path = cleaned_dir / f"voter_{idx:03d}.jpg"
        cleaned_img.save(img_path, "JPEG", quality=95)
        cleaned_image_paths.append(img_path)
    
    logger.info(f"[PREPROCESS] Saved {len(cleaned_image_paths)} cleaned images to {cleaned_dir}")
    
    # Cleanup in-memory images
    del voter_images, cleaned_images, page_1_image
    import gc
    gc.collect()
    
    elapsed_time = time.time() - start_time
    logger.info(f"[PREPROCESS] Complete for {pdf_name} in {elapsed_time:.2f} seconds")
    
    return {
        'pdf_name': pdf_name,
        'pdf_path': pdf_path,
        'total_voters': len(cleaned_image_paths),
        'cleaned': len(cleaned_image_paths),
        'page_1_saved': page_1_saved,
        'cleaned_dir': str(cleaned_dir),
        'preprocess_success': True,
        'time_taken': elapsed_time
    }


def ocr_preprocessed_pdf(pdf_name: str,
                         cleaned_dir: str,
                         output_dir: str,
                         batch_size: int = 5,
                         num_workers: int = 2,
                         cleanup_cleaned_images: bool = False) -> Dict:
    """
    Phase 2: Run OCR on preprocessed images (loads from disk).
    This runs after all preprocessing is complete to prevent memory overlap.
    
    Args:
        pdf_name: Name of the PDF (without extension)
        cleaned_dir: Directory containing cleaned images
        output_dir: Directory to save OCR results
        batch_size: Number of images per OCR batch
        num_workers: Number of OCR worker processes
        cleanup_cleaned_images: Whether to delete cleaned images after OCR
    
    Returns:
        Dictionary with OCR statistics
    """
    start_time = time.time()
    
    logger.info(f"\n[OCR] Processing OCR for: {pdf_name}")
    
    # Find all cleaned images
    cleaned_dir_path = Path(cleaned_dir)
    if not cleaned_dir_path.exists():
        logger.error(f"[OCR] Cleaned directory not found: {cleaned_dir}")
        return {
            'pdf_name': pdf_name,
            'ocr_successful': 0,
            'ocr_failed': 0,
            'page_1_ocr_successful': False,
            'time_taken': time.time() - start_time
        }
    
    cleaned_image_paths = sorted(cleaned_dir_path.glob("*.jpg"))
    
    if not cleaned_image_paths:
        logger.warning(f"[OCR] No cleaned images found in {cleaned_dir}")
        return {
            'pdf_name': pdf_name,
            'ocr_successful': 0,
            'ocr_failed': 0,
            'page_1_ocr_successful': False,
            'time_taken': time.time() - start_time
        }
    
    logger.info(f"[OCR] Found {len(cleaned_image_paths)} images to process")
    
    # Process page 1 OCR if it exists (load from clean_preprocessed directory)
    page_1_ocr_successful = False
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_output_dir = output_path / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Page 1 is in the cleaned_dir, not ocr_results
    cleaned_dir_path_obj = Path(cleaned_dir)
    page_1_path = cleaned_dir_path_obj / "page_1.jpg"
    if page_1_path.exists():
        try:
            logger.info("[OCR] Running OCR on page 1...")
            foundation, detection_predictor, recognition_predictor = load_predictors()
            page_1_image = Image.open(page_1_path)
            page_1_text = ocr_image(page_1_image, recognition_predictor, detection_predictor)
            
            page_1_text_path = pdf_output_dir / "page_1.txt"
            with open(page_1_text_path, "w", encoding="utf-8") as f:
                f.write(page_1_text)
            
            page_1_ocr_successful = True
            logger.info(f"[OCR] Page 1 OCR complete: {len(page_1_text)} characters")
            del page_1_image
        except Exception as e:
            logger.error(f"[OCR] Error running OCR on page 1: {e}", exc_info=True)
    
    # Run OCR on voter images
    successful, total = process_ocr_images_from_paths(
        cleaned_image_paths,
        pdf_output_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    failed = total - successful
    
    # Optional: Cleanup cleaned images after OCR
    if cleanup_cleaned_images:
        logger.info(f"[OCR] Cleaning up cleaned images from {cleaned_dir}...")
        for img_path in cleaned_image_paths:
            try:
                img_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {img_path}: {e}")
        
        try:
            cleaned_dir_path.rmdir()
        except:
            pass
        
        logger.info("[OCR] Cleaned images removed")
    
    elapsed_time = time.time() - start_time
    logger.info(f"[OCR] Complete for {pdf_name}: {successful}/{total} successful in {elapsed_time:.2f} seconds")
    
    return {
        'pdf_name': pdf_name,
        'ocr_successful': successful,
        'ocr_failed': failed,
        'page_1_ocr_successful': page_1_ocr_successful,
        'time_taken': elapsed_time
    }


def process_pdf_pipeline(pdf_path: str,
                        output_dir: str = "ocr_results",
                        batch_size: int = 5,
                        num_workers: int = 2,
                        cleanup_cleaned_images: bool = False) -> Dict:
    """
    Complete pipeline: Extract -> Clean -> Save -> OCR for a single PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save OCR results
        batch_size: Number of images per OCR batch
        num_workers: Number of OCR worker processes
        cleanup_cleaned_images: Whether to delete cleaned images after OCR (saves disk space)
    
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    pdf_name = Path(pdf_path).stem
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing PDF: {pdf_name}")
    logger.info(f"{'='*70}")
    
    # Step 1: Extract voters from PDF
    logger.info("\n[Step 1] Extracting voters from PDF...")
    voter_images, page_1_image = extract_voters_from_pdf(pdf_path)
    
    # Step 1.5: Save page 1 to clean_preprocessed directory (not ocr_results)
    page_1_saved = False
    page_1_ocr_successful = False
    
    # Save page 1 to clean_preprocessed directory (will be processed in OCR phase)
    cleaned_dir = Path("clean_preprocessed") / pdf_name
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    if page_1_image:
        page_1_path = cleaned_dir / "page_1.jpg"
        page_1_image.save(page_1_path, "JPEG", quality=95)
        page_1_saved = True
        logger.info(f"Saved page 1 to: {page_1_path}")
    else:
        logger.warning("Page 1 image not available")
    
    if not voter_images:
        logger.warning(f"No voter images extracted from {pdf_name}")
        return {
            'pdf_name': pdf_name,
            'total_voters': 0,
            'cleaned': 0,
            'ocr_successful': 0,
            'ocr_failed': 0,
            'page_1_saved': page_1_saved,
            'page_1_ocr_successful': page_1_ocr_successful,
            'time_taken': time.time() - start_time
        }
    
    logger.info(f"Extracted {len(voter_images)} voter images")
    
    # Step 2: Clean voter images
    logger.info("\n[Step 2] Cleaning voter images...")
    cleaned_images = clean_voter_images(voter_images)
    
    logger.info(f"Cleaned {len(cleaned_images)} voter images")
    
    # Step 2.5: Save cleaned images to disk (required for OCR to avoid memory issues)
    cleaned_dir = Path("clean_preprocessed") / pdf_name
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n[Step 2.5] Saving cleaned images to disk...")
    cleaned_image_paths = []
    for idx, cleaned_img in enumerate(cleaned_images, 1):
        img_path = cleaned_dir / f"voter_{idx:03d}.jpg"
        cleaned_img.save(img_path, "JPEG", quality=95)
        cleaned_image_paths.append(img_path)
    
    logger.info(f"Saved {len(cleaned_image_paths)} cleaned images to {cleaned_dir}")
    
    # Step 3: Run OCR on saved images (loads from disk to avoid memory issues)
    logger.info("\n[Step 3] Running OCR on cleaned images (loading from disk)...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_output_dir = output_path / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    successful, total = process_ocr_images_from_paths(
        cleaned_image_paths,
        pdf_output_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    failed = total - successful
    
    # Optional: Cleanup cleaned images after OCR to save disk space
    if cleanup_cleaned_images:
        logger.info(f"\nCleaning up cleaned images from {cleaned_dir}...")
        for img_path in cleaned_image_paths:
            try:
                img_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {img_path}: {e}")
        
        # Remove directory if empty
        try:
            cleaned_dir.rmdir()
        except:
            pass
        
        logger.info("Cleaned images removed to save disk space")
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing complete for {pdf_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Page 1 saved: {page_1_saved}")
    logger.info(f"Page 1 OCR: {'Success' if page_1_ocr_successful else 'Failed'}")
    logger.info(f"Total voters extracted: {len(voter_images)}")
    logger.info(f"Voters cleaned: {len(cleaned_images)}")
    logger.info(f"OCR successful: {successful}")
    logger.info(f"OCR failed: {failed}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to: {pdf_output_dir}")
    logger.info(f"{'='*70}\n")
    
    return {
        'pdf_name': pdf_name,
        'page_1_saved': page_1_saved,
        'page_1_ocr_successful': page_1_ocr_successful,
        'total_voters': len(voter_images),
        'cleaned': len(cleaned_images),
        'ocr_successful': successful,
        'ocr_failed': failed,
        'time_taken': elapsed_time
    }


def preprocess_pdf_wrapper(pdf_path: str) -> Dict:
    """Wrapper function for preprocessing multiprocessing."""
    return preprocess_pdf(pdf_path)


def ocr_preprocessed_pdf_wrapper(args: Tuple[str, str, str, int, int, bool]) -> Dict:
    """Wrapper function for OCR multiprocessing."""
    pdf_name, cleaned_dir, output_dir, batch_size, num_workers, cleanup_cleaned_images = args
    return ocr_preprocessed_pdf(pdf_name, cleaned_dir, output_dir, batch_size, num_workers, cleanup_cleaned_images)


def main():
    """Main execution function."""
    # Record start time for total execution
    total_start_time = time.time()
    
    print("\n" + "="*70)
    print("Unified Extraction Pipeline")
    print("Extract -> Clean -> OCR")
    print("="*70 + "\n")
    
    # Setup directories
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n[ERROR] No PDF files found in '{INPUT_DIR}' directory.")
        print(f"Please place your PDF files in the '{INPUT_DIR}' folder and run again.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Configuration
    output_dir = "ocr_results"
    batch_size = 5
    num_workers = 2  # Will be optimized based on GPU stats
    
    # Get GPU stats for display
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print(f"GPU: {gpu_stats['name']}")
        print(f"GPU Utilization: {gpu_stats['utilization_percent']}%")
        print(f"GPU Memory: {gpu_stats['used_memory_gb']:.1f} GB / {gpu_stats['total_memory_gb']:.1f} GB used")
        print(f"GPU Memory Free: {gpu_stats['free_memory_gb']:.1f} GB")
        if gpu_stats['temperature_c']:
            print(f"GPU Temperature: {gpu_stats['temperature_c']}°C")
        
        optimal_workers = calculate_optimal_workers(gpu_stats)
        print(f"\nRecommended OCR workers: {optimal_workers}")
        if num_workers == 2:  # Default
            num_workers = optimal_workers
        print()
    
    # Configuration for batch processing
    PDF_BATCH_SIZE = 10  # Process 10 PDFs at a time to avoid too many processes
    cleanup_cleaned_images = False  # Set to True to delete cleaned images after OCR
    
    # Process PDFs in batches with separated preprocessing and OCR phases
    if USE_MULTIPROCESSING and len(pdf_files) > 1:
        num_pdf_workers = min(MAX_WORKERS, len(pdf_files), PDF_BATCH_SIZE)
        total_batches = (len(pdf_files) + PDF_BATCH_SIZE - 1) // PDF_BATCH_SIZE
        
        print(f"Processing {len(pdf_files)} PDFs in {total_batches} batch(es) of up to {PDF_BATCH_SIZE} PDFs each")
        print(f"Using {num_pdf_workers} workers for preprocessing")
        print(f"OCR processing will use {num_workers} workers per PDF")
        print(f"\nNOTE: Preprocessing and OCR are separated to prevent memory issues")
        print(f"      Phase 1: Preprocess all PDFs in batch (extract, clean, save)")
        print(f"      Phase 2: OCR all PDFs in batch (load from disk)\n")
        
        successful = 0
        
        for batch_idx in range(0, len(pdf_files), PDF_BATCH_SIZE):
            batch_pdfs = pdf_files[batch_idx:batch_idx + PDF_BATCH_SIZE]
            batch_num = (batch_idx // PDF_BATCH_SIZE) + 1
            
            print(f"\n{'='*70}")
            print(f"Processing PDF batch {batch_num}/{total_batches} ({len(batch_pdfs)} PDFs)")
            print(f"{'='*70}\n")
            
            # PHASE 1: Preprocessing (extract, clean, save) - NO OCR
            print(f"[PHASE 1] Preprocessing {len(batch_pdfs)} PDFs...")
            preprocess_results = []
            
            with ProcessPoolExecutor(max_workers=num_pdf_workers) as executor:
                futures = {
                    executor.submit(preprocess_pdf_wrapper, str(pdf_path)): pdf_path
                    for pdf_path in batch_pdfs
                }
                
                for future in as_completed(futures):
                    pdf_path = futures[future]
                    try:
                        result = future.result()
                        preprocess_results.append(result)
                        if result['preprocess_success']:
                            print(f"[PREPROCESS OK] {result['pdf_name']}: {result['cleaned']} images saved")
                        else:
                            print(f"[PREPROCESS FAILED] {result['pdf_name']}")
                    except Exception as e:
                        pdf_name = Path(pdf_path).stem
                        logger.error(f"Exception preprocessing {pdf_name}: {e}", exc_info=True)
                        print(f"[PREPROCESS ERROR] {pdf_name}: {e}")
            
            print(f"\n[PHASE 1] Complete: {len([r for r in preprocess_results if r['preprocess_success']])}/{len(preprocess_results)} successful\n")
            
            # PHASE 2: OCR (load from disk) - NO image processing
            print(f"[PHASE 2] Running OCR on {len(preprocess_results)} preprocessed PDFs...")
            
            # Prepare OCR tasks
            ocr_tasks = []
            for result in preprocess_results:
                if result['preprocess_success'] and result.get('cleaned_dir'):
                    ocr_tasks.append((
                        result['pdf_name'],
                        result['cleaned_dir'],
                        output_dir,
                        batch_size,
                        num_workers,
                        cleanup_cleaned_images
                    ))
            
            if ocr_tasks:
                # Run OCR sequentially or with limited parallelism to avoid GPU memory issues
                # Since OCR already uses multiprocessing internally, we run OCR tasks sequentially
                for task in ocr_tasks:
                    pdf_name, cleaned_dir, out_dir, bs, nw, cleanup = task
                    try:
                        ocr_result = ocr_preprocessed_pdf(pdf_name, cleaned_dir, out_dir, bs, nw, cleanup)
                        if ocr_result['ocr_successful'] > 0:
                            successful += 1
                        print(f"[OCR OK] {pdf_name}: {ocr_result['ocr_successful']}/{ocr_result['ocr_successful'] + ocr_result['ocr_failed']} successful")
                    except Exception as e:
                        logger.error(f"Exception in OCR for {pdf_name}: {e}", exc_info=True)
                        print(f"[OCR ERROR] {pdf_name}: {e}")
            
            print(f"\n[PHASE 2] Complete\n")
            print(f"Batch {batch_num} complete\n")
    else:
        # Process PDFs sequentially with separated phases
        print(f"Processing PDFs sequentially (preprocessing then OCR)\n")
        successful = 0
        
        for pdf_path in pdf_files:
            # Phase 1: Preprocess
            preprocess_result = preprocess_pdf(str(pdf_path))
            
            if preprocess_result['preprocess_success'] and preprocess_result.get('cleaned_dir'):
                # Phase 2: OCR
                ocr_result = ocr_preprocessed_pdf(
                    preprocess_result['pdf_name'],
                    preprocess_result['cleaned_dir'],
                    output_dir,
                    batch_size,
                    num_workers,
                    cleanup_cleaned_images
                )
                
                if ocr_result['ocr_successful'] > 0:
                    successful += 1
            print()
    
    # Calculate total execution time
    total_elapsed_time = time.time() - total_start_time
    total_minutes = int(total_elapsed_time // 60)
    total_seconds = total_elapsed_time % 60
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(pdf_files) - successful}")
    print(f"Results saved to: {output_dir}/")
    if total_minutes > 0:
        print(f"Total time taken: {total_minutes} minute(s) {total_seconds:.2f} seconds ({total_elapsed_time:.2f} seconds)")
    else:
        print(f"Total time taken: {total_seconds:.2f} seconds")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

