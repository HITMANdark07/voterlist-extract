#!/usr/bin/env python3
"""
OCR Processor Module
=====================
Processes all extracted images using PaddleOCRVL and saves OCR results
in the same directory structure.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional
from paddleocr import PaddleOCRVL

from config import OUTPUT_DIR

# Setup logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global OCR pipeline instance (loaded once)
_ocr_pipeline: Optional[PaddleOCRVL] = None


def get_ocr_pipeline(use_layout_detection: bool = False,
                     force_reload: bool = False) -> PaddleOCRVL:
    """
    Get or initialize the global OCR pipeline instance.
    The model is loaded only once and reused for all images.
    
    Args:
        use_layout_detection: Whether to use layout detection module
        use_doc_orientation_classify: Whether to use document orientation classification
        use_doc_unwarping: Whether to use text image correction module
        force_reload: Force reload even if pipeline already exists
        
    Returns:
        Initialized PaddleOCRVL pipeline (global singleton)
    """
    global _ocr_pipeline
    
    if _ocr_pipeline is None or force_reload:
        try:
            logger.info("Initializing PaddleOCRVL pipeline (loading models)...")
            _ocr_pipeline = PaddleOCRVL(
                use_layout_detection=use_layout_detection,
            )
            logger.info("[OK] PaddleOCRVL pipeline initialized successfully (models loaded)")
        except Exception as e:
            logger.error(f"Error initializing PaddleOCRVL pipeline: {e}", exc_info=True)
            raise
    else:
        logger.debug("Using existing OCR pipeline instance (models already loaded)")
    
    return _ocr_pipeline


def initialize_ocr_pipeline(use_layout_detection: bool = False,
                            ) -> PaddleOCRVL:
    """
    Initialize PaddleOCRVL pipeline with specified options.
    This is a wrapper around get_ocr_pipeline() for backward compatibility.
    
    Args:
        use_layout_detection: Whether to use layout detection module
        use_doc_orientation_classify: Whether to use document orientation classification
        use_doc_unwarping: Whether to use text image correction module
        
    Returns:
        Initialized PaddleOCRVL pipeline (global singleton)
    """
    return get_ocr_pipeline(
        use_layout_detection=use_layout_detection,
    )


def process_image(pipeline: PaddleOCRVL, image_path: Path, output_dir: Path) -> bool:
    """
    Process a single image with OCR and save results.
    
    Args:
        pipeline: PaddleOCRVL pipeline instance
        image_path: Path to the image file
        output_dir: Directory to save OCR results (same as image directory)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Run OCR
        output = pipeline.predict(str(image_path))
        
        if not output:
            logger.warning(f"No OCR results for {image_path}")
            return False
        
        # Process each result (usually one per image)
        for idx, res in enumerate(output):
            # Generate base filename from image name
            image_stem = image_path.stem
            
            # Save JSON result
            json_path = output_dir / f"{image_stem}.json"
            res.save_to_json(save_path=str(output_dir))
            logger.info(f"[OK] Saved JSON result: {json_path}")
            
            # Save Markdown result
            md_path = output_dir / f"{image_stem}.md"
            res.save_to_markdown(save_path=str(output_dir))
            logger.info(f"[OK] Saved Markdown result: {md_path}")
            
            # Print structured output (optional, for debugging)
            # res.print()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return False


def find_all_images(base_dir: Path) -> List[Path]:
    """
    Find all image files in the output directory structure.
    
    Args:
        base_dir: Base output directory (output_images)
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []
    
    if not base_dir.exists():
        logger.error(f"Output directory does not exist: {base_dir}")
        return images
    
    # Find all image files recursively
    for ext in image_extensions:
        images.extend(base_dir.rglob(f"*{ext}"))
        images.extend(base_dir.rglob(f"*{ext.upper()}"))
    
    # Sort for consistent processing order
    images.sort()
    
    logger.info(f"Found {len(images)} image(s) to process")
    return images


def process_all_images(pipeline: PaddleOCRVL, base_dir: Path = None) -> dict:
    """
    Process all images in the output directory structure.
    
    Args:
        pipeline: PaddleOCRVL pipeline instance
        base_dir: Base output directory (defaults to OUTPUT_DIR from config)
        
    Returns:
        Dictionary with processing statistics
    """
    if base_dir is None:
        base_dir = Path(OUTPUT_DIR)
    
    # Find all images
    images = find_all_images(base_dir)
    
    if not images:
        logger.warning(f"No images found in {base_dir}")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0
        }
    
    # Process each image
    stats = {
        'total': len(images),
        'successful': 0,
        'failed': 0
    }
    
    for image_path in images:
        # Output directory is the same as image directory
        output_dir = image_path.parent
        
        if process_image(pipeline, image_path, output_dir):
            stats['successful'] += 1
        else:
            stats['failed'] += 1
    
    return stats


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("OCR Processing Pipeline")
    print("Processing all extracted images with PaddleOCRVL")
    print("="*70 + "\n")
    
    # Check if output directory exists
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        print(f"\n[ERROR] Output directory does not exist: {OUTPUT_DIR}")
        print(f"Please run extract_voters.py first to extract images.")
        return
    
    # Initialize OCR pipeline (loaded once globally, reused for all images)
    # Configure based on your needs:
    # - use_layout_detection=False: Disable layout detection (as in your example)
    # - use_doc_orientation_classify: Enable if needed
    # - use_doc_unwarping: Enable if needed
    print("\nLoading OCR models (this may take a moment)...")
    try:
        pipeline = get_ocr_pipeline(
            use_layout_detection=False,
        )
        print("[OK] OCR models loaded successfully\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize OCR pipeline: {e}")
        logger.error(f"Failed to initialize OCR pipeline: {e}", exc_info=True)
        return
    
    # Process all images (using the same global pipeline instance)
    print(f"Processing images from: {output_dir}")
    stats = process_all_images(pipeline, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("OCR PROCESSING COMPLETE")
    print("="*70)
    print(f"Total images processed: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    # Log summary
    logger.info(f"OCR processing complete. Total: {stats['total']}, "
                f"Successful: {stats['successful']}, Failed: {stats['failed']}")


if __name__ == "__main__":
    main()

