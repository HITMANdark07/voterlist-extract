#!/usr/bin/env python3
"""
Main PDF Processing Script
==========================
Processes PDFs:
1. Page 1: Detect contours, crop the page, and store in directory
2. Pages 3 to n-1: Use grid detector to detect grids and save each grid as a hierarchical folder
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import modules
from config import INPUT_DIR, OUTPUT_DIR, USE_MULTIPROCESSING, MAX_WORKERS
from pdf_converter import get_all_pages, get_single_page, get_page_range
from grid_detector import detect_voter_blocks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def detect_page_contour(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the main content area of a page using contour detection.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (x, y, width, height) bounding box or None if detection fails
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found on page")
            return None
        
        # Find the largest contour (main content area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding
        padding = 20
        height, width = image.size[1], image.size[0]
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        logger.info(f"Detected page contour: x={x}, y={y}, w={w}, h={h}")
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
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    x2 = max(0, min(x + w, width))
    y2 = max(0, min(y + h, height))
    
    cropped = image.crop((x, y, x2, y2))
    return cropped


def process_page_1(pdf_path: str, pdf_name: str, output_base_dir: Path) -> bool:
    """
    Process page 1: detect contours, crop, and save directly in pdf_name folder.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_name: Name of the PDF (without extension)
        output_base_dir: Base output directory for this PDF
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing page 1 for {pdf_name}")
        
        # Get page 1
        page_image = get_single_page(pdf_path, 1)
        if not page_image:
            logger.error(f"Could not load page 1 from {pdf_path}")
            return False
        
        # Detect contour
        bbox = detect_page_contour(page_image)
        if not bbox:
            logger.warning(f"Could not detect contour for page 1, saving full page")
            cropped_image = page_image
        else:
            # Crop the page
            cropped_image = crop_page(page_image, bbox)
        
        # Save cropped page directly in pdf_name folder
        output_path = output_base_dir / "page_1.jpg"
        cropped_image.save(output_path, 'JPEG', quality=95)
        
        logger.info(f"✅ Saved page 1 to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing page 1 for {pdf_name}: {e}", exc_info=True)
        return False


def process_grid_pages(pdf_path: str, pdf_name: str, output_base_dir: Path, 
                       start_page: int, end_page: int) -> bool:
    """
    Process pages 3 to n-1: detect grids and save each grid in voters folder.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_name: Name of the PDF (without extension)
        output_base_dir: Base output directory for this PDF
        start_page: Starting page number (1-indexed, inclusive)
        end_page: Ending page number (1-indexed, exclusive)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing pages {start_page} to {end_page-1} for {pdf_name}")
        
        # Get all pages in the range
        page_images = get_page_range(pdf_path, start_page, end_page)
        
        if not page_images:
            logger.error(f"Could not load pages {start_page} to {end_page-1}")
            return False
        
        # Create voters directory
        voters_dir = output_base_dir / "voters"
        voters_dir.mkdir(parents=True, exist_ok=True)
        
        total_grids = 0
        grid_counter = 1  # Global counter across all pages
        
        for page_num, page_image in page_images:
            logger.info(f"Processing page {page_num}")
            
            # Detect grids on this page
            grids = detect_voter_blocks(page_image)
            
            if not grids:
                logger.warning(f"No grids detected on page {page_num}")
                continue
            
            logger.info(f"Detected {len(grids)} grids on page {page_num}")
            
            # Save each grid in voters folder
            for grid_idx, grid_image in enumerate(grids):
                # Save grid image with sequential numbering
                grid_path = voters_dir / f"voter_{grid_counter:03d}.jpg"
                grid_image.save(grid_path, 'JPEG', quality=95)
                
                total_grids += 1
                grid_counter += 1
            
            logger.info(f"✅ Saved {len(grids)} grids from page {page_num}")
        
        logger.info(f"✅ Total grids saved: {total_grids}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing grid pages for {pdf_name}: {e}", exc_info=True)
        return False


def process_pdf(pdf_path: str) -> bool:
    """
    Process a single PDF file:
    1. Process page 1: contour, crop, and save
    2. Process pages 3 to n-1: detect grids and save each grid
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_name = Path(pdf_path).stem
        
        # Get all pages to determine total page count
        all_pages = get_all_pages(pdf_path)
        if not all_pages:
            logger.error(f"Could not load pages from {pdf_path}")
            return False
        
        total_pages = len(all_pages)
        logger.info(f"PDF {pdf_name} has {total_pages} pages")
        
        # Create output directory for this PDF
        output_base_dir = Path(OUTPUT_DIR) / pdf_name
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process page 1
        if not process_page_1(pdf_path, pdf_name, output_base_dir):
            logger.warning(f"Failed to process page 1 for {pdf_name}")
        
        # Process pages 3 to n-1 (if there are enough pages)
        if total_pages >= 3:
            start_page = 3
            end_page = total_pages  # n-1 means up to but not including last page, so end_page = total_pages
            
            if end_page > start_page:
                if not process_grid_pages(pdf_path, pdf_name, output_base_dir, start_page, end_page):
                    logger.warning(f"Failed to process grid pages for {pdf_name}")
            else:
                logger.info(f"Not enough pages to process grid pages (need at least 3 pages)")
        else:
            logger.info(f"PDF has only {total_pages} pages, skipping grid processing")
        
        logger.info(f"✅ Completed processing {pdf_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return False


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def process_pdf_wrapper(pdf_path: str) -> Tuple[str, bool]:
    """
    Wrapper function for process_pdf to work with multiprocessing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (pdf_name, success_status)
    """
    pdf_name = Path(pdf_path).stem
    success = process_pdf(pdf_path)
    return (pdf_name, success)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PDF Processing Pipeline")
    print("Page 1: Contour detection and cropping")
    print("Pages 3 to n-1: Grid detection and extraction")
    print("="*70 + "\n")
    
    # Setup
    setup_directories()
    
    # Find all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in '{INPUT_DIR}' directory.")
        print(f"Please place your PDF files in the '{INPUT_DIR}' folder and run again.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Determine processing mode
    if USE_MULTIPROCESSING and len(pdf_files) > 1:
        num_workers = min(MAX_WORKERS, len(pdf_files))
        print(f"Processing {len(pdf_files)} PDFs in parallel using {num_workers} workers\n")
        
        # Process PDFs in parallel
        successful = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_pdf_wrapper, str(pdf_path)): pdf_path 
                for pdf_path in pdf_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    pdf_name, success = future.result()
                    if success:
                        successful += 1
                        print(f"✅ Completed: {pdf_name}")
                    else:
                        print(f"❌ Failed: {pdf_name}")
                except Exception as e:
                    pdf_name = Path(pdf_path).stem
                    logger.error(f"Exception processing {pdf_name}: {e}", exc_info=True)
                    print(f"❌ Error processing {pdf_name}: {e}")
    else:
        # Process PDFs sequentially
        if USE_MULTIPROCESSING:
            print(f"Processing sequentially (only 1 PDF found)\n")
        else:
            print(f"Processing sequentially (multiprocessing disabled)\n")
        
        successful = 0
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            if process_pdf(str(pdf_path)):
                successful += 1
            print()
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(pdf_files) - successful}")
    print(f"Output location: {OUTPUT_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
