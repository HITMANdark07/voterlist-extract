#!/usr/bin/env python3
"""
Main Extraction Orchestrator
==============================
Orchestrates the PDF to CSV extraction pipeline using modular components.
"""

import logging
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageFilter
import numpy as np
import cv2

# Import modules
from config import INPUT_DIR, OUTPUT_DIR, TEMP_DIR, USE_MULTIPROCESSING, MAX_WORKERS
from pdf_converter import pdf_to_images
from ocr_processor import ocr_serial_number, ocr_epic_number, ocr_details
from text_parser import parse_from_regions
from grid_detector import detect_voter_blocks
from block_splitter import split_voter_block
from data_saver import save_voters_to_csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voter_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def sharpen_image(image: Image.Image) -> Image.Image:
    """
    Sharpen the image using unsharp mask filter.
    Enhances edges and improves OCR accuracy while maintaining high quality.
    
    Args:
        image: PIL Image object
        
    Returns:
        Sharpened PIL Image object
    """
    # Apply unsharp mask filter (radius=2, percent=150, threshold=3)
    # This enhances edges and improves OCR accuracy
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return sharpened


def remove_lines_and_boxes(image: Image.Image) -> Image.Image:
    """
    Remove non-character lines and boxes from an image region.
    This helps clean up serial numbers and EPIC numbers by removing
    grid lines and borders that interfere with OCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        Cleaned PIL Image object with lines/boxes removed
    """
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Create a copy for processing
        cleaned = gray.copy()
        
        # Convert to binary for better line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect and remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts_h = cv2.findContours(detected_lines_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1]
        for c in cnts_h:
            # Fill detected horizontal lines with white (background color)
            cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 3)
        
        # Detect and remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detected_lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts_v = cv2.findContours(detected_lines_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1]
        for c in cnts_v:
            # Fill detected vertical lines with white (background color)
            cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 3)
        
        # Detect and remove rectangular boxes (boundaries)
        # Find contours that might be boxes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            img_area = width * height
            
            # Check if this is likely a border box (large area, near edges)
            is_border = (
                (x < 5 or y < 5 or x + w > width - 5 or y + h > height - 5) and
                area > img_area * 0.3 and  # Large area
                (w > width * 0.7 or h > height * 0.7)  # Spans most of width/height
            )
            
            if is_border:
                # Fill the border with white
                cv2.drawContours(cleaned, [contour], -1, (255, 255, 255), 3)
                # Also fill the rectangle area
                cv2.rectangle(cleaned, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
        # Convert back to PIL Image
        # Ensure it's RGB if original was RGB
        if len(img_array.shape) == 3:
            cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(cleaned_rgb)
        else:
            return Image.fromarray(cleaned)
            
    except Exception as e:
        logger.debug(f"Error removing lines/boxes: {e}, using original image")
        return image  # Return original on error


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories setup complete: {INPUT_DIR}, {OUTPUT_DIR}, {TEMP_DIR}")


def extract_voter_data_from_image(image: Image.Image, page_num: int) -> List[Dict[str, any]]:
    """
    Extract voter data from a single image using grid detection.
    
    Args:
        image: PIL Image object
        page_num: Page number for logging
        
    Returns:
        List of voter dictionaries
    """
    logger.info(f"Processing page {page_num}...")
    
    # Step 1: Detect grid and extract individual voter blocks
    voter_blocks = detect_voter_blocks(image)
    
    if not voter_blocks:
        logger.warning(f"No voter blocks detected on page {page_num}")
        return []
    
    logger.info(f"Page {page_num}: Detected {len(voter_blocks)} voter blocks")
    
    # Step 2: Split blocks into regions and OCR each region separately
    voters = []
    for block_idx, block_image in enumerate(voter_blocks):
        try:
            # Sharpen the block image before processing (maintains high quality)
            sharpened_block = sharpen_image(block_image)
            
            # Split block into regions
            regions = split_voter_block(sharpened_block)
            
            # Clean serial_no and epic regions: remove lines and boxes
            # Keep details region as-is (no cleaning to preserve formatting)
            cleaned_serial = remove_lines_and_boxes(regions['serial_no'])
            cleaned_epic = remove_lines_and_boxes(regions['epic'])
            
            # Perform OCR on each region with appropriate PSM mode
            serial_text = ocr_serial_number(cleaned_serial)
            epic_text = ocr_epic_number(cleaned_epic)
            details_text = ocr_details(regions['details'])  # Use original details region
            
            if not epic_text.strip() and not details_text.strip():
                logger.debug(f"Page {page_num}, Block {block_idx + 1}: No text extracted from regions")
                continue
            
            # Parse voter data from regions
            voter_data = parse_from_regions(serial_text, epic_text, details_text)
            
            if voter_data:
                voters.append(voter_data)
            else:
                logger.debug(f"Page {page_num}, Block {block_idx + 1}: Could not parse voter data from regions")
                
        except Exception as e:
            logger.debug(f"Error processing block {block_idx + 1} on page {page_num}: {e}")
            continue
    
    logger.info(f"Page {page_num}: Extracted {len(voters)} voters from {len(voter_blocks)} blocks")
    return voters


def process_pdf(pdf_path: str) -> bool:
    """
    Process a single PDF file and extract voter data.
    
    This function orchestrates the entire pipeline:
    1. Convert PDF to images
    2. Perform OCR on each image
    3. Parse text to extract voter data
    4. Save results to CSV
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_name = Path(pdf_path).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing PDF: {pdf_name}")
        logger.info(f"{'='*60}")
        
        # Step 1: Convert PDF to images
        page_images = pdf_to_images(pdf_path)
        
        if not page_images:
            logger.warning(f"No pages to process in {pdf_name}")
            return False
        
        # Step 2 & 3: Process all pages and accumulate voters
        all_voters = []
        
        for page_num, image in page_images:
            # Extract voter data from current page
            voters = extract_voter_data_from_image(image, page_num)
            
            if voters:
                all_voters.extend(voters)
                logger.info(f"✅ Page {page_num}: Extracted {len(voters)} voters (Total: {len(all_voters)})")
            else:
                logger.warning(f"Page {page_num}: No voters extracted")
        
        # Step 4: Save all accumulated voters to CSV at the end
        if all_voters:
            output_path = save_voters_to_csv(all_voters, pdf_name)
            if output_path:
                logger.info(f"✅ Successfully extracted {len(all_voters)} voters from all pages")
                logger.info(f"✅ Saved to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save CSV for {pdf_name}")
                return False
        else:
            logger.warning(f"No voter data extracted from {pdf_name}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return False


def process_pdf_wrapper(pdf_path: str):
    """Wrapper function for multiprocessing."""
    return (pdf_path, process_pdf(pdf_path))


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Indian Election Voter Data Extraction Script")
    print("="*70 + "\n")
    
    # Setup
    setup_directories()
    
    # Find all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {INPUT_DIR}")
        print(f"\n❌ No PDF files found in '{INPUT_DIR}' directory.")
        print(f"Please place your PDF files in the '{INPUT_DIR}' folder and run again.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process PDFs
    if USE_MULTIPROCESSING and len(pdf_files) > 1:
        logger.info(f"Using multiprocessing with {MAX_WORKERS} workers")
        from multiprocessing import Pool
        
        with Pool(MAX_WORKERS) as pool:
            results = pool.map(process_pdf_wrapper, [str(p) for p in pdf_files])
        
        successful = sum(1 for _, success in results if success)
    else:
        logger.info("Processing PDFs sequentially")
        successful = 0
        for pdf_path in pdf_files:
            if process_pdf(str(pdf_path)):
                successful += 1
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(pdf_files) - successful}")
    print(f"Output location: {OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
