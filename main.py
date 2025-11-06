#!/usr/bin/env python3
"""
Main Extraction Script
======================
Complete pipeline for extracting voter data from PDFs:
1. Detect grids using grid_detector
2. Detect boxes inside grids using box_detector
3. Segment grids (60% left for details, 40% right for EPIC)
4. Perform OCR on each section
5. Parse and save data to CSV
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# Import modules
from config import INPUT_DIR, OUTPUT_DIR, TEMP_DIR, OCR_ENGINE, TEST_MODE
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from box_detector import process_grid
from ocr_factory import extract_text_from_grid_segments
from text_parser import parse_from_regions, extract_voter_info
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


def extract_voter_data_from_grid(grid_data: Dict, page_num: int, grid_idx: int) -> Optional[Dict]:
    """
    Extract voter data from a processed grid.
    
    Args:
        grid_data: Dictionary containing processed grid data from box_detector
        page_num: Page number for logging
        grid_idx: Grid index for logging
    
    Returns:
        voter_data_dict or None if extraction failed
    """
    try:
        # Extract text from grid segments using OCR
        ocr_results = extract_text_from_grid_segments(grid_data, page_num, grid_idx)
        
        serial_text = ocr_results['serial_text']
        epic_text = ocr_results['epic_text']
        details_text = ocr_results['details_text']
        
        voter_data = parse_from_regions(serial_text, epic_text, details_text)
        
        if voter_data:
            return voter_data
        
        # Fallback: try parsing from details text directly
        if details_text.strip():
            epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
            epic_match = re.search(epic_pattern, details_text)
            if epic_match:
                epic_no = epic_match.group(1)
                voter_data = extract_voter_info(details_text, epic_no)
                if voter_data:
                    # Add serial number if available
                    if serial_text.strip():
                        serial_match = re.search(r'\d+', serial_text)
                        if serial_match:
                            try:
                                voter_data['serial_no'] = int(serial_match.group())
                            except ValueError:
                                pass
                    return voter_data
        
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Failed to extract voter data")
        return None
            
    except Exception as e:
        logger.error(f"Error extracting voter data from grid {grid_idx + 1} on page {page_num}: {e}")
        return None


def extract_voter_data_from_image(image: Image.Image, page_num: int) -> tuple:
    """
    Extract voter data from a single page image.
    
    Args:
        image: PIL Image object
        page_num: Page number for logging
    
    Returns:
        Tuple of (list of voter dictionaries, number of grids detected)
    """
    voter_blocks = detect_voter_blocks(image)
    num_grids = len(voter_blocks)
    
    if not voter_blocks:
        return [], 0
    
    voters = []
    for grid_idx, grid_image in enumerate(voter_blocks):
        try:
            grid_data = process_grid(grid_image)
            voter_data = extract_voter_data_from_grid(grid_data, page_num, grid_idx)
            
            if voter_data:
                voters.append(voter_data)
        except Exception as e:
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Error - {e}")
            continue
    
    return voters, num_grids


def process_pdf(pdf_path: str) -> bool:
    """
    Process a single PDF file and extract voter data.
    
    This function orchestrates the entire pipeline:
    1. Convert PDF to images
    2. Detect grids and boxes
    3. Perform OCR on each section
    4. Parse text to extract voter data
    5. Save results to CSV
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_name = Path(pdf_path).stem
        page_images = pdf_to_images(pdf_path)
        
        if not page_images:
            return False
        
        if TEST_MODE:
            page_images = page_images[:1]
        
        all_voters = []
        for page_num, image in page_images:
            voters, num_grids = extract_voter_data_from_image(image, page_num)
            
            # Log grid count and voter count for this page
            num_voters = len(voters)
            logger.info(f"Page {page_num}: Grids detected: {num_grids}, Voters extracted: {num_voters}")
            
            # Stop if counts don't match
            if num_grids != num_voters:
                logger.error(f"Page {page_num}: Mismatch detected! Grids: {num_grids}, Voters: {num_voters}")
                logger.error(f"Stopping execution due to mismatch on page {page_num}")
                raise ValueError(f"Grid count ({num_grids}) does not match voter count ({num_voters}) on page {page_num}")
            
            all_voters.extend(voters)
        
        if not all_voters:
            return False
        
        output_path = save_voters_to_csv(all_voters, pdf_name)
        return output_path is not None
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return False


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Voter Data Extraction Pipeline")
    print("Using Grid Detection + Box Detection + OCR")
    print(f"OCR Engine: {OCR_ENGINE.upper()}")
    if TEST_MODE:
        print("🧪 TEST MODE: Processing only first page")
    print("="*70 + "\n")
    
    # Setup
    setup_directories()
    
    # Find all PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in '{INPUT_DIR}' directory.")
        print(f"Please place your PDF files in the '{INPUT_DIR}' folder and run again.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Process PDFs sequentially
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

