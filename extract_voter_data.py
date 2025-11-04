#!/usr/bin/env python3
"""
Main Extraction Orchestrator
==============================
Orchestrates the PDF to CSV extraction pipeline using modular components.
"""

import logging
from pathlib import Path
from typing import List, Dict
from PIL import Image

# Import modules
from config import INPUT_DIR, OUTPUT_DIR, TEMP_DIR, USE_MULTIPROCESSING, MAX_WORKERS
from pdf_converter import pdf_to_images
from ocr_processor import perform_ocr
from text_parser import parse_single_block
from grid_detector import detect_voter_blocks
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
    
    # Step 2: OCR each block separately
    voters = []
    for block_idx, block_image in enumerate(voter_blocks):
        try:
            # Perform OCR on individual block
            ocr_text = perform_ocr(block_image)
            
            if not ocr_text.strip():
                logger.debug(f"Page {page_num}, Block {block_idx + 1}: No text extracted")
                continue
            
            # Parse voter data from block
            voter_data = parse_single_block(ocr_text)
            
            if voter_data:
                voters.append(voter_data)
            else:
                logger.debug(f"Page {page_num}, Block {block_idx + 1}: Could not parse voter data")
                
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
        
        # Step 2 & 3: Extract voter data from all pages
        all_voters = []
        
        for page_num, image in page_images:
            voters = extract_voter_data_from_image(image, page_num)
            all_voters.extend(voters)
        
        if not all_voters:
            logger.warning(f"No voter data extracted from {pdf_name}")
            return False
        
        # Step 4: Save to CSV
        output_path = save_voters_to_csv(all_voters, pdf_name)
        
        if output_path:
            logger.info(f"✅ Successfully extracted {len(all_voters)} voters")
            logger.info(f"✅ Saved to: {output_path}")
            return True
        else:
            logger.error(f"Failed to save CSV for {pdf_name}")
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
