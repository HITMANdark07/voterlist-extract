#!/usr/bin/env python3
"""
OCR Test Script for Block Split Images
=======================================
Performs OCR on images in temp_images/block_split folders
and saves the extracted text as .txt files alongside the images.
"""

import logging
from pathlib import Path
from PIL import Image

from ocr_processor import ocr_serial_number, ocr_epic_number, ocr_details
from config import TEMP_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to block split directory
BLOCK_SPLIT_DIR = Path(TEMP_DIR) / "block_split"


def process_block_folder(folder_path: Path):
    """
    Process a single block folder:
    1. Read serial_no.jpg, epic.jpg, details.jpg
    2. Perform OCR on each
    3. Save OCR text to .txt files
    
    Args:
        folder_path: Path to the numbered folder (e.g., 0001/)
    """
    folder_name = folder_path.name
    logger.info(f"Processing folder: {folder_name}")
    
    # Define image files to process
    image_files = {
        'serial_no': folder_path / 'serial_no.jpg',
        'epic': folder_path / 'epic.jpg',
        'details': folder_path / 'details.jpg'
    }
    
    # Define OCR functions for each region
    ocr_functions = {
        'serial_no': ocr_serial_number,
        'epic': ocr_epic_number,
        'details': ocr_details
    }
    
    processed_count = 0
    
    for region_name, image_path in image_files.items():
        if not image_path.exists():
            logger.warning(f"  ⚠️  {region_name}.jpg not found in {folder_name}")
            continue
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Perform OCR
            ocr_func = ocr_functions[region_name]
            text = ocr_func(image)
            
            # Save text to file
            txt_path = folder_path / f"{region_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Log result
            text_preview = text[:50].replace('\n', ' ') if text else "(empty)"
            logger.info(f"  ✅ {region_name}: {text_preview}...")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {region_name} in {folder_name}: {e}")
            continue
    
    if processed_count == 0:
        logger.warning(f"  ⚠️  No images processed in {folder_name}")
    else:
        logger.info(f"  ✅ Processed {processed_count} images in {folder_name}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("OCR Test Script for Block Split Images")
    print("="*70 + "\n")
    
    # Check if block_split directory exists
    if not BLOCK_SPLIT_DIR.exists():
        logger.error(f"Block split directory not found: {BLOCK_SPLIT_DIR}")
        print(f"\n❌ Directory not found: {BLOCK_SPLIT_DIR}")
        print("Please run test_block_splitter.py first to generate block split images.")
        return
    
    # Get all numbered folders
    folders = sorted([f for f in BLOCK_SPLIT_DIR.iterdir() if f.is_dir()])
    
    if not folders:
        logger.warning(f"No folders found in {BLOCK_SPLIT_DIR}")
        print(f"\n❌ No folders found in {BLOCK_SPLIT_DIR}")
        return
    
    logger.info(f"Found {len(folders)} folder(s) to process")
    print(f"\n📁 Found {len(folders)} folder(s) to process\n")
    
    # Process each folder
    successful = 0
    failed = 0
    
    for folder_path in folders:
        try:
            process_block_folder(folder_path)
            successful += 1
        except Exception as e:
            logger.error(f"Failed to process folder {folder_path.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("OCR TEST SUMMARY")
    print("="*70)
    print(f"Total folders processed: {len(folders)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output location: {BLOCK_SPLIT_DIR}/")
    print("="*70 + "\n")
    print("✅ OCR text files saved alongside images in each folder!")
    print(f"   Each folder now contains:")
    print(f"   - serial_no.jpg + serial_no.txt")
    print(f"   - epic.jpg + epic.txt")
    print(f"   - details.jpg + details.txt")
    print(f"   - block.jpg")


if __name__ == "__main__":
    main()

