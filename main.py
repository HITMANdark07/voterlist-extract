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
from config import INPUT_DIR, OUTPUT_DIR, TEMP_DIR
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from box_detector import process_grid
from ocr_processor import perform_ocr, ocr_serial_number, ocr_epic_number, ocr_details
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

# Test mode: Process only first page for testing
TEST_MODE = True  # Set to False to process all pages


def extract_serial_numbers_from_boxes(serial_boxes: List[List[int]], 
                                      cropped_box_images: List[Image.Image],
                                      boxes: List[List[int]]) -> List[str]:
    """
    Extract serial numbers from serial number boxes.
    
    Args:
        serial_boxes: List of serial box coordinates
        cropped_box_images: List of all cropped box images
        boxes: List of all box coordinates
    
    Returns:
        List of serial number texts
    """
    serial_texts = []
    
    # Create mapping from box coordinates to cropped images
    box_to_image = {}
    for i, box_coords in enumerate(boxes):
        if i < len(cropped_box_images):
            box_to_image[tuple(box_coords)] = cropped_box_images[i]
    
    # Extract text from each serial box
    for serial_box in serial_boxes:
        box_key = tuple(serial_box)
        if box_key in box_to_image:
            box_image = box_to_image[box_key]
            serial_text = ocr_serial_number(box_image)
            if serial_text.strip():
                serial_texts.append(serial_text.strip())
    
    return serial_texts


def save_grid_images_before_ocr(grid_data: Dict, pdf_name: str, page_num: int, 
                                grid_idx: int, output_dir: Path):
    """
    Save grid images before OCR processing for inspection.
    
    Args:
        grid_data: Dictionary containing processed grid data
        pdf_name: Name of the PDF file
        page_num: Page number
        grid_idx: Grid index
        output_dir: Base output directory
    """
    try:
        grid_dir = output_dir / pdf_name / f"page_{page_num}" / f"grid_{grid_idx + 1:03d}_before_ocr"
        grid_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original grid
        original_image = grid_data.get('original_image')
        if original_image:
            original_path = grid_dir / "00_original_grid.jpg"
            original_image.save(original_path, 'JPEG', quality=95)
        
        # Save grid with white boxes
        image_with_white_boxes = grid_data.get('image_with_white_boxes')
        if image_with_white_boxes:
            white_boxes_path = grid_dir / "01_grid_with_white_boxes.jpg"
            image_with_white_boxes.save(white_boxes_path, 'JPEG', quality=95)
        
        # Save left half (60% - details)
        left_half = grid_data.get('left_half')
        if left_half:
            left_half_path = grid_dir / "02_left_half_60_percent.jpg"
            left_half.save(left_half_path, 'JPEG', quality=95)
        
        # Save right half (40% - EPIC)
        right_half = grid_data.get('right_half')
        if right_half:
            right_half_path = grid_dir / "03_right_half_40_percent.jpg"
            right_half.save(right_half_path, 'JPEG', quality=95)
        
        # Save all detected boxes separately
        boxes = grid_data.get('boxes', [])
        cropped_box_images = grid_data.get('cropped_box_images', [])
        serial_boxes = grid_data.get('serial_boxes', [])
        photo_box = grid_data.get('photo_box')
        
        boxes_dir = grid_dir / "detected_boxes"
        boxes_dir.mkdir(exist_ok=True)
        
        # Create mapping
        box_to_image = {}
        for i, box_coords in enumerate(boxes):
            if i < len(cropped_box_images):
                box_to_image[tuple(box_coords)] = cropped_box_images[i]
        
        # Save all boxes with labels
        for idx, box_coords in enumerate(boxes):
            box_key = tuple(box_coords)
            if box_key in box_to_image:
                box_image = box_to_image[box_key]
                
                # Determine box type
                if photo_box and box_coords == photo_box:
                    box_type = "photo"
                    box_num = len(serial_boxes) + 1
                elif box_coords in [tuple(sb) for sb in serial_boxes]:
                    box_type = "serial"
                    # Find serial box number
                    box_num = next((i+1 for i, sb in enumerate(serial_boxes) if tuple(sb) == box_key), idx + 1)
                else:
                    box_type = "other"
                    box_num = idx + 1
                
                box_filename = f"box_{idx+1:02d}_{box_type}_{box_num:02d}_x{box_coords[0]}_y{box_coords[1]}.jpg"
                box_path = boxes_dir / box_filename
                box_image.save(box_path, 'JPEG', quality=95)
        
        # Save visualization with boxes drawn
        import cv2
        import numpy as np
        img_array = np.array(original_image) if original_image else np.array(image_with_white_boxes)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Draw all boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw serial boxes in blue
        for box in serial_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw photo box in red
        if photo_box:
            x1, y1, x2, y2 = photo_box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        boxes_visualization = Image.fromarray(img_array)
        visualization_path = grid_dir / "04_boxes_visualization.jpg"
        boxes_visualization.save(visualization_path, 'JPEG', quality=95)
        
        # Save metadata
        metadata_path = grid_dir / "metadata.txt"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Grid Index: {grid_idx + 1}\n")
            f.write(f"Page Number: {page_num}\n")
            f.write(f"Total Boxes Detected: {len(boxes)}\n")
            f.write(f"Serial Number Boxes: {len(serial_boxes)}\n")
            f.write(f"Photo Box: {'Yes' if photo_box else 'No'}\n\n")
            
            f.write("All Box Coordinates:\n")
            for i, box in enumerate(boxes):
                f.write(f"  Box {i+1}: {box}\n")
            
            f.write("\nSerial Box Coordinates:\n")
            for i, box in enumerate(serial_boxes):
                f.write(f"  Serial Box {i+1}: {box}\n")
            
            if photo_box:
                f.write(f"\nPhoto Box Coordinates: {photo_box}\n")
        
        logger.debug(f"Saved grid images before OCR to: {grid_dir}")
        
    except Exception as e:
        logger.error(f"Error saving grid images before OCR: {e}")


def extract_voter_data_from_grid(grid_data: Dict, page_num: int, grid_idx: int) -> tuple:
    """
    Extract voter data from a processed grid.
    
    Args:
        grid_data: Dictionary containing processed grid data from box_detector
        page_num: Page number for logging
        grid_idx: Grid index for logging
    
    Returns:
        Tuple of (voter_data_dict, images_dict) or (None, None)
        images_dict contains: 'serial_boxes', 'left_half', 'right_half', 'photo_box'
    """
    try:
        # Extract serial numbers from serial boxes
        serial_boxes = grid_data.get('serial_boxes', [])
        cropped_box_images = grid_data.get('cropped_box_images', [])
        boxes = grid_data.get('boxes', [])
        
        serial_texts = extract_serial_numbers_from_boxes(
            serial_boxes, cropped_box_images, boxes
        )
        
        # Combine serial numbers into a single string
        serial_text = ' '.join(serial_texts) if serial_texts else ''
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Serial boxes detected: {len(serial_boxes)}, "
                    f"Serial text extracted: '{serial_text}'")
        
        # Perform OCR on right half (EPIC number section - 40%)
        right_half = grid_data.get('right_half')
        epic_text = ''
        if right_half:
            epic_text = ocr_epic_number(right_half)
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half OCR (EPIC): '{epic_text[:100] if epic_text else 'EMPTY'}'")
        else:
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half (EPIC) is None")
        
        # Perform OCR on left half (details section - 60%)
        left_half = grid_data.get('left_half')
        details_text = ''
        if left_half:
            details_text = ocr_details(left_half)
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half OCR (Details): "
                        f"'{details_text[:200] if details_text else 'EMPTY'}...'")
        else:
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half (Details) is None")
        
        # If no text extracted, try OCR on the whole grid with white boxes
        if not epic_text.strip() and not details_text.strip():
            logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ No text from segments, trying full grid OCR")
            image_with_white_boxes = grid_data.get('image_with_white_boxes')
            if image_with_white_boxes:
                full_text = perform_ocr(image_with_white_boxes)
                logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Full grid OCR text length: {len(full_text)}")
                # Try to extract EPIC from full text
                epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
                epic_match = re.search(epic_pattern, full_text)
                if epic_match:
                    epic_text = epic_match.group(1)
                    logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Found EPIC in full text: {epic_text}")
                else:
                    logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ No EPIC pattern found in full text")
                # Use full text as details
                details_text = full_text
            else:
                logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ image_with_white_boxes is None")
        
        # Parse voter data from regions
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Attempting to parse voter data...")
        logger.debug(f"  - Serial: '{serial_text}'")
        logger.debug(f"  - EPIC: '{epic_text}'")
        logger.debug(f"  - Details length: {len(details_text)} chars")
        
        voter_data = parse_from_regions(serial_text, epic_text, details_text)
        
        # Prepare images dictionary
        images_dict = {
            'serial_boxes': [],
            'left_half': left_half,
            'right_half': right_half,
            'photo_box': None
        }
        
        # Extract serial box images
        box_to_image = {}
        for i, box_coords in enumerate(boxes):
            if i < len(cropped_box_images):
                box_to_image[tuple(box_coords)] = cropped_box_images[i]
        
        for serial_box in serial_boxes:
            box_key = tuple(serial_box)
            if box_key in box_to_image:
                images_dict['serial_boxes'].append(box_to_image[box_key])
        
        # Extract photo box image
        photo_box = grid_data.get('photo_box')
        if photo_box:
            photo_box_key = tuple(photo_box)
            if photo_box_key in box_to_image:
                images_dict['photo_box'] = box_to_image[photo_box_key]
        
        if voter_data:
            logger.info(f"Page {page_num}, Grid {grid_idx + 1}: ✅ Successfully extracted voter data - "
                       f"EPIC: {voter_data.get('epic_no', 'N/A')}, "
                       f"Name: {voter_data.get('name', 'N/A')[:30]}...")
            return voter_data, images_dict
        else:
            logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ❌ Initial parsing failed, trying fallback...")
            
            # Fallback: try parsing from details text directly
            if details_text.strip():
                epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
                epic_match = re.search(epic_pattern, details_text)
                if epic_match:
                    epic_no = epic_match.group(1)
                    logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Fallback - Found EPIC: {epic_no}")
                    voter_data = extract_voter_info(details_text, epic_no)
                    if voter_data:
                        logger.info(f"Page {page_num}, Grid {grid_idx + 1}: ✅ Fallback parsing succeeded - "
                                   f"EPIC: {epic_no}, Name: {voter_data.get('name', 'N/A')[:30]}...")
                        # Add serial number if available
                        if serial_text.strip():
                            serial_match = re.search(r'\d+', serial_text)
                            if serial_match:
                                try:
                                    voter_data['serial_no'] = int(serial_match.group())
                                except ValueError:
                                    pass
                        return voter_data, images_dict
                    else:
                        logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ❌ Fallback parsing failed - "
                                      f"EPIC found but extract_voter_info returned None")
                else:
                    logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ❌ Fallback failed - "
                                  f"No EPIC pattern found in details text")
            else:
                logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ❌ Fallback failed - Details text is empty")
            
            # Final failure logging
            logger.error(f"Page {page_num}, Grid {grid_idx + 1}: ❌❌ FAILED TO EXTRACT VOTER DATA")
            logger.error(f"  Summary:")
            logger.error(f"    - Serial boxes: {len(serial_boxes)}")
            logger.error(f"    - Serial text: '{serial_text}'")
            logger.error(f"    - EPIC text: '{epic_text}'")
            logger.error(f"    - Details text length: {len(details_text)}")
            logger.error(f"    - Details preview: '{details_text[:150] if details_text else 'EMPTY'}...'")
            return None, None
            
    except Exception as e:
        logger.error(f"Error extracting voter data from grid {grid_idx + 1} on page {page_num}: {e}")
        return None, None


def extract_voter_data_from_image(image: Image.Image, page_num: int, pdf_name: str = None, output_images_dir: Path = None) -> tuple:
    """
    Extract voter data from a single page image.
    
    Args:
        image: PIL Image object
        page_num: Page number for logging
    
    Returns:
        Tuple of (list of voter dictionaries, list of image dictionaries)
    """
    logger.info(f"Processing page {page_num}...")
    
    # Step 1: Detect grids using grid_detector
    voter_blocks = detect_voter_blocks(image)
    
    if not voter_blocks:
        logger.warning(f"No grids detected on page {page_num}")
        return [], []
    
    logger.info(f"Page {page_num}: Detected {len(voter_blocks)} grids")
    
    # Step 2: Process each grid with box_detector
    voters = []
    voter_images = []
    successful_grids = 0
    failed_grids = 0
    
    logger.info(f"Page {page_num}: Processing {len(voter_blocks)} grids...")
    
    # Create output directory for grid images
    output_images_dir = Path(OUTPUT_DIR) / "images"
    
    for grid_idx, grid_image in enumerate(voter_blocks):
        try:
            # Process grid: detect boxes, color white, segment
            grid_data = process_grid(grid_image)
            
            num_boxes = len(grid_data.get('boxes', []))
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}/{len(voter_blocks)}: Detected {num_boxes} boxes")
            
            # Save grid images BEFORE OCR for inspection
            if pdf_name and output_images_dir:
                save_grid_images_before_ocr(grid_data, pdf_name, page_num, grid_idx, output_images_dir)
            
            # Step 3: Extract voter data from processed grid
            voter_data, images_dict = extract_voter_data_from_grid(grid_data, page_num, grid_idx)
            
            if voter_data:
                voters.append(voter_data)
                voter_images.append(images_dict)
                successful_grids += 1
            else:
                failed_grids += 1
                logger.warning(f"Page {page_num}, Grid {grid_idx + 1}/{len(voter_blocks)}: ❌ Failed to extract voter data")
                
        except Exception as e:
            failed_grids += 1
            logger.error(f"Page {page_num}, Grid {grid_idx + 1}/{len(voter_blocks)}: ❌ Exception during processing: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    logger.info(f"Page {page_num}: ✅ Extracted {len(voters)} voters from {len(voter_blocks)} grids "
               f"(Success: {successful_grids}, Failed: {failed_grids})")
    return voters, voter_images


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
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing PDF: {pdf_name}")
        logger.info(f"{'='*60}")
        
        # Step 1: Convert PDF to images
        page_images = pdf_to_images(pdf_path)
        
        if not page_images:
            logger.warning(f"No pages to process in {pdf_name}")
            return False
        
        logger.info(f"Found {len(page_images)} pages to process")
        
        # Limit to first page if in test mode
        if TEST_MODE:
            page_images = page_images[:1]
            logger.info("🧪 TEST MODE: Processing only first page")
        
        # Step 2: Extract voter data from pages
        all_voters = []
        output_images_dir = Path(OUTPUT_DIR) / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        
        for page_num, image in page_images:
            voters, voter_images_list = extract_voter_data_from_image(image, page_num, pdf_name, output_images_dir)
            
            # Save images for each voter
            if voters and voter_images_list and len(voters) == len(voter_images_list):
                for voter_idx, (voter_data, images_dict) in enumerate(zip(voters, voter_images_list)):
                    save_voter_images(voter_data, images_dict, pdf_name, page_num, voter_idx, output_images_dir)
            
            all_voters.extend(voters)
        
        if not all_voters:
            logger.warning(f"No voter data extracted from {pdf_name}")
            return False
        
        # Step 3: Save to CSV
        output_path = save_voters_to_csv(all_voters, pdf_name)
        
        if output_path:
            logger.info(f"✅ Successfully extracted {len(all_voters)} voters")
            logger.info(f"✅ Saved to: {output_path}")
            
            # Summary statistics
            if len(page_images) > 0:
                avg_voters_per_page = len(all_voters) / len(page_images)
                logger.info(f"📊 Statistics: Average {avg_voters_per_page:.1f} voters per page")
            
            return True
        else:
            logger.error(f"Failed to save CSV for {pdf_name}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return False


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories setup complete: {INPUT_DIR}, {OUTPUT_DIR}, {TEMP_DIR}")


def save_voter_images(voter_data: Dict, images_dict: Dict, pdf_name: str, 
                      page_num: int, voter_idx: int, output_dir: Path):
    """
    Save images for a single voter in a nested directory structure.
    
    Args:
        voter_data: Dictionary with voter information
        images_dict: Dictionary containing images ('serial_boxes', 'left_half', 'right_half', 'photo_box')
        pdf_name: Name of the PDF file
        page_num: Page number
        voter_idx: Voter index on the page
        output_dir: Base output directory
    """
    try:
        # Create directory structure: output_images/pdf_name/page_X/voter_Y/
        voter_dir = output_dir / pdf_name / f"page_{page_num}" / f"voter_{voter_idx + 1:03d}"
        voter_dir.mkdir(parents=True, exist_ok=True)
        
        # Save serial number boxes
        serial_boxes = images_dict.get('serial_boxes', [])
        for idx, serial_box_img in enumerate(serial_boxes):
            serial_path = voter_dir / f"serial_box_{idx + 1:02d}.jpg"
            serial_box_img.save(serial_path, 'JPEG', quality=95)
        
        # Save left half (details)
        left_half = images_dict.get('left_half')
        if left_half:
            left_path = voter_dir / "left_half_details.jpg"
            left_half.save(left_path, 'JPEG', quality=95)
        
        # Save right half (EPIC)
        right_half = images_dict.get('right_half')
        if right_half:
            right_path = voter_dir / "right_half_epic.jpg"
            right_half.save(right_path, 'JPEG', quality=95)
        
        # Save photo box
        photo_box = images_dict.get('photo_box')
        if photo_box:
            photo_path = voter_dir / "photo_box.jpg"
            photo_box.save(photo_path, 'JPEG', quality=95)
        
        # Save metadata
        metadata_path = voter_dir / "metadata.txt"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Voter Index: {voter_idx + 1}\n")
            f.write(f"Page Number: {page_num}\n")
            f.write(f"Serial Number: {voter_data.get('serial_no', 'N/A')}\n")
            f.write(f"EPIC Number: {voter_data.get('epic_no', 'N/A')}\n")
            f.write(f"Name: {voter_data.get('name', 'N/A')}\n")
            f.write(f"Serial Boxes: {len(serial_boxes)}\n")
            f.write(f"Photo Box: {'Yes' if photo_box else 'No'}\n")
        
    except Exception as e:
        logger.error(f"Error saving voter images for voter {voter_idx + 1} on page {page_num}: {e}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Voter Data Extraction Pipeline")
    print("Using Grid Detection + Box Detection + OCR")
    if TEST_MODE:
        print("🧪 TEST MODE: Processing only first page")
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
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()

