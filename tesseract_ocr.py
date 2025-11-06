#!/usr/bin/env python3
"""
Tesseract OCR Extractor Module
================================
Handles OCR extraction from grid segments using Tesseract:
- Serial number boxes
- Left half (details - 60%)
- Right half (EPIC - 40%)
- Fallback full grid OCR
"""

import logging
import re
from typing import List, Dict
from PIL import Image

from ocr_processor import perform_ocr, ocr_serial_number, ocr_epic_number, ocr_details

logger = logging.getLogger(__name__)


def extract_serial_numbers_from_boxes(serial_boxes: List[List[int]], 
                                      cropped_box_images: List[Image.Image],
                                      boxes: List[List[int]]) -> List[str]:
    """
    Extract serial numbers from serial number boxes using Tesseract OCR.
    
    Args:
        serial_boxes: List of serial box coordinates
        cropped_box_images: List of all cropped box images
        boxes: List of all box coordinates
    
    Returns:
        List of serial number texts extracted from OCR
    """
    serial_texts = []
    
    # Create mapping from box coordinates to cropped images
    box_to_image = {}
    for i, box_coords in enumerate(boxes):
        if i < len(cropped_box_images):
            box_to_image[tuple(box_coords)] = cropped_box_images[i]
    
    # Extract text from each serial box using OCR
    for serial_box in serial_boxes:
        box_key = tuple(serial_box)
        if box_key in box_to_image:
            box_image = box_to_image[box_key]
            serial_text = ocr_serial_number(box_image)
            if serial_text.strip():
                serial_texts.append(serial_text.strip())
    
    return serial_texts


def extract_text_from_grid_segments(grid_data: Dict, page_num: int = 0, grid_idx: int = 0) -> Dict[str, str]:
    """
    Extract text from all grid segments using Tesseract OCR.
    
    Args:
        grid_data: Dictionary containing processed grid data from box_detector
        page_num: Page number for logging
        grid_idx: Grid index for logging
    
    Returns:
        Dictionary with extracted text:
        - 'serial_text': Combined serial numbers
        - 'epic_text': EPIC number from right half
        - 'details_text': Details from left half
    """
    result = {
        'serial_text': '',
        'epic_text': '',
        'details_text': ''
    }
    
    # Extract serial numbers from serial boxes
    serial_boxes = grid_data.get('serial_boxes', [])
    cropped_box_images = grid_data.get('cropped_box_images', [])
    boxes = grid_data.get('boxes', [])
    
    serial_texts = extract_serial_numbers_from_boxes(
        serial_boxes, cropped_box_images, boxes
    )
    
    # Combine serial numbers into a single string
    result['serial_text'] = ' '.join(serial_texts) if serial_texts else ''
    logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Serial boxes detected: {len(serial_boxes)}, "
                f"Serial text extracted: '{result['serial_text']}'")
    
    # Perform OCR on right half (EPIC number section - 40%)
    right_half = grid_data.get('right_half')
    if right_half:
        result['epic_text'] = ocr_epic_number(right_half)
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half OCR (EPIC): "
                    f"'{result['epic_text'][:100] if result['epic_text'] else 'EMPTY'}'")
    else:
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half (EPIC) is None")
    
    # Perform OCR on left half (details section - 60%)
    left_half = grid_data.get('left_half')
    if left_half:
        result['details_text'] = ocr_details(left_half)
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half OCR (Details): "
                    f"'{result['details_text'][:200] if result['details_text'] else 'EMPTY'}...'")
    else:
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half (Details) is None")
    
    # If no text extracted, try OCR on the whole grid with white boxes (fallback)
    if not result['epic_text'].strip() and not result['details_text'].strip():
        logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ No text from segments, trying full grid OCR")
        image_with_white_boxes = grid_data.get('image_with_white_boxes')
        if image_with_white_boxes:
            full_text = perform_ocr(image_with_white_boxes)
            logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Full grid OCR text length: {len(full_text)}")
            
            # Try to extract EPIC from full text
            epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
            epic_match = re.search(epic_pattern, full_text)
            if epic_match:
                result['epic_text'] = epic_match.group(1)
                logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Found EPIC in full text: {result['epic_text']}")
            else:
                logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ No EPIC pattern found in full text")
            
            # Use full text as details
            result['details_text'] = full_text
        else:
            logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ image_with_white_boxes is None")
    
    return result

