#!/usr/bin/env python3
"""
OCR Detector Module (EasyOCR)
==============================
Uses EasyOCR for text detection (bounding boxes) within voter blocks.
"""

import logging
import warnings
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Suppress PyTorch MPS pin_memory warnings (harmless on Apple Silicon)
warnings.filterwarnings('ignore', message='.*pin_memory.*MPS.*', category=UserWarning)

# Lazy import EasyOCR to avoid import errors if not installed
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")


# Global EasyOCR reader instance (initialized lazily)
_reader = None


def get_easyocr_reader():
    """Initialize and return EasyOCR reader instance."""
    global _reader
    if not EASYOCR_AVAILABLE:
        return None
    
    if _reader is None:
        try:
            # Suppress warnings during initialization
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*pin_memory.*MPS.*', category=UserWarning)
                # Initialize EasyOCR reader with Hindi and English
                # Use CPU to avoid MPS issues (gpu=False)
                _reader = easyocr.Reader(['en', 'hi'], gpu=False)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            return None
    
    return _reader


def detect_text_regions(block_image: Image.Image) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect text regions in a voter block using EasyOCR.
    
    Args:
        block_image: PIL Image object of a voter block
        
    Returns:
        List of bounding boxes: [(x1, y1, x2, y2, confidence), ...]
        Returns empty list if EasyOCR unavailable or detection fails
    """
    if not EASYOCR_AVAILABLE:
        logger.debug("EasyOCR not available, skipping detection")
        return []
    
    reader = get_easyocr_reader()
    if reader is None:
        return []
    
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(block_image)
        
        # EasyOCR detection (only detection, not recognition)
        # Suppress warnings during detection
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*pin_memory.*MPS.*', category=UserWarning)
            # Returns list of tuples: (bbox, text, confidence)
            # bbox format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            results = reader.readtext(img_array)
        
        bounding_boxes = []
        for result in results:
            bbox = result[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            confidence = result[2]
            
            # Convert polygon to bounding box (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))
            
            bounding_boxes.append((x1, y1, x2, y2, confidence))
        
        logger.debug(f"EasyOCR detected {len(bounding_boxes)} text regions")
        return bounding_boxes
        
    except Exception as e:
        logger.error(f"Error in EasyOCR detection: {e}")
        return []


def classify_detected_regions(
    bounding_boxes: List[Tuple[int, int, int, int, float]],
    block_width: int,
    block_height: int,
    min_confidence: float = 0.3
) -> Dict[str, Optional[Tuple[int, int, int, int, float]]]:
    """
    Classify detected bounding boxes into serial, EPIC, and details regions.
    
    Args:
        bounding_boxes: List of (x1, y1, x2, y2, confidence) tuples
        block_width: Width of the block image
        block_height: Height of the block image
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with best boxes for each region:
        {'serial_no': box or None, 'epic': box or None, 'details': list of boxes}
    """
    if not bounding_boxes:
        return {'serial_no': None, 'epic': None, 'details': []}
    
    # Filter by confidence
    filtered_boxes = [box for box in bounding_boxes if box[4] >= min_confidence]
    
    if not filtered_boxes:
        logger.debug("No boxes above confidence threshold")
        return {'serial_no': None, 'epic': None, 'details': []}
    
    # Thresholds for region classification
    top_section_height = block_height * 0.20
    serial_max_width = block_width * 0.50
    epic_min_x = block_width * 0.50
    details_start_y = block_height * 0.20
    details_max_width = block_width * 0.70
    
    # Classify boxes
    serial_candidates = []
    epic_candidates = []
    details_boxes = []
    
    for box in filtered_boxes:
        x1, y1, x2, y2, confidence = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Serial number: Top-left, small box
        if (y2 < top_section_height and 
            x2 < serial_max_width and
            20 < width < 150 and  # Reasonable size for serial
            15 < height < 80):
            serial_candidates.append(box)
        
        # EPIC number: Top-right, medium width
        elif (y2 < top_section_height and
              x1 > epic_min_x and
              50 < width < 300 and  # Reasonable size for EPIC
              15 < height < 60):
            epic_candidates.append(box)
        
        # Details: Below top section, left side
        elif (y1 > details_start_y and
              x2 < details_max_width):
            details_boxes.append(box)
    
    # Select best candidate for serial (leftmost, highest confidence)
    serial_box = None
    if serial_candidates:
        serial_box = min(serial_candidates, key=lambda b: (b[0], -b[4]))
        logger.debug(f"Found serial box at ({serial_box[0]}, {serial_box[1]}) with confidence {serial_box[4]:.2f}")
    
    # Select best candidate for EPIC (rightmost, highest confidence)
    epic_box = None
    if epic_candidates:
        epic_box = max(epic_candidates, key=lambda b: (b[2], -b[4]))
        logger.debug(f"Found EPIC box at ({epic_box[0]}, {epic_box[1]}) with confidence {epic_box[4]:.2f}")
    
    return {
        'serial_no': serial_box,
        'epic': epic_box,
        'details': sorted(details_boxes, key=lambda b: (b[1], b[0]))  # Sort by y, then x
    }


def extract_regions_from_detection(
    block_image: Image.Image,
    classified_boxes: Dict[str, Optional[Tuple[int, int, int, int, float]]]
) -> Dict[str, Image.Image]:
    """
    Extract region images from detected bounding boxes.
    
    Args:
        block_image: PIL Image object of a voter block
        classified_boxes: Dictionary with classified boxes from classify_detected_regions
        
    Returns:
        Dictionary with region images: {'serial_no': image, 'epic': image, 'details': image}
    """
    width, height = block_image.size
    padding = 5  # Add small padding around detected boxes
    
    regions = {}
    
    # Extract serial number region
    if classified_boxes['serial_no']:
        x1, y1, x2, y2, _ = classified_boxes['serial_no']
        regions['serial_no'] = block_image.crop((
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding)
        ))
    else:
        regions['serial_no'] = None
    
    # Extract EPIC number region
    if classified_boxes['epic']:
        x1, y1, x2, y2, _ = classified_boxes['epic']
        regions['epic'] = block_image.crop((
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding)
        ))
    else:
        regions['epic'] = None
    
    # Extract details region (merge all detail boxes or use first/largest)
    if classified_boxes['details']:
        # Merge all detail boxes into one region
        detail_boxes = classified_boxes['details']
        if len(detail_boxes) == 1:
            x1, y1, x2, y2, _ = detail_boxes[0]
        else:
            # Find bounding box of all detail boxes
            x1 = min(box[0] for box in detail_boxes)
            y1 = min(box[1] for box in detail_boxes)
            x2 = max(box[2] for box in detail_boxes)
            y2 = max(box[3] for box in detail_boxes)
        
        # Extend to cover full width up to details_max_width
        details_max_width = int(width * 0.70)
        regions['details'] = block_image.crop((
            0,  # Start from left edge
            max(0, y1 - padding),
            min(width, details_max_width),
            height  # Extend to bottom
        ))
    else:
        regions['details'] = None
    
    return regions


def detect_and_split_block(block_image: Image.Image) -> Optional[Dict[str, Image.Image]]:
    """
    Complete pipeline: Detect text regions and split block into regions.
    
    Args:
        block_image: PIL Image object of a voter block
        
    Returns:
        Dictionary with region images, or None if detection fails
    """
    if not EASYOCR_AVAILABLE:
        return None
    
    try:
        # Step 1: Detect text regions
        bounding_boxes = detect_text_regions(block_image)
        
        if not bounding_boxes:
            logger.debug("No text regions detected by EasyOCR")
            return None
        
        # Step 2: Classify regions
        width, height = block_image.size
        classified_boxes = classify_detected_regions(bounding_boxes, width, height)
        
        # Step 3: Check if we found serial and EPIC
        if not classified_boxes['serial_no'] or not classified_boxes['epic']:
            logger.debug("EasyOCR didn't find both serial and EPIC regions")
            return None
        
        # Step 4: Extract regions
        regions = extract_regions_from_detection(block_image, classified_boxes)
        
        # Validate that we have all regions
        if regions['serial_no'] and regions['epic'] and regions['details']:
            logger.debug("EasyOCR detection successful")
            return regions
        else:
            logger.debug("EasyOCR detection incomplete")
            return None
            
    except Exception as e:
        logger.error(f"Error in EasyOCR detection pipeline: {e}")
        return None

