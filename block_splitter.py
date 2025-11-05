#!/usr/bin/env python3
"""
Block Splitter Module
=====================
Splits voter blocks into regions using EasyOCR detection with hardcoded fallback.
"""

import logging
from PIL import Image
from typing import Dict

# Import EasyOCR detector
try:
    from ocr_detector import detect_and_split_block
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


def split_voter_block_hardcoded(block_image: Image.Image) -> Dict[str, Image.Image]:
    """
    Split a voter block into regions using hardcoded percentages (fallback method).
    - serial_no: Top-left, extends 20% down and 50% right
    - epic: Starts at 50% from left, 20% from top
    - details: Starts at 20% from top, 70% from left
    
    Args:
        block_image: PIL Image object of a voter block
        
    Returns:
        Dictionary with region images
    """
    try:
        width, height = block_image.size
        
        # Serial number: Top-left to 20% down and 50% right
        serial_region = block_image.crop((
            0,                              # Start from left
            0,                              # Start from top
            int(width * 0.50),             # 50% width
            int(height * 0.20)             # 20% height
        ))
        
        # EPIC number: Starts at 50% from left, 20% from top
        # Extends to right edge (or 70% width?) and 20% height
        epic_region = block_image.crop((
            int(width * 0.50),             # Start at 50% from left
            0,                              # Start from top
            width,                          # Extend to right edge
            int(height * 0.20)             # 20% height
        ))
        
        # Details: Starts at 20% from top, 70% from left
        details_region = block_image.crop((
            0,                              # Start from left
            int(height * 0.20),            # Start at 20% from top
            int(width * 0.70),             # 70% width
            height                          # Extend to bottom
        ))
        
        return {
            'serial_no': serial_region,
            'details': details_region,
            'epic': epic_region
        }
        
    except Exception as e:
        logger.error(f"Error in hardcoded split: {e}")
        # Return full block as fallback
        return {
            'serial_no': block_image,
            'details': block_image,
            'epic': block_image
        }


def split_voter_block(block_image: Image.Image) -> Dict[str, Image.Image]:
    """
    Split a voter block into regions using EasyOCR detection with hardcoded fallback.
    
    Strategy:
    1. Try EasyOCR detection first (if available)
    2. If EasyOCR fails or unavailable, use hardcoded percentages
    
    Args:
        block_image: PIL Image object of a voter block
        
    Returns:
        Dictionary with region images
    """
    # Try EasyOCR detection first (if available)
    if EASYOCR_AVAILABLE:
        try:
            easyocr_regions = detect_and_split_block(block_image)
            if easyocr_regions:
                logger.debug("Using EasyOCR detection for block splitting")
                return easyocr_regions
            else:
                logger.debug("EasyOCR detection failed, using hardcoded fallback")
        except Exception as e:
            logger.debug(f"EasyOCR detection error: {e}, using hardcoded fallback")
    else:
        logger.debug("EasyOCR not available, using hardcoded split")
    
    # Fallback to hardcoded method
    return split_voter_block_hardcoded(block_image)

