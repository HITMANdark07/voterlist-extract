#!/usr/bin/env python3
"""
OCR Factory Module
==================
Factory module to switch between Tesseract and PaddleOCR based on configuration.
"""

import logging
from typing import Dict
from PIL import Image

from config import OCR_ENGINE

logger = logging.getLogger(__name__)


def extract_text_from_grid_segments(grid_data: Dict, page_num: int = 0, grid_idx: int = 0) -> Dict[str, str]:
    """
    Extract text from grid segments using the configured OCR engine.
    
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
    ocr_engine = OCR_ENGINE.lower()
    
    if ocr_engine == "paddleocr":
        logger.debug(f"Using PaddleOCR for text extraction")
        from paddleocr import extract_text_from_grid_segments as paddle_extract
        return paddle_extract(grid_data, page_num, grid_idx)
    else:
        # Default to Tesseract
        logger.debug(f"Using Tesseract OCR for text extraction")
        from tesseract_ocr import extract_text_from_grid_segments as tesseract_extract
        return tesseract_extract(grid_data, page_num, grid_idx)

