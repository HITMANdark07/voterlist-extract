#!/usr/bin/env python3
"""
OCR Processor Module
====================
Performs OCR (Optical Character Recognition) on images to extract text.
"""

import logging
from PIL import Image
import pytesseract

from config import OCR_LANGUAGE, TESSERACT_CONFIG

logger = logging.getLogger(__name__)


def perform_ocr(image: Image.Image) -> str:
    """
    Perform OCR on an image using Tesseract.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text from image
    """
    try:
        # Perform OCR with Hindi and English
        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE,
            config=TESSERACT_CONFIG
        )
        return text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


def perform_ocr_from_file(image_path: str) -> str:
    """
    Perform OCR on an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text from image
    """
    try:
        image = Image.open(image_path)
        return perform_ocr(image)
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        return ""

