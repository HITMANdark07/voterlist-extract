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


def ocr_serial_number(region: Image.Image) -> str:
    """
    Perform OCR on serial number region (small box, single line).
    Uses PSM 7 (single text line) for better accuracy.
    
    Args:
        region: PIL Image object of serial number region
        
    Returns:
        Extracted serial number text
    """
    try:
        text = pytesseract.image_to_string(
            region,
            lang=OCR_LANGUAGE,
            config='--psm 7'  # Single text line
        )
        return text.strip()
    except Exception as e:
        logger.error(f"OCR error on serial number: {e}")
        return ""


def ocr_epic_number(region: Image.Image) -> str:
    """
    Perform OCR on EPIC number region (top-right, alphanumeric).
    Uses PSM 7 (single text line) for better accuracy.
    
    Args:
        region: PIL Image object of EPIC number region
        
    Returns:
        Extracted EPIC number text
    """
    try:
        text = pytesseract.image_to_string(
            region,
            lang=OCR_LANGUAGE,
            config='--psm 7'  # Single text line
        )
        return text.strip()
    except Exception as e:
        logger.error(f"OCR error on EPIC number: {e}")
        return ""


def ocr_details(region: Image.Image) -> str:
    """
    Perform OCR on details region (voter information block).
    Uses PSM 6 (uniform block of text) for better accuracy.
    
    Args:
        region: PIL Image object of details region
        
    Returns:
        Extracted details text
    """
    try:
        text = pytesseract.image_to_string(
            region,
            lang=OCR_LANGUAGE,
            config='--psm 6'  # Uniform block of text
        )
        return text
    except Exception as e:
        logger.error(f"OCR error on details: {e}")
        return ""

