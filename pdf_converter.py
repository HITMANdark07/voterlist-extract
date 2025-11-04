#!/usr/bin/env python3
"""
PDF Converter Module
===================
Converts PDF pages to images for OCR processing.
"""

import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from pdf2image import convert_from_path

from config import IMAGE_DPI, SKIP_FIRST_N_PAGES, SKIP_LAST_N_PAGES

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str) -> List[Tuple[int, Image.Image]]:
    """
    Convert PDF pages to images, skipping first 2 and last page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples (page_number, image) for relevant pages
    """
    try:
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Get total page count first
        images = convert_from_path(
            pdf_path,
            dpi=IMAGE_DPI,
            fmt='jpeg',
            thread_count=4
        )
        
        total_pages = len(images)
        logger.info(f"Total pages in PDF: {total_pages}")
        
        # Calculate which pages to process
        start_page = SKIP_FIRST_N_PAGES
        end_page = total_pages - SKIP_LAST_N_PAGES
        
        if end_page <= start_page:
            logger.warning(f"Not enough pages to process in {pdf_path}")
            return []
        
        # Keep only relevant pages
        relevant_images = [
            (i + 1, img) for i, img in enumerate(images)
            if start_page <= i < end_page
        ]
        
        logger.info(f"Processing pages {start_page + 1} to {end_page} ({len(relevant_images)} pages)")
        return relevant_images
        
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {e}")
        return []


def convert_single_page(pdf_path: str, page_number: int, output_path: str = None) -> bool:
    """
    Convert a single PDF page to an image file.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (1-indexed)
        output_path: Optional output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        images = convert_from_path(
            pdf_path,
            dpi=IMAGE_DPI,
            first_page=page_number,
            last_page=page_number,
            fmt='jpeg'
        )
        
        if not images:
            logger.error(f"Could not convert page {page_number}")
            return False
        
        if output_path:
            images[0].save(output_path, 'JPEG', quality=95)
            logger.info(f"Saved page {page_number} to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting page {page_number}: {e}")
        return False

