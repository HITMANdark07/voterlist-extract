#!/usr/bin/env python3
"""
PDF Converter Module
===================
Converts PDF pages to images for processing.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from pdf2image import convert_from_path

from config import IMAGE_DPI

logger = logging.getLogger(__name__)


def get_all_pages(pdf_path: str) -> List[Tuple[int, Image.Image]]:
    """
    Convert all PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples (page_number, image) for all pages
    """
    try:
        logger.info(f"Converting all PDF pages to images: {pdf_path}")
        
        images = convert_from_path(
            pdf_path,
            dpi=IMAGE_DPI,
            fmt='jpeg',
            thread_count=4
        )
        
        total_pages = len(images)
        logger.info(f"Total pages in PDF: {total_pages}")
        
        page_images = [(i + 1, img) for i, img in enumerate(images)]
        return page_images
        
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {e}")
        return []


def get_page_range(pdf_path: str, start_page: int, end_page: int) -> List[Tuple[int, Image.Image]]:
    """
    Convert a range of PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Starting page number (1-indexed, inclusive)
        end_page: Ending page number (1-indexed, exclusive)
        
    Returns:
        List of tuples (page_number, image) for the specified page range
    """
    try:
        logger.info(f"Converting pages {start_page} to {end_page-1} from PDF: {pdf_path}")
        
        images = convert_from_path(
            pdf_path,
            dpi=IMAGE_DPI,
            first_page=start_page,
            last_page=end_page - 1,
            fmt='jpeg',
            thread_count=4
        )
        
        page_images = [(start_page + i, img) for i, img in enumerate(images)]
        logger.info(f"Converted {len(page_images)} pages")
        return page_images
        
    except Exception as e:
        logger.error(f"Error converting pages {start_page}-{end_page-1} from PDF {pdf_path}: {e}")
        return []


def get_single_page(pdf_path: str, page_number: int) -> Optional[Image.Image]:
    """
    Convert a single PDF page to an image.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (1-indexed)
        
    Returns:
        PIL Image object or None if conversion failed
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
            return None
        
        return images[0]
        
    except Exception as e:
        logger.error(f"Error converting page {page_number}: {e}")
        return None


def pdf_to_images(pdf_path: str) -> List[Tuple[int, Image.Image]]:
    """
    Convert PDF pages to images, skipping first 2 and last page.
    This is kept for backward compatibility.
    
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
        
        # Calculate which pages to process (skip first 2 and last page)
        start_page = 2  # Skip first 2 pages (0-indexed: pages 0 and 1)
        end_page = total_pages - 1  # Skip last page
        
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

