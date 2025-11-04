#!/usr/bin/env python3
"""
Block Splitter Module
=====================
Splits voter blocks into regions using hardcoded positions.
"""

import logging
from PIL import Image
from typing import Dict

logger = logging.getLogger(__name__)


def split_voter_block(block_image: Image.Image) -> Dict[str, Image.Image]:
    """
    Split a voter block into regions using hardcoded percentages:
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
        logger.error(f"Error splitting block: {e}")
        # Return full block as fallback
        return {
            'serial_no': block_image,
            'details': block_image,
            'epic': block_image
        }

