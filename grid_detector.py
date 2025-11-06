#!/usr/bin/env python3
"""
Grid Detector Module (Improved)
================================
Detects grid structure in voter pages and extracts individual voter blocks using OpenCV.
"""

import logging
import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2

logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for grid detection."""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply adaptive threshold for better line detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary


def detect_grid_lines(binary_image: np.ndarray, orientation: str = 'horizontal') -> List[int]:
    """
    Detect actual grid lines (not text lines).
    Uses morphological operations to find long lines.
    """
    height, width = binary_image.shape
    
    if orientation == 'horizontal':
        # Use longer kernel for horizontal lines (should span most of width)
        kernel_length = width // 4  # Lines should span at least 25% of width
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        
        # Detect lines
        detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        detected_lines = cv2.dilate(detected_lines, horizontal_kernel, iterations=1)
        
        # Find line positions (more strict - line should span at least 40% of width)
        line_positions = []
        for i in range(height):
            if np.sum(detected_lines[i, :]) > width * 0.4:
                line_positions.append(i)
    
    else:  # vertical
        # Use longer kernel for vertical lines (should span most of height)
        kernel_length = height // 4  # Lines should span at least 25% of height
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        # Detect lines
        detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        detected_lines = cv2.dilate(detected_lines, vertical_kernel, iterations=1)
        
        # Find line positions (more strict - line should span at least 40% of height)
        line_positions = []
        for j in range(width):
            if np.sum(detected_lines[:, j]) > height * 0.4:
                line_positions.append(j)
    
    return sorted(line_positions)


def merge_close_lines(line_positions: List[int], min_distance: int = 20) -> List[int]:
    """Merge lines that are too close together."""
    if not line_positions:
        return []
    
    merged = [line_positions[0]]
    
    for pos in line_positions[1:]:
        if pos - merged[-1] >= min_distance:
            merged.append(pos)
        else:
            # Merge: keep the line closer to the middle of the range
            merged[-1] = (merged[-1] + pos) // 2
    
    return merged


def extract_grid_cells(image: Image.Image, 
                       horizontal_lines: List[int], 
                       vertical_lines: List[int]) -> List[Tuple[int, Image.Image]]:
    """Extract individual cells from the grid."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    cells = []
    
    # Ensure we have boundary lines
    h_lines = [0] + horizontal_lines + [height]
    v_lines = [0] + vertical_lines + [width]
    
    # Extract cells
    cell_index = 0
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]
            
            # Skip very small cells
            cell_height = y2 - y1
            cell_width = x2 - x1
            
            if cell_height < 80 or cell_width < 150:
                continue
            
            # Extract cell
            cell = img_array[y1:y2, x1:x2]
            cell_image = Image.fromarray(cell)
            
            cells.append((cell_index, cell_image))
            cell_index += 1
    
    return cells


def filter_blocks_strict(blocks: List[Image.Image], page_height: int, page_width: int) -> List[Image.Image]:
    """Apply stricter filtering to remove invalid blocks."""
    filtered = []
    
    for block in blocks:
        h, w = block.size[1], block.size[0]
        
        # Stricter criteria:
        # - Height should be between 150-400 pixels
        # - Width should be between 300-600 pixels
        # - Area should be reasonable (not too small)
        
        if (150 < h < 400 and 
            300 < w < 600 and
            w * h > 50000):  # Minimum area
            filtered.append(block)
    
    return filtered


def detect_voter_blocks(image: Image.Image) -> List[Image.Image]:
    """
    Detect and extract individual voter blocks from a page image.
    Uses contour-based detection as primary method (proven to work better for this PDF format).
    Falls back to grid detection if contour method fails.
    """
    try:
        logger.info("Starting block detection using contour method...")
        
        # Use contour-based detection as primary method (works better for this PDF format)
        blocks = extract_blocks_fallback(image)
        
        if blocks and len(blocks) > 0:
            logger.info(f"Extracted {len(blocks)} voter blocks using contour method")
            return blocks
        
        # If contour method fails, try grid detection as backup
        logger.info("Contour method failed, trying grid detection as backup...")
        binary = preprocess_image(image)
        
        horizontal_lines = detect_grid_lines(binary, 'horizontal')
        vertical_lines = detect_grid_lines(binary, 'vertical')
        
        horizontal_lines = merge_close_lines(horizontal_lines, min_distance=30)
        vertical_lines = merge_close_lines(vertical_lines, min_distance=100)
        
        logger.info(f"Grid detection found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            cells = extract_grid_cells(image, horizontal_lines, vertical_lines)
            
            voter_blocks = []
            height, width = image.size[1], image.size[0]
            
            for cell_idx, cell in cells:
                cell_h, cell_w = cell.size[1], cell.size[0]
                aspect_ratio = cell_w / cell_h if cell_h > 0 else 0
                
                if (100 < cell_h < height * 0.35 and 
                    200 < cell_w < width * 0.7 and
                    1.5 < aspect_ratio < 5.0):
                    voter_blocks.append(cell)
            
            if voter_blocks:
                logger.info(f"Grid detection extracted {len(voter_blocks)} blocks")
                
                # If we got too many blocks, try stricter filtering
                if len(voter_blocks) > 50:
                    logger.warning(f"Too many blocks detected ({len(voter_blocks)}), applying stricter filtering")
                    voter_blocks = filter_blocks_strict(voter_blocks, height, width)
                    logger.info(f"After strict filtering: {len(voter_blocks)} blocks")
                
                return voter_blocks
        
        logger.warning("All detection methods failed, returning empty list")
        return []
        
    except Exception as e:
        logger.error(f"Error in block detection: {e}")
        return []


def extract_blocks_fallback(image: Image.Image) -> List[Image.Image]:
    """
    Extract voter blocks using contour detection.
    This is the primary method as it works well for this PDF format.
    """
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours - use RETR_TREE to find nested contours (voter grids inside table structure)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        height, width = image.size[1], image.size[0]
        
        # Store blocks with their positions for proper sorting
        blocks_with_pos = []
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Relaxed filtering for voter blocks to catch more grids
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # More lenient criteria:
            # - Height should be reasonable (100-500 pixels, or up to 40% of page height)
            # - Width should be reasonable (200-800 pixels, or up to 80% of page width)
            # - Aspect ratio should be reasonable (1.2-6.0 for voter blocks)
            # - Minimum area should be reasonable (30000 instead of 50000)
            
            if (100 < h < max(500, height * 0.4) and 
                200 < w < max(800, width * 0.8) and
                1.2 < aspect_ratio < 6.0 and
                area > 30000):
                
                # Extract block with padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                block = img_array[y1:y2, x1:x2]
                block_image = Image.fromarray(block)
                
                # Store with position for sorting
                blocks_with_pos.append((y, x, block_image))
        
        # Sort blocks by position (top to bottom, then left to right)
        blocks_with_pos.sort(key=lambda b: (b[0], b[1]))
        
        # Remove duplicate/overlapping blocks
        filtered_blocks = []
        for y, x, block_img in blocks_with_pos:
            # Check if this block overlaps significantly with any existing block
            is_duplicate = False
            for existing_y, existing_x, existing_block in filtered_blocks:
                # Calculate overlap
                overlap_threshold = 0.7  # 70% overlap means duplicate
                
                # Simple overlap check based on position and size
                y_diff = abs(y - existing_y)
                x_diff = abs(x - existing_x)
                h_overlap = min(block_img.size[1], existing_block.size[1])
                w_overlap = min(block_img.size[0], existing_block.size[0])
                
                if y_diff < h_overlap * overlap_threshold and x_diff < w_overlap * overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_blocks.append((y, x, block_img))
        
        blocks = [block for _, _, block in filtered_blocks]
        
        logger.info(f"Contour method extracted {len(blocks)} blocks (after deduplication)")
        return blocks
        
    except Exception as e:
        logger.error(f"Error in contour extraction: {e}")
        return [image]
