#!/usr/bin/env python3
"""
PaddleOCR Extractor Module
===========================
Handles OCR extraction from grid segments using PaddleOCR:
- Serial number boxes
- Left half (details - 60%)
- Right half (EPIC - 40%)
- Fallback full grid OCR
"""

import logging
import re
import numpy as np
from typing import List, Dict
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import PaddleOCR (classic)
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr")

# Try to import PaddleOCR-VL (vision-language)
try:
    from paddleocr import PaddleOCRVL  # available in newer paddleocr releases
    HAS_PADDLEOCR_VL = False
except Exception:
    HAS_PADDLEOCR_VL = False
    # Keep silent here; we'll fallback to classic and log only if both unavailable


def _get_paddleocr_engine():
    """Get or create PaddleOCR engine instance."""
    if not HAS_PADDLEOCR:
        raise ImportError("PaddleOCR is not installed. Install with: pip install paddleocr")
    
    # Create engine with Hindi language support
    # Note: GPU usage is controlled by PaddlePaddle environment, not PaddleOCR constructor
    return PaddleOCR(lang='hi', use_doc_orientation_classify=True)


def _get_paddleocr_vl_pipeline():
    """Get or create PaddleOCR-VL pipeline instance."""
    if not HAS_PADDLEOCR_VL:
        raise ImportError("PaddleOCRVL is not available in your paddleocr install.")
    # Initialize with default models; auto-rotation enabled
    # Note: model_name parameter doesn't exist, using default model names
    return PaddleOCRVL(use_doc_orientation_classify=True)


def extract_serial_numbers_from_boxes(serial_boxes: List[List[int]], 
                                      cropped_box_images: List[Image.Image],
                                      boxes: List[List[int]]) -> List[str]:
    """
    Extract serial numbers from serial number boxes using PaddleOCR.
    
    Args:
        serial_boxes: List of serial box coordinates
        cropped_box_images: List of all cropped box images
        boxes: List of all box coordinates
    
    Returns:
        List of serial number texts extracted from OCR
    """
    if not HAS_PADDLEOCR:
        logger.error("PaddleOCR not available")
        return []
    
    serial_texts = []
    ocr_engine = _get_paddleocr_engine()
    
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
            
            # Convert PIL Image to numpy array for PaddleOCR
            img_array = np.array(box_image)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Perform OCR
            try:
                result = ocr_engine.ocr(img_array)
                if result and result[0]:
                    # Combine all detected text
                    text_lines = []
                    for line in result[0]:
                        if line and len(line) > 1:
                            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                            if text.strip():
                                text_lines.append(text.strip())
                    
                    if text_lines:
                        serial_text = ' '.join(text_lines)
                        serial_texts.append(serial_text)
            except Exception as e:
                logger.debug(f"Error in PaddleOCR for serial box: {e}")
                continue
    
    return serial_texts


def _perform_paddleocr(image: Image.Image) -> str:
    """
    Perform OCR on an image using PaddleOCR.
    
    Args:
        image: PIL Image object
    
    Returns:
        Extracted text from image
    """
    if not HAS_PADDLEOCR:
        logger.error("PaddleOCR not available")
        return ""
    
    try:
        ocr_engine = _get_paddleocr_engine()
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Perform OCR
        result = ocr_engine.ocr(img_array)
        
        if result and result[0]:
            text_lines = []
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    if text.strip():
                        text_lines.append(text.strip())
            
            return '\n'.join(text_lines)
        
        return ""
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")
        return ""


def _safe_predict_vl(pipeline, image: Image.Image) -> dict:
    """
    Run PaddleOCR-VL prediction on a PIL Image.
    Tries numpy array; falls back to temporary file path if needed.
    Returns the raw result dict (or {}).
    """
    import tempfile
    import os
    try:
        # Try numpy array directly
        img_array = np.array(image)
        return pipeline.predict(img_array)  # type: ignore[attr-defined]
    except Exception:
        # Fallback to temp file path
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
            image.save(temp_path)
            try:
                result = pipeline.predict(temp_path)  # type: ignore[attr-defined]
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error(f"PaddleOCR-VL predict error: {e}")
            return {}


def _perform_paddleocr_vl(image: Image.Image) -> str:
    """
    Perform OCR on an image using PaddleOCR-VL and return concatenated text.
    """
    if not HAS_PADDLEOCR_VL:
        return ""
    try:
        pipeline = _get_paddleocr_vl_pipeline()
        result = _safe_predict_vl(pipeline, image) or {}
        # Expected structure per sample: result["text"] is a list of strings (or dicts)
        text_lines = []
        if isinstance(result, dict):
            text_field = result.get("text")
            if isinstance(text_field, list):
                for item in text_field:
                    if isinstance(item, str):
                        if item.strip():
                            text_lines.append(item.strip())
                    elif isinstance(item, dict):
                        # Some variants may return dicts with 'text' key
                        val = item.get("text")
                        if isinstance(val, str) and val.strip():
                            text_lines.append(val.strip())
        # Fallback: try to gather any string values in dict
        if not text_lines and isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, str) and v.strip():
                    text_lines.append(v.strip())
        return "\n".join(text_lines)
    except Exception as e:
        logger.error(f"PaddleOCR-VL error: {e}")
        return ""


def extract_text_from_grid_segments(grid_data: Dict, page_num: int = 0, grid_idx: int = 0) -> Dict[str, str]:
    """
    Extract text from all grid segments using PaddleOCR.
    
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
    if not (HAS_PADDLEOCR or HAS_PADDLEOCR_VL):
        logger.error("Neither PaddleOCR nor PaddleOCR-VL are available. Returning empty results.")
        return {'serial_text': '', 'epic_text': '', 'details_text': ''}
    
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
        # Prefer OCR-VL if available, else classic
        if HAS_PADDLEOCR_VL:
            result['epic_text'] = _perform_paddleocr_vl(right_half).strip()
        else:
            result['epic_text'] = _perform_paddleocr(right_half).strip()
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half OCR (EPIC): "
                    f"'{result['epic_text'][:100] if result['epic_text'] else 'EMPTY'}'")
    else:
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Right half (EPIC) is None")
    
    # Perform OCR on left half (details section - 60%)
    left_half = grid_data.get('left_half')
    if left_half:
        if HAS_PADDLEOCR_VL:
            result['details_text'] = _perform_paddleocr_vl(left_half)
        else:
            result['details_text'] = _perform_paddleocr(left_half)
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half OCR (Details): "
                    f"'{result['details_text'][:200] if result['details_text'] else 'EMPTY'}...'")
    else:
        logger.debug(f"Page {page_num}, Grid {grid_idx + 1}: Left half (Details) is None")
    
    # If no text extracted, try OCR on the whole grid with white boxes (fallback)
    if not result['epic_text'].strip() and not result['details_text'].strip():
        logger.warning(f"Page {page_num}, Grid {grid_idx + 1}: ⚠️ No text from segments, trying full grid OCR")
        image_with_white_boxes = grid_data.get('image_with_white_boxes')
        if image_with_white_boxes:
            if HAS_PADDLEOCR_VL:
                full_text = _perform_paddleocr_vl(image_with_white_boxes)
            else:
                full_text = _perform_paddleocr(image_with_white_boxes)
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

