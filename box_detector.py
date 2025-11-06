#!/usr/bin/env python3
"""
Box Detector Module
===================
Detects boxes inside grids, colors them white, and segments the grid into
details (left half) and EPIC number (right half) sections.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def find_inner_boxes(filepath: str, min_box_width: int = 20, min_box_height: int = 20, 
                     border_threshold: float = 0.9) -> Tuple[List[List[int]], np.ndarray]:
    """
    Detect all boxes (inner + outer) from structured form-like images.
    Automatically ignores the largest outer border box.

    Args:
        filepath (str): Path to image file
        min_box_width (int): Minimum width of valid boxes
        min_box_height (int): Minimum height of valid boxes
        border_threshold (float): Ignore boxes covering > threshold * image area

    Returns:
        boxes (list): List of detected box coordinates [x1, y1, x2, y2]
        img_with_boxes (np.ndarray): Image with boxes drawn
    """
    # 1️⃣ Load grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")

    # 2️⃣ Binarize (invert: make lines white)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3️⃣ Morphological close to strengthen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4️⃣ Use RETR_TREE to get nested boxes (not just outer)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_h, img_w = img.shape
    img_area = img_h * img_w

    # 5️⃣ Loop over contours
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Skip small noisy boxes
        if w < min_box_width or h < min_box_height:
            continue

        # Skip large outer border (covers most of the image)
        if area > border_threshold * img_area:
            continue

        # Keep only rectangular-like contours
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            boxes.append([x, y, x + w, y + h])
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sort boxes top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes, img_with_boxes


def find_inner_boxes_from_image(image: Image.Image, min_box_width: int = 20, 
                                min_box_height: int = 20, border_threshold: float = 0.9) -> Tuple[List[List[int]], np.ndarray]:
    """
    Detect all boxes from a PIL Image object.
    
    Args:
        image (Image.Image): PIL Image object
        min_box_width (int): Minimum width of valid boxes
        min_box_height (int): Minimum height of valid boxes
        border_threshold (float): Ignore boxes covering > threshold * image area

    Returns:
        boxes (list): List of detected box coordinates [x1, y1, x2, y2]
        img_with_boxes (np.ndarray): Image with boxes drawn
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()

    # 2️⃣ Binarize (invert: make lines white)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3️⃣ Morphological close to strengthen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4️⃣ Use RETR_TREE to get nested boxes (not just outer)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_h, img_w = img.shape
    img_area = img_h * img_w

    # 5️⃣ Loop over contours
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Skip small noisy boxes
        if w < min_box_width or h < min_box_height:
            continue

        # Skip large outer border (covers most of the image)
        if area > border_threshold * img_area:
            continue

        # Keep only rectangular-like contours
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            boxes.append([x, y, x + w, y + h])
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sort boxes top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes, img_with_boxes


def color_boxes_white(image: Image.Image, boxes: List[List[int]]) -> Image.Image:
    """
    Color detected boxes white in the image.
    
    Args:
        image (Image.Image): PIL Image object
        boxes (list): List of box coordinates [x1, y1, x2, y2]
    
    Returns:
        Image.Image: Image with boxes colored white
    """
    img_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Color each box white
    for box in boxes:
        x1, y1, x2, y2 = box
        img_array[y1:y2, x1:x2] = 255
    
    return Image.fromarray(img_array)


def segment_grid(image: Image.Image, left_ratio: float = 0.6) -> Tuple[Image.Image, Image.Image]:
    """
    Divide grid into left portion (details) and right portion (EPIC number).
    
    Args:
        image (Image.Image): PIL Image object
        left_ratio (float): Ratio for left portion (default 0.6 for 60%)
    
    Returns:
        left_half (Image.Image): Left portion containing details (60%)
        right_half (Image.Image): Right portion containing EPIC number (40%)
    """
    width, height = image.size
    split_x = int(width * left_ratio)
    
    # Crop left and right portions
    left_half = image.crop((0, 0, split_x, height))
    right_half = image.crop((split_x, 0, width, height))
    
    return left_half, right_half


def identify_serial_and_photo_boxes(boxes: List[List[int]], grid_width: int) -> Dict[str, List[List[int]]]:
    """
    Identify serial number boxes and photo detail box from detected boxes.
    
    Rules:
    - Rightmost box is the photo detail box
    - Other boxes (2-3 boxes) are serial number boxes
    
    Args:
        boxes (list): List of box coordinates [x1, y1, x2, y2]
        grid_width (int): Width of the grid image
    
    Returns:
        dict: Dictionary with 'serial_boxes' and 'photo_box' keys
    """
    if not boxes:
        return {'serial_boxes': [], 'photo_box': None}
    
    # Sort boxes by x-coordinate (left to right)
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    
    # Rightmost box is photo box
    photo_box = sorted_boxes[-1]
    
    # All other boxes are serial number boxes (should be 2-3 boxes)
    serial_boxes = sorted_boxes[:-1]
    
    return {
        'serial_boxes': serial_boxes,
        'photo_box': photo_box
    }


def crop_boxes(image: Image.Image, boxes: List[List[int]]) -> List[Image.Image]:
    """
    Crop individual boxes from the image.
    
    Args:
        image (Image.Image): PIL Image object
        boxes (list): List of box coordinates [x1, y1, x2, y2]
    
    Returns:
        list: List of cropped box images
    """
    cropped_boxes = []
    width, height = image.size
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Crop the box
        if x2 > x1 and y2 > y1:
            cropped = image.crop((x1, y1, x2, y2))
            cropped_boxes.append(cropped)
    
    return cropped_boxes


def process_grid(image: Image.Image, min_box_width: int = 20, min_box_height: int = 20,
                 border_threshold: float = 0.9) -> Dict:
    """
    Process a single grid image:
    1. Detect boxes inside the grid
    2. Color boxes white
    3. Segment into left (60% details) and right (40% EPIC) portions
    4. Identify serial number boxes and photo box
    5. Crop individual boxes as separate images
    
    Args:
        image (Image.Image): PIL Image object representing a grid
        min_box_width (int): Minimum width of valid boxes
        min_box_height (int): Minimum height of valid boxes
        border_threshold (float): Ignore boxes covering > threshold * image area
    
    Returns:
        dict: Dictionary containing:
            - 'original_image': Original grid image
            - 'image_with_white_boxes': Image with boxes colored white
            - 'left_half': Left portion (60%) containing details
            - 'right_half': Right portion (40%) containing EPIC number
            - 'boxes': All detected boxes
            - 'serial_boxes': Serial number boxes (2-3 boxes)
            - 'photo_box': Photo detail box (rightmost)
            - 'cropped_box_images': List of cropped box images
    """
    width, height = image.size
    
    # Step 1: Detect boxes
    boxes, img_with_boxes = find_inner_boxes_from_image(
        image, min_box_width, min_box_height, border_threshold
    )
    
    logger.info(f"Detected {len(boxes)} boxes in grid")
    
    # Step 2: Color boxes white
    image_with_white_boxes = color_boxes_white(image, boxes)
    
    # Step 3: Segment grid into left (60%) and right (40%) portions
    left_half, right_half = segment_grid(image_with_white_boxes, left_ratio=0.6)
    
    # Step 4: Identify serial and photo boxes
    box_info = identify_serial_and_photo_boxes(boxes, width)
    
    # Step 5: Crop individual boxes
    cropped_box_images = crop_boxes(image, boxes)
    
    return {
        'original_image': image,
        'image_with_white_boxes': image_with_white_boxes,
        'left_half': left_half,
        'right_half': right_half,
        'boxes': boxes,
        'serial_boxes': box_info['serial_boxes'],
        'photo_box': box_info['photo_box'],
        'cropped_box_images': cropped_box_images
    }

