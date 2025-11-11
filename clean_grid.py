#!/usr/bin/env python3
"""
Voter Image Cleaner
===================
Processes already extracted voter images from output_images/{pdf_name}/voters/:
- For each voter image:
  - Removes outer border rectangle
  - Finds and removes small boxes inside it using find_inner_boxes
  - Saves the cleaned image
- Saves to "voter split" folder
"""

import logging
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import List, Tuple, Optional

from config import OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_outer_border(image: Image.Image, 
                        border_threshold: float = 0.85) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the outer border rectangle of the image.
    
    Args:
        image: PIL Image object
        border_threshold: Minimum area threshold to consider as outer border (default: 0.85 = 85% of image)
    
    Returns:
        Tuple of (x, y, width, height) of the outer border, or None if not found
    """
    # Convert PIL Image to numpy array (grayscale)
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    # Binarize (invert: make lines white)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological close to strengthen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Use RETR_EXTERNAL to get only outer contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = img.shape
    img_area = img_h * img_w
    
    # Find the largest contour that covers most of the image (outer border)
    outer_border = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Check if this is a large border (covers most of the image)
        if area > border_threshold * img_area:
            # Keep only rectangular-like contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4 and area > max_area:
                max_area = area
                outer_border = (x, y, w, h)
    
    return outer_border


def remove_outer_border(image: Image.Image, 
                       border_threshold: float = 0.85,
                       padding: int = 5) -> Image.Image:
    """
    Remove the outer border rectangle from the image by cropping inside it.
    
    Args:
        image: PIL Image object
        border_threshold: Minimum area threshold to consider as outer border
        padding: Padding to add inside the border when cropping
    
    Returns:
        Image with outer border removed (cropped)
    """
    outer_border = detect_outer_border(image, border_threshold)
    
    if outer_border is None:
        logger.debug("No outer border detected, returning original image")
        return image
    
    x, y, w, h = outer_border
    img_width, img_height = image.size
    
    # Crop inside the border with padding
    # The border is at (x, y) with size (w, h), so we crop from (x+padding, y+padding) 
    # to (x+w-padding, y+h-padding)
    crop_x1 = max(0, x + padding)
    crop_y1 = max(0, y + padding)
    crop_x2 = min(img_width, x + w - padding)
    crop_y2 = min(img_height, y + h - padding)
    
    # Ensure valid crop coordinates
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        logger.warning("Invalid crop coordinates, returning original image")
        return image
    
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    logger.debug(f"Removed outer border: cropped from ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    
    return cropped


def find_inner_boxes(image: Image.Image, 
                     min_box_width: int = 20, 
                     min_box_height: int = 20, 
                     border_threshold: float = 0.9) -> List[List[int]]:
    """
    Detect all boxes (inner + outer) from structured form-like images.
    Automatically ignores the largest outer border box.

    Args:
        image: PIL Image object
        min_box_width: Minimum width of valid boxes
        min_box_height: Minimum height of valid boxes
        border_threshold: Ignore boxes covering > threshold * image area

    Returns:
        boxes: List of detected box coordinates [x1, y1, x2, y2]
    """
    # Convert PIL Image to numpy array (grayscale)
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array
    
    # Binarize (invert: make lines white)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological close to strengthen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Use RETR_TREE to get nested boxes (not just outer)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_h, img_w = img.shape
    img_area = img_h * img_w
    
    # Loop over contours
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
    
    # Sort boxes top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    return boxes


def remove_boxes_from_image(image: Image.Image, 
                           boxes: List[List[int]],
                           fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Remove detected boxes from the image by filling them with background color.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2] to remove
        fill_color: Color to fill the boxes with (default: white)
    
    Returns:
        Image with boxes removed
    """
    img_array = np.array(image).copy()
    
    # Fill each box
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Add small padding to ensure complete removal
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_array.shape[1], x2 + padding)
        y2 = min(img_array.shape[0], y2 + padding)
        
        if len(img_array.shape) == 2:  # Grayscale
            img_array[y1:y2, x1:x2] = 255  # White for grayscale
        elif len(img_array.shape) == 3:  # Color image
            if img_array.shape[2] == 3:  # RGB
                img_array[y1:y2, x1:x2] = fill_color[:3]
            elif img_array.shape[2] == 4:  # RGBA
                img_array[y1:y2, x1:x2] = (*fill_color[:3], 255)
            else:
                img_array[y1:y2, x1:x2] = 255
    
    return Image.fromarray(img_array)


def split_image(image: Image.Image, left_percent: float = 0.6) -> Tuple[Image.Image, Image.Image]:
    """
    Split image into left and right parts.
    
    Args:
        image: Input image
        left_percent: Percentage of width for left part (default: 0.6 for 60%)
    
    Returns:
        Tuple of (left_image, right_image)
    """
    width, height = image.size
    split_x = int(width * left_percent)
    
    left_image = image.crop((0, 0, split_x, height))
    right_image = image.crop((split_x, 0, width, height))
    
    return left_image, right_image


def process_voter_image(voter_image_path: Path,
                       output_dir: Path,
                       min_box_width: int = 20,
                       min_box_height: int = 20) -> bool:
    """
    Process a single voter image:
    - Remove outer border rectangle
    - Find small boxes inside it
    - Remove those inner boxes
    - Save the cleaned image
    
    Args:
        voter_image_path: Path to the voter image file
        output_dir: Output directory for cleaned images
        min_box_width: Minimum width for boxes to detect
        min_box_height: Minimum height for boxes to detect
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load voter image
        voter_image = Image.open(voter_image_path)
        
        # Step 1: Remove outer border rectangle
        logger.debug(f"Removing outer border from {voter_image_path.name}")
        voter_image = remove_outer_border(voter_image, border_threshold=0.85, padding=5)
        
        # Step 2: Find inner boxes in this voter image
        inner_boxes = find_inner_boxes(
            voter_image,
            min_box_width=min_box_width,
            min_box_height=min_box_height,
            border_threshold=0.9
        )
        
        logger.info(f"Found {len(inner_boxes)} inner boxes in {voter_image_path.name}")
        
        # Step 3: Remove inner boxes from the voter image
        cleaned_image = remove_boxes_from_image(voter_image, inner_boxes)
        
        # Save the cleaned image (no splitting)
        voter_name = voter_image_path.stem  # e.g., "voter_001"
        output_path = output_dir / f"{voter_name}.jpg"
        
        cleaned_image.save(output_path, "JPEG", quality=95)
        logger.info(f"Saved cleaned image: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing voter image {voter_image_path}: {e}", exc_info=True)
        return False


def process_pdf_voters(pdf_output_dir: Path, output_base_dir: Path) -> int:
    """
    Process all voter images for a single PDF.
    
    Args:
        pdf_output_dir: Directory containing the PDF's output (e.g., output_images/{pdf_name})
        output_base_dir: Base output directory for cleaned images (e.g., "voter split")
    
    Returns:
        Number of successfully processed voters
    """
    pdf_name = pdf_output_dir.name
    voters_dir = pdf_output_dir / "voters"
    
    if not voters_dir.exists():
        logger.warning(f"Voters directory not found: {voters_dir}")
        return 0
    
    # Find all voter images
    voter_images = sorted(voters_dir.glob("voter_*.jpg"))
    
    if not voter_images:
        logger.warning(f"No voter images found in {voters_dir}")
        return 0
    
    logger.info(f"Found {len(voter_images)} voter images in {pdf_name}")
    
    # Create output directory for this PDF: voter split/{pdf_name}/
    output_dir = output_base_dir / pdf_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each voter image
    success_count = 0
    for voter_image_path in voter_images:
        if process_voter_image(voter_image_path, output_dir):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(voter_images)} voters from {pdf_name}")
    return success_count


def process_all_voters(output_dir: str = None, split_output_dir: str = "voter split") -> None:
    """
    Process all voter images from output_images directory.
    
    Args:
        output_dir: Directory containing extracted voter images (default: from config)
        split_output_dir: Output directory for cleaned images
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    # Find all PDF directories in output_images
    pdf_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    if not pdf_dirs:
        logger.warning(f"No PDF directories found in {output_dir}")
        return
    
    logger.info(f"Found {len(pdf_dirs)} PDF directory(ies) to process")
    
    # Create base output directory
    split_output_path = Path(split_output_dir)
    split_output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each PDF's voters
    total_processed = 0
    for pdf_dir in pdf_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing voters from: {pdf_dir.name}")
        logger.info(f"{'='*60}")
        
        processed = process_pdf_voters(pdf_dir, split_output_path)
        total_processed += processed
        logger.info(f"Completed: {pdf_dir.name} ({processed} voters processed)\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total voters processed: {total_processed}")
    logger.info(f"Output location: {split_output_path}/")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific output directory
        output_dir = sys.argv[1]
        split_output_dir = sys.argv[2] if len(sys.argv) > 2 else "voter split"
        process_all_voters(output_dir, split_output_dir)
    else:
        # Process all voters from output_images directory
        process_all_voters()
