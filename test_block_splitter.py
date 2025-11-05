#!/usr/bin/env python3
"""
Block Splitter Test Script
==========================
Test script to visualize block splitting and save split regions.

Tests the block_splitter module by:
1. Extracting voter blocks from PDF (using grid detection)
2. Splitting each block into regions
3. Saving original block and split regions to disk
"""

import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2

# Import modules
from config import INPUT_DIR, TEMP_DIR
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from block_splitter import split_voter_block

# Setup
BLOCK_SPLIT_OUTPUT_DIR = Path(TEMP_DIR) / "block_split"


def sharpen_image(image: Image.Image) -> Image.Image:
    """
    Sharpen the image using unsharp mask filter.
    
    Args:
        image: PIL Image object
        
    Returns:
        Sharpened PIL Image object
    """
    # Apply unsharp mask filter (radius=2, percent=150, threshold=3)
    # This enhances edges and improves OCR accuracy
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return sharpened


def remove_lines_and_boxes(image: Image.Image) -> Image.Image:
    """
    Remove non-character lines and boxes from an image region.
    This helps clean up serial numbers, EPIC numbers, and details regions
    by removing grid lines and borders that interfere with OCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        Cleaned PIL Image object with lines/boxes removed
    """
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Create a copy for processing
        cleaned = gray.copy()
        
        # Convert to binary for better line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect and remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts_h = cv2.findContours(detected_lines_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1]
        for c in cnts_h:
            # Fill detected horizontal lines with white (background color)
            cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 3)
        
        # Detect and remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detected_lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts_v = cv2.findContours(detected_lines_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1]
        for c in cnts_v:
            # Fill detected vertical lines with white (background color)
            cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 3)
        
        # Detect and remove rectangular boxes (boundaries)
        # Find contours that might be boxes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            img_area = width * height
            
            # Check if this is likely a border box (large area, near edges)
            is_border = (
                (x < 5 or y < 5 or x + w > width - 5 or y + h > height - 5) and
                area > img_area * 0.3 and  # Large area
                (w > width * 0.7 or h > height * 0.7)  # Spans most of width/height
            )
            
            if is_border:
                # Fill the border with white
                cv2.drawContours(cleaned, [contour], -1, (255, 255, 255), 3)
                # Also fill the rectangle area
                cv2.rectangle(cleaned, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
        # Convert back to PIL Image
        # Ensure it's RGB if original was RGB
        if len(img_array.shape) == 3:
            cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(cleaned_rgb)
        else:
            return Image.fromarray(cleaned)
            
    except Exception as e:
        print(f"   ⚠️  Error removing lines/boxes: {e}, using original image")
        return image  # Return original on error


def setup_output_dir():
    """Create block split output directory."""
    BLOCK_SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Block split output directory: {BLOCK_SPLIT_OUTPUT_DIR}")


def test_block_splitter(pdf_path: str):
    """
    Test block splitter on a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    """
    pdf_name = Path(pdf_path).stem
    print("\n" + "="*70)
    print(f"Testing Block Splitter: {pdf_name}")
    print("="*70)
    
    # Step 1: Convert PDF to images
    print("\n📄 Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)
    
    if not page_images:
        print("❌ No pages to process")
        return
    
    print(f"✅ Found {len(page_images)} pages to process")
    
    total_blocks = 0
    sequence = 1
    
    # Step 2: Process each page
    for page_num, image in page_images:
        print(f"\n{'─'*70}")
        print(f"📖 Processing Page {page_num}...")
        print(f"   Image size: {image.size[0]} x {image.size[1]} pixels")
        
        # Step 3: Detect voter blocks
        print(f"   Detecting grid and extracting blocks...")
        voter_blocks = detect_voter_blocks(image)
        
        if not voter_blocks:
            print(f"   ⚠️  No blocks detected on page {page_num}")
            continue
        
        print(f"   ✅ Detected {len(voter_blocks)} voter blocks")
        
        # Step 4: Split each block and save regions
        for block_idx, block_image in enumerate(voter_blocks):
            try:
                # Sharpen the block image before processing
                sharpened_block = sharpen_image(block_image)
                
                # Create sequence folder
                seq_folder = BLOCK_SPLIT_OUTPUT_DIR / f"{sequence:04d}"
                seq_folder.mkdir(parents=True, exist_ok=True)
                
                # Save original block (sharpened)
                block_path = seq_folder / "block.jpg"
                sharpened_block.save(block_path, 'JPEG', quality=95)
                
                # Split block into regions (using sharpened image)
                regions = split_voter_block(sharpened_block)
                
                # Clean serial_no and epic regions: remove lines and boxes before saving
                # Keep details region as-is (no cleaning)
                cleaned_regions = {
                    'serial_no': remove_lines_and_boxes(regions['serial_no']),
                    'epic': remove_lines_and_boxes(regions['epic']),
                    'details': regions['details']  # Keep original, no cleaning
                }
                
                # Save each cleaned region
                serial_path = seq_folder / "serial_no.jpg"
                epic_path = seq_folder / "epic.jpg"
                details_path = seq_folder / "details.jpg"
                
                cleaned_regions['serial_no'].save(serial_path, 'JPEG', quality=95)
                cleaned_regions['epic'].save(epic_path, 'JPEG', quality=95)
                cleaned_regions['details'].save(details_path, 'JPEG', quality=95)
                
                # Update print statement to use cleaned regions
                regions = cleaned_regions
                
                # Print region sizes
                print(f"   Block {block_idx + 1}: Serial={regions['serial_no'].size}, "
                      f"EPIC={regions['epic'].size}, Details={regions['details'].size}")
                
                total_blocks += 1
                sequence += 1
                
            except Exception as e:
                print(f"   ❌ Error processing block {block_idx + 1}: {e}")
                continue
        
        print(f"   💾 Saved {len(voter_blocks)} blocks from page {page_num}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"PDF: {pdf_name}")
    print(f"Pages processed: {len(page_images)}")
    print(f"Total blocks processed: {total_blocks}")
    print(f"Output location: {BLOCK_SPLIT_OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    print(f"✅ All blocks and split regions saved successfully!")
    print(f"\n📁 Check the '{BLOCK_SPLIT_OUTPUT_DIR}/' folder:")
    print(f"   Each sequence folder contains:")
    print(f"     - block.jpg (original block)")
    print(f"     - serial_no.jpg (serial number region)")
    print(f"     - epic.jpg (EPIC number region)")
    print(f"     - details.jpg (details region)")
    print()


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Block Splitter Test Script")
    print("="*70)
    
    # Setup output directory
    setup_output_dir()
    
    # Find PDF files
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in '{INPUT_DIR}' directory.")
        print(f"Please place a PDF file in '{INPUT_DIR}' and run again.")
        return
    
    if len(pdf_files) > 1:
        print(f"\n⚠️  Found {len(pdf_files)} PDF files. Processing the first one:")
        print(f"   {pdf_files[0].name}")
        print(f"\nTo test a specific file, modify the script or pass it as argument.")
    
    # Test first PDF
    test_block_splitter(str(pdf_files[0]))
    
    print("\n💡 Next steps:")
    print("   1. Check the split regions in 'temp_images/block_split/' folder")
    print("   2. Verify that regions are correctly split:")
    print("      - serial_no.jpg should contain only the serial number")
    print("      - epic.jpg should contain only the EPIC number")
    print("      - details.jpg should contain voter details (name, relation, etc.)")
    print("   3. If regions look good, the region-based OCR should work better")
    print("\n")


if __name__ == "__main__":
    main()

