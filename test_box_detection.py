#!/usr/bin/env python3
"""
Box Detection Test Script
==========================
Test script to detect grids, detect boxes inside grids, and save segmented images
for verification.

This script:
1. Detects all grids using grid_detector.py
2. For each grid, detects boxes inside it using box_detector.py
3. Colors boxes white and segments the grid
4. Saves segmented images in separate directories for each grid
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Import modules
from config import INPUT_DIR, TEMP_DIR, SKIP_FIRST_N_PAGES, SKIP_LAST_N_PAGES
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from box_detector import process_grid

# Setup
BOX_OUTPUT_DIR = Path(TEMP_DIR) / "box_detection"


def setup_box_output_dir():
    """Create box detection output directory."""
    BOX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Box detection output directory: {BOX_OUTPUT_DIR}")


def save_grid_segments(grid_idx: int, page_num: int, pdf_name: str, grid_data: dict):
    """
    Save segmented images for a grid.
    
    Args:
        grid_idx (int): Index of the grid
        page_num (int): Page number
        pdf_name (str): Name of the PDF file
        grid_data (dict): Dictionary containing processed grid data
    """
    grid_dir = BOX_OUTPUT_DIR / f"{pdf_name}_page_{page_num}_grid_{grid_idx + 1:03d}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original grid
    original_path = grid_dir / "00_original_grid.jpg"
    grid_data['original_image'].save(original_path, 'JPEG', quality=95)
    
    # Save image with white boxes
    white_boxes_path = grid_dir / "01_grid_with_white_boxes.jpg"
    grid_data['image_with_white_boxes'].save(white_boxes_path, 'JPEG', quality=95)
    
    # Save left half (details)
    left_half_path = grid_dir / "02_left_half_details.jpg"
    grid_data['left_half'].save(left_half_path, 'JPEG', quality=95)
    
    # Save right half (EPIC number)
    right_half_path = grid_dir / "03_right_half_epic.jpg"
    grid_data['right_half'].save(right_half_path, 'JPEG', quality=95)
    
    # Save image with boxes drawn (for visualization)
    if 'boxes' in grid_data and grid_data['boxes']:
        import cv2
        img_array = np.array(grid_data['original_image'])
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Draw all boxes
        for box in grid_data['boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw serial boxes in blue
        for box in grid_data['serial_boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw photo box in red
        if grid_data['photo_box']:
            x1, y1, x2, y2 = grid_data['photo_box']
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        boxes_visualization = Image.fromarray(img_array)
        boxes_path = grid_dir / "04_boxes_visualization.jpg"
        boxes_visualization.save(boxes_path, 'JPEG', quality=95)
    
    # Save cropped box images
    if 'cropped_box_images' in grid_data and grid_data['cropped_box_images']:
        boxes_dir = grid_dir / "boxes"
        boxes_dir.mkdir(exist_ok=True)
        
        # Create mapping of box coordinates to indices
        boxes_list = grid_data['boxes']
        cropped_images = grid_data['cropped_box_images']
        
        # Sort boxes by position (top-to-bottom, then left-to-right) for consistent naming
        sorted_indices = sorted(range(len(boxes_list)), key=lambda i: (boxes_list[i][1], boxes_list[i][0]))
        
        for idx, original_idx in enumerate(sorted_indices):
            box_coords = boxes_list[original_idx]
            box_image = cropped_images[original_idx]
            
            # Determine if it's a serial box or photo box
            if grid_data['photo_box'] and box_coords == grid_data['photo_box']:
                box_type = "photo"
                box_num = len(grid_data['serial_boxes']) + 1
            else:
                box_type = "serial"
                # Find which serial box this is
                box_num = next((i+1 for i, sb in enumerate(grid_data['serial_boxes']) if sb == box_coords), idx + 1)
            
            box_filename = f"05_box_{idx+1:02d}_{box_type}_{box_num:02d}.jpg"
            box_path = boxes_dir / box_filename
            box_image.save(box_path, 'JPEG', quality=95)
    
    # Save metadata
    metadata_path = grid_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Grid Index: {grid_idx + 1}\n")
        f.write(f"Page Number: {page_num}\n")
        f.write(f"Total Boxes Detected: {len(grid_data['boxes'])}\n")
        f.write(f"Serial Number Boxes: {len(grid_data['serial_boxes'])}\n")
        f.write(f"Photo Box: {'Yes' if grid_data['photo_box'] else 'No'}\n\n")
        
        f.write("Serial Box Coordinates:\n")
        for i, box in enumerate(grid_data['serial_boxes']):
            f.write(f"  Box {i+1}: {box}\n")
        
        if grid_data['photo_box']:
            f.write(f"\nPhoto Box Coordinates: {grid_data['photo_box']}\n")
        
        if 'cropped_box_images' in grid_data:
            f.write(f"\nCropped Box Images: {len(grid_data['cropped_box_images'])} saved in 'boxes/' directory\n")
    
    return grid_dir


def test_box_detection(pdf_path: str):
    """
    Test box detection on a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    """
    pdf_name = Path(pdf_path).stem
    print("\n" + "="*70)
    print(f"Testing Box Detection: {pdf_name}")
    print("="*70)
    
    # Step 1: Convert PDF to images
    print("\n📄 Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)
    
    if not page_images:
        print("❌ No pages to process")
        return
    
    print(f"✅ Found {len(page_images)} pages to process")
    
    total_grids = 0
    total_boxes = 0
    
    # Step 2: Process each page
    for page_num, image in page_images:
        print(f"\n{'─'*70}")
        print(f"📖 Processing Page {page_num}...")
        print(f"   Image size: {image.size[0]} x {image.size[1]} pixels")
        
        # Step 3: Detect voter blocks (grids)
        print(f"   Detecting grids...")
        voter_blocks = detect_voter_blocks(image)
        
        if not voter_blocks:
            print(f"   ⚠️  No grids detected on page {page_num}")
            continue
        
        print(f"   ✅ Detected {len(voter_blocks)} grids")
        
        # Step 4: Process each grid
        for grid_idx, grid_image in enumerate(voter_blocks):
            print(f"\n   🔲 Processing Grid {grid_idx + 1}/{len(voter_blocks)}...")
            print(f"      Grid size: {grid_image.size[0]} x {grid_image.size[1]} pixels")
            
            try:
                # Process grid: detect boxes, color white, segment
                grid_data = process_grid(grid_image)
                
                num_boxes = len(grid_data['boxes'])
                num_serial = len(grid_data['serial_boxes'])
                has_photo = grid_data['photo_box'] is not None
                
                print(f"      ✅ Detected {num_boxes} boxes")
                print(f"         - Serial boxes: {num_serial}")
                print(f"         - Photo box: {'Yes' if has_photo else 'No'}")
                
                # Save segmented images
                grid_dir = save_grid_segments(grid_idx, page_num, pdf_name, grid_data)
                print(f"      💾 Saved segments to: {grid_dir.name}/")
                
                total_grids += 1
                total_boxes += num_boxes
                
            except Exception as e:
                print(f"      ❌ Error processing grid {grid_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n   ✅ Processed {len(voter_blocks)} grids from page {page_num}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"PDF: {pdf_name}")
    print(f"Pages processed: {len(page_images)}")
    print(f"Total grids processed: {total_grids}")
    print(f"Total boxes detected: {total_boxes}")
    print(f"Output location: {BOX_OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    print(f"✅ All grid segments saved successfully!")
    print(f"   Check the '{BOX_OUTPUT_DIR}/' folder to inspect the segmented grids.")
    print(f"\n   Each grid has its own directory containing:")
    print(f"   - 00_original_grid.jpg: Original grid image")
    print(f"   - 01_grid_with_white_boxes.jpg: Grid with detected boxes colored white")
    print(f"   - 02_left_half_details.jpg: Left portion (60%) containing details")
    print(f"   - 03_right_half_epic.jpg: Right portion (40%) containing EPIC number")
    print(f"   - 04_boxes_visualization.jpg: Visualization of detected boxes")
    print(f"   - boxes/ directory: Individual cropped box images")
    print(f"   - metadata.txt: Information about detected boxes")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Box Detection Test Script")
    print("="*70)
    
    # Setup output directory
    setup_box_output_dir()
    
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
    test_box_detection(str(pdf_files[0]))
    
    print("\n💡 Next steps:")
    print("   1. Check the segmented grids in 'temp_images/box_detection/' folder")
    print("   2. Verify that boxes are detected correctly")
    print("   3. Verify that left/right halves are segmented properly")
    print("   4. Check that serial boxes and photo box are identified correctly")
    print("\n")


if __name__ == "__main__":
    main()

