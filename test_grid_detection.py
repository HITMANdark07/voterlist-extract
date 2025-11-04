#!/usr/bin/env python3
"""
Grid Detection Test Script
==========================
Test script to extract voter pages, detect grids, and save cropped blocks for inspection.

This script helps visualize and debug grid detection before running full extraction.
"""

import sys
from pathlib import Path
from PIL import Image

# Import modules
from config import INPUT_DIR, TEMP_DIR, SKIP_FIRST_N_PAGES, SKIP_LAST_N_PAGES
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks

# Setup
GRID_OUTPUT_DIR = Path(TEMP_DIR) / "grid"


def setup_grid_output_dir():
    """Create grid output directory."""
    GRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Grid output directory: {GRID_OUTPUT_DIR}")


def test_grid_detection(pdf_path: str):
    """
    Test grid detection on a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    """
    pdf_name = Path(pdf_path).stem
    print("\n" + "="*70)
    print(f"Testing Grid Detection: {pdf_name}")
    print("="*70)
    
    # Step 1: Convert PDF to images
    print("\n📄 Converting PDF to images...")
    page_images = pdf_to_images(pdf_path)
    
    if not page_images:
        print("❌ No pages to process")
        return
    
    print(f"✅ Found {len(page_images)} pages to process")
    
    total_blocks = 0
    
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
        
        # Step 4: Save each block
        for block_idx, block_image in enumerate(voter_blocks):
            # Generate filename
            filename = f"{pdf_name}_page_{page_num}_block_{block_idx + 1:03d}.jpg"
            output_path = GRID_OUTPUT_DIR / filename
            
            # Save block
            block_image.save(output_path, 'JPEG', quality=95)
            
            total_blocks += 1
        
        print(f"   💾 Saved {len(voter_blocks)} blocks from page {page_num}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"PDF: {pdf_name}")
    print(f"Pages processed: {len(page_images)}")
    print(f"Total blocks extracted: {total_blocks}")
    print(f"Output location: {GRID_OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    print(f"✅ All blocks saved successfully!")
    print(f"   Check the '{GRID_OUTPUT_DIR}/' folder to inspect the cropped blocks.")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Grid Detection Test Script")
    print("="*70)
    
    # Setup output directory
    setup_grid_output_dir()
    
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
    test_grid_detection(str(pdf_files[0]))
    
    print("\n💡 Next steps:")
    print("   1. Check the cropped blocks in 'temp_images/grid/' folder")
    print("   2. Verify that blocks contain individual voter data")
    print("   3. If blocks look good, run full extraction: python extract_voter_data.py")
    print("   4. If blocks need adjustment, modify thresholds in grid_detector.py")
    print("\n")


if __name__ == "__main__":
    main()

