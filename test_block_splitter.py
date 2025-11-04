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
from PIL import Image

# Import modules
from config import INPUT_DIR, TEMP_DIR
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from block_splitter import split_voter_block

# Setup
BLOCK_SPLIT_OUTPUT_DIR = Path(TEMP_DIR) / "block_split"


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
                # Create sequence folder
                seq_folder = BLOCK_SPLIT_OUTPUT_DIR / f"{sequence:04d}"
                seq_folder.mkdir(parents=True, exist_ok=True)
                
                # Save original block
                block_path = seq_folder / "block.jpg"
                block_image.save(block_path, 'JPEG', quality=95)
                
                # Split block into regions
                regions = split_voter_block(block_image)
                
                # Save each region
                serial_path = seq_folder / "serial_no.jpg"
                epic_path = seq_folder / "epic.jpg"
                details_path = seq_folder / "details.jpg"
                
                regions['serial_no'].save(serial_path, 'JPEG', quality=95)
                regions['epic'].save(epic_path, 'JPEG', quality=95)
                regions['details'].save(details_path, 'JPEG', quality=95)
                
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

