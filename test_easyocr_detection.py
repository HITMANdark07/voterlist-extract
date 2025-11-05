#!/usr/bin/env python3
"""
Test Script for EasyOCR Detection
===================================
Tests EasyOCR detection on block split images and compares with hardcoded approach.
"""

import logging
from pathlib import Path
from PIL import Image

from ocr_detector import detect_text_regions, classify_detected_regions, extract_regions_from_detection
from block_splitter import split_voter_block_hardcoded, split_voter_block
from config import TEMP_DIR

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to block split directory
BLOCK_SPLIT_DIR = Path(TEMP_DIR) / "block_split"


def visualize_detection(block_image: Image.Image, bounding_boxes, output_path: Path):
    """Visualize detected bounding boxes on the block image."""
    try:
        import cv2
        import numpy as np
        
        img_array = np.array(block_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, confidence = box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{confidence:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(str(output_path), img_bgr)
        logger.info(f"Saved visualization to {output_path}")
    except ImportError:
        logger.warning("OpenCV not available, skipping visualization")


def test_easyocr_detection(limit: int = 10):
    """Test EasyOCR detection on block images."""
    print("\n" + "="*70)
    print("EasyOCR Detection Test")
    print("="*70 + "\n")
    
    if not BLOCK_SPLIT_DIR.exists():
        print(f"❌ Block split directory not found: {BLOCK_SPLIT_DIR}")
        print("Please run test_block_splitter.py first to generate block images.")
        return
    
    # Get all numbered folders
    folders = sorted([f for f in BLOCK_SPLIT_DIR.iterdir() if f.is_dir()])[:limit]
    
    if not folders:
        print(f"❌ No folders found in {BLOCK_SPLIT_DIR}")
        return
    
    print(f"📁 Testing on {len(folders)} block(s)\n")
    
    easyocr_success = 0
    hardcoded_only = 0
    total = 0
    
    for folder_path in folders:
        block_path = folder_path / "block.jpg"
        if not block_path.exists():
            continue
        
        total += 1
        folder_name = folder_path.name
        print(f"\n{'─'*70}")
        print(f"Testing block: {folder_name}")
        
        try:
            block_image = Image.open(block_path)
            width, height = block_image.size
            print(f"  Block size: {width}x{height}")
            
            # Test EasyOCR detection
            print("\n  🔍 EasyOCR Detection:")
            bounding_boxes = detect_text_regions(block_image)
            print(f"    Detected {len(bounding_boxes)} text regions")
            
            if bounding_boxes:
                classified = classify_detected_regions(bounding_boxes, width, height)
                print(f"    Serial: {'✅' if classified['serial_no'] else '❌'}")
                print(f"    EPIC: {'✅' if classified['epic'] else '❌'}")
                print(f"    Details: {len(classified['details'])} boxes")
                
                # Visualize detection
                vis_path = folder_path / "easyocr_detection.jpg"
                visualize_detection(block_image, bounding_boxes, vis_path)
                
                # Try full detection pipeline
                regions = split_voter_block(block_image)
                if regions['serial_no'] and regions['epic']:
                    print("    ✅ EasyOCR detection successful")
                    easyocr_success += 1
                    
                    # Save EasyOCR regions for comparison
                    regions['serial_no'].save(folder_path / "serial_no_easyocr.jpg", 'JPEG', quality=95)
                    regions['epic'].save(folder_path / "epic_easyocr.jpg", 'JPEG', quality=95)
                    regions['details'].save(folder_path / "details_easyocr.jpg", 'JPEG', quality=95)
                else:
                    print("    ⚠️  EasyOCR detection incomplete")
                    hardcoded_only += 1
            else:
                print("    ❌ No text regions detected")
                hardcoded_only += 1
            
            # Compare with hardcoded
            print("\n  📐 Hardcoded Split:")
            hardcoded_regions = split_voter_block_hardcoded(block_image)
            print(f"    All regions extracted: ✅")
            
        except Exception as e:
            logger.error(f"Error processing {folder_name}: {e}", exc_info=True)
            continue
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total blocks tested: {total}")
    print(f"EasyOCR successful: {easyocr_success}")
    print(f"Hardcoded fallback used: {hardcoded_only}")
    print(f"Success rate: {easyocr_success/total*100:.1f}%" if total > 0 else "N/A")
    print("="*70 + "\n")
    
    print("✅ Test complete!")
    print(f"📁 Check individual folders for comparison images:")
    print(f"   - easyocr_detection.jpg (visualization)")
    print(f"   - serial_no_easyocr.jpg, epic_easyocr.jpg, details_easyocr.jpg")


if __name__ == "__main__":
    test_easyocr_detection(limit=20)  # Test first 20 blocks

