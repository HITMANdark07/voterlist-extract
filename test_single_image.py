#!/usr/bin/env python3
"""
Single Image OCR Test Script
============================
Test OCR on a single image file to verify setup and check extraction accuracy.

Usage:
    python test_single_image.py <path_to_image.jpg>
    
Example:
    python test_single_image.py temp_images/page_3.jpg
"""

import sys
import re
from pathlib import Path

# Import modules
from ocr_processor import perform_ocr_from_file
from text_parser import extract_voter_blocks
from config import OCR_LANGUAGE, TESSERACT_CONFIG


def extract_voter_info_simple(text: str):
    """Extract and display voter information from OCR text."""
    print("\n" + "="*70)
    print("EXTRACTED VOTER INFORMATION")
    print("="*70)
    
    # Find all EPIC numbers
    epic_pattern = r'([A-Z]{2,3}[A-Z0-9]{7,10})'
    epic_matches = re.findall(epic_pattern, text)
    
    if epic_matches:
        print(f"\n✅ Found {len(epic_matches)} EPIC numbers:")
        for i, epic in enumerate(epic_matches, 1):
            print(f"   {i}. {epic}")
    else:
        print("\n❌ No EPIC numbers found")
    
    # Find serial numbers
    serial_pattern = r'\b(\d{4})\b'
    serial_matches = re.findall(serial_pattern, text)
    if serial_matches:
        print(f"\n✅ Found {len(serial_matches)} potential serial numbers:")
        for serial in serial_matches[:10]:  # Show first 10
            print(f"   - {serial}")
    
    # Find names (निर्वाचक का नाम)
    name_pattern = r'निर्वाचक का नाम\s*[:：]\s*([^\n:]+)'
    names = re.findall(name_pattern, text)
    if names:
        print(f"\n✅ Found {len(names)} voter names:")
        for i, name in enumerate(names[:5], 1):  # Show first 5
            print(f"   {i}. {name.strip()}")
    
    # Find ages
    age_pattern = r'आयु\s*[:：]\s*(\d+)'
    ages = re.findall(age_pattern, text)
    if ages:
        print(f"\n✅ Found {len(ages)} ages:")
        print(f"   {', '.join(ages[:10])}")
    
    # Find genders
    gender_count = {
        'महिला': len(re.findall(r'महिला', text)),
        'पुरुष': len(re.findall(r'पुरुष', text))
    }
    print(f"\n✅ Gender distribution:")
    print(f"   महिला (Female): {gender_count['महिला']}")
    print(f"   पुरुष (Male): {gender_count['पुरुष']}")
    
    print("\n" + "="*70)


def save_ocr_output(text: str, output_path: str):
    """Save OCR text to a file for review."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\n💾 Full OCR text saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving OCR text: {e}")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Single Image OCR Test")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: No image path provided")
        print("\nUsage:")
        print("  python test_single_image.py <path_to_image>")
        print("\nExample:")
        print("  python test_single_image.py temp_images/page_3.jpg")
        print("  python test_single_image.py sample_voter_page.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"\n📄 Processing image: {image_path}")
    print(f"🔍 OCR Language: {OCR_LANGUAGE}")
    print(f"⚙️  Tesseract Config: {TESSERACT_CONFIG}")
    
    # Perform OCR
    print("\n⏳ Performing OCR... (this may take a few seconds)")
    text = perform_ocr_from_file(image_path)
    
    if not text.strip():
        print("\n❌ No text extracted from image!")
        print("\nPossible reasons:")
        print("  - Tesseract is not installed correctly")
        print("  - Hindi language pack is missing")
        print("  - Image quality is too low")
        print("\nTry running: python verify_setup.py")
        sys.exit(1)
    
    print("\n✅ OCR completed successfully!")
    print(f"   Total characters extracted: {len(text)}")
    print(f"   Total lines: {len(text.splitlines())}")
    
    # Save full OCR output
    output_path = Path(image_path).stem + "_ocr_output.txt"
    save_ocr_output(text, output_path)
    
    # Extract and display voter information
    extract_voter_info_simple(text)
    
    # Also try using the parser module
    print("\n" + "="*70)
    print("PARSED VOTER DATA (using text_parser module)")
    print("="*70)
    parsed_voters = extract_voter_blocks(text)
    if parsed_voters:
        print(f"\n✅ Parsed {len(parsed_voters)} voter records:")
        for i, voter in enumerate(parsed_voters[:5], 1):  # Show first 5
            print(f"\n   Voter {i}:")
            print(f"     EPIC: {voter.get('epic_no', 'N/A')}")
            print(f"     Name: {voter.get('name', 'N/A')}")
            print(f"     Age: {voter.get('age', 'N/A')}")
            print(f"     Gender: {voter.get('gender', 'N/A')}")
    else:
        print("\n❌ No voters parsed from text")
    
    # Show sample of raw OCR text
    print("\n" + "="*70)
    print("RAW OCR TEXT SAMPLE (first 500 characters)")
    print("="*70)
    print(text[:500])
    if len(text) > 500:
        print("... (truncated)")
    
    print("\n" + "="*70)
    print("✅ Test Complete!")
    print("="*70)
    print(f"\nFull OCR output saved to: {output_path}")
    print("Review this file to check OCR accuracy and adjust settings if needed.")
    print("\n")


if __name__ == "__main__":
    main()

