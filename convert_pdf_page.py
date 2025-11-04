#!/usr/bin/env python3
"""
PDF Page to Image Converter
===========================
Convert a specific page from a PDF to an image for testing.

Usage:
    python convert_pdf_page.py <pdf_file> <page_number> [output_name]
    
Example:
    python convert_pdf_page.py input_pdfs/booth_001.pdf 3
    python convert_pdf_page.py input_pdfs/booth_001.pdf 3 test_page.jpg
"""

import sys
from pathlib import Path

# Import modules
from pdf_converter import convert_single_page
from config import TEMP_DIR, IMAGE_DPI


def convert_page_to_image(pdf_path: str, page_number: int, output_name: str = None) -> bool:
    """
    Convert a specific PDF page to an image.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (1-indexed)
        output_name: Optional output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        print(f"\n📄 Converting page {page_number} from: {pdf_path}")
        print(f"⚙️  DPI: {IMAGE_DPI}")
        print("⏳ Processing...")
        
        # Generate output filename
        if output_name is None:
            pdf_name = Path(pdf_path).stem
            output_name = f"{pdf_name}_page_{page_number}.jpg"
        
        # Ensure .jpg extension
        if not output_name.lower().endswith(('.jpg', '.jpeg')):
            output_name += '.jpg'
        
        output_path = Path(TEMP_DIR) / output_name
        
        # Convert using the module function
        # Note: We need to use pdf2image directly here since convert_single_page
        # doesn't support custom output paths yet
        from pdf2image import convert_from_path
        
        images = convert_from_path(
            pdf_path,
            dpi=IMAGE_DPI,
            first_page=page_number,
            last_page=page_number,
            fmt='jpeg'
        )
        
        if not images:
            print(f"❌ Error: Could not convert page {page_number}")
            return False
        
        # Save image
        images[0].save(output_path, 'JPEG', quality=95)
        
        # Get file size
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n✅ Success!")
        print(f"   Output: {output_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Resolution: {images[0].size[0]} x {images[0].size[1]} pixels")
        
        print(f"\n💡 Next step: Test OCR on this image:")
        print(f"   python test_single_image.py {output_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ Error: PDF file not found: {pdf_path}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("\n" + "="*70)
    print("PDF Page to Image Converter")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 3:
        print("\n❌ Error: Missing required arguments")
        print("\nUsage:")
        print("  python convert_pdf_page.py <pdf_file> <page_number> [output_name]")
        print("\nExamples:")
        print("  python convert_pdf_page.py input_pdfs/booth_001.pdf 3")
        print("  python convert_pdf_page.py input_pdfs/booth_001.pdf 5 test_page.jpg")
        print("\nNote: Page numbers start from 1")
        print("      Typical voter pages are 3 onwards")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Parse page number
    try:
        page_number = int(sys.argv[2])
        if page_number < 1:
            print("\n❌ Error: Page number must be at least 1")
            sys.exit(1)
    except ValueError:
        print(f"\n❌ Error: Invalid page number: {sys.argv[2]}")
        sys.exit(1)
    
    # Optional output name
    output_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Convert page
    success = convert_page_to_image(pdf_path, page_number, output_name)
    
    if not success:
        print("\n" + "="*70)
        print("❌ Conversion Failed")
        print("="*70)
        print("\nPossible issues:")
        print("  - PDF file doesn't exist")
        print("  - Page number is out of range")
        print("  - Poppler not installed (run: brew install poppler)")
        print("  - pdf2image not installed (run: pip install pdf2image)")
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

