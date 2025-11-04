#!/usr/bin/env python3
"""
Setup Verification Script
========================
Checks if all required dependencies and system tools are properly installed.
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(check_name, success, message=""):
    """Print check result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {check_name}")
    if message:
        print(f"       {message}")

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    success = version.major == 3 and version.minor >= 8
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    print_result("Python Version (≥ 3.8)", success, message)
    return success

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_result(f"Python package: {package_name}", True)
        return True
    except ImportError:
        print_result(f"Python package: {package_name}", False, 
                    f"Install with: pip install {package_name}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_result("Tesseract OCR", True, version)
            return True
        else:
            print_result("Tesseract OCR", False, "Tesseract not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_result("Tesseract OCR", False, 
                    "Install with: brew install tesseract (macOS)")
        return False

def check_tesseract_languages():
    """Check if Hindi and English language packs are installed."""
    try:
        result = subprocess.run(['tesseract', '--list-langs'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            languages = result.stdout.lower()
            
            has_hindi = 'hin' in languages
            has_english = 'eng' in languages
            
            langs_found = []
            if has_hindi:
                langs_found.append("Hindi")
            if has_english:
                langs_found.append("English")
            
            success = has_hindi and has_english
            message = f"Available: {', '.join(langs_found)}" if langs_found else "Missing required languages"
            
            print_result("Tesseract Languages (Hindi + English)", success, message)
            
            if not has_hindi:
                print("       Install Hindi: brew install tesseract-lang (macOS)")
            
            return success
        else:
            print_result("Tesseract Languages", False, "Cannot check languages")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_result("Tesseract Languages", False, "Tesseract not available")
        return False

def check_poppler():
    """Check if Poppler (PDF conversion tools) is installed."""
    try:
        # Check for pdftoppm which is used by pdf2image
        result = subprocess.run(['pdftoppm', '-v'], 
                              capture_output=True, text=True, timeout=5)
        # pdftoppm outputs version info (may be in stdout or stderr)
        output = (result.stdout + result.stderr).lower()
        if 'pdftoppm' in output or 'poppler' in output or result.returncode == 0:
            version_line = (result.stderr + result.stdout).split('\n')[0].strip()
            if version_line:
                print_result("Poppler (PDF conversion)", True, version_line)
            else:
                print_result("Poppler (PDF conversion)", True)
            return True
        else:
            print_result("Poppler (PDF conversion)", False, "Poppler not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_result("Poppler (PDF conversion)", False, 
                    "Install with: brew install poppler (macOS)")
        return False

def check_directories():
    """Check if required directories exist."""
    dirs = ['input_pdfs', 'output_csv', 'temp_images']
    all_exist = True
    
    for directory in dirs:
        exists = Path(directory).exists()
        print_result(f"Directory: {directory}/", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks."""
    print_header("OCR POC Setup Verification")
    
    print("\n📋 Checking System Requirements...")
    
    results = []
    
    # Python version
    results.append(check_python_version())
    
    # System tools
    print("\n📦 Checking System Tools...")
    results.append(check_tesseract())
    results.append(check_tesseract_languages())
    results.append(check_poppler())
    
    # Python packages
    print("\n🐍 Checking Python Packages...")
    packages = [
        ('pdf2image', 'pdf2image'),
        ('pytesseract', 'pytesseract'),
        ('Pillow', 'PIL'),
        ('pandas', 'pandas'),
    ]
    
    for package_name, import_name in packages:
        results.append(check_python_package(package_name, import_name))
    
    # Directories
    print("\n📁 Checking Project Directories...")
    results.append(check_directories())
    
    # Summary
    print_header("SUMMARY")
    passed = sum(results)
    total = len(results)
    
    print(f"\nChecks Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the extraction script.")
        print("\nNext steps:")
        print("1. Place your PDF files in the 'input_pdfs/' folder")
        print("2. Run: python extract_voter_data.py")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies.")
        print("\nQuick installation guide:")
        print("\n  macOS:")
        print("    brew install tesseract tesseract-lang poppler")
        print("    pip install -r requirements.txt")
        print("\n  Linux (Ubuntu/Debian):")
        print("    sudo apt-get install tesseract-ocr tesseract-ocr-hin poppler-utils")
        print("    pip install -r requirements.txt")
    
    print("\n" + "="*60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

