#!/usr/bin/env python3
"""
Configuration Module
====================
Centralized configuration constants for the voter data extraction system.
"""

from multiprocessing import cpu_count

# Directory paths
INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "output_csv"
TEMP_DIR = "temp_images"

# OCR settings
OCR_LANGUAGE = "hin+eng"  # Hindi + English
IMAGE_DPI = 400  # Higher DPI for better OCR accuracy
TESSERACT_CONFIG = '--psm 6'  # Assume uniform block of text

# Processing settings
SKIP_FIRST_N_PAGES = 2  # Skip metadata and booth pages
SKIP_LAST_N_PAGES = 1   # Skip summary page
USE_MULTIPROCESSING = False  # Set to True for parallel processing
MAX_WORKERS = cpu_count() - 1 or 1

# Logging configuration
LOG_FILE = 'voter_extraction.log'
LOG_LEVEL = 'INFO'

