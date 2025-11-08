#!/usr/bin/env python3
"""
Configuration Module
====================
Centralized configuration constants for the PDF processing system.
"""

from multiprocessing import cpu_count

# Directory paths
INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "output_images"
TEMP_DIR = "temp_images"

# Image processing settings
IMAGE_DPI = 400  # Higher DPI for better image quality

# Processing settings
USE_MULTIPROCESSING = False  # Set to True for parallel processing
MAX_WORKERS = cpu_count() - 1 or 1

# Logging configuration
LOG_FILE = 'pdf_processing.log'
LOG_LEVEL = 'INFO'

