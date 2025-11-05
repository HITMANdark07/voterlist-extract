# PaddleOCR Setup Instructions

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/

### 2. Setup Python Environment

```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PaddleOCR dependencies manually
pip install paddleocr paddlepaddle opencv-python

# OPTIONAL: Install PPStructureV3 dependencies for layout detection
# This enables automatic table/cell detection but requires additional dependencies
pip install "paddlex[ocr]"

# After installation, freeze requirements
pip freeze > requirements.txt
```

### 3. Verify Installation

```bash
python verify_setup.py
```

## Usage

### Extract Voter Data

```bash
# Place PDFs in input_pdfs/
cp /path/to/pdfs/*.pdf input_pdfs/

# Run PaddleOCR extraction
python extract_voter_data_paddleocr.py

# Check results
ls ocr_output/
```

## Output

- **CSV File**: `ocr_output/voter_list_extracted_<pdf_name>.csv`
  - Contains structured voter data with columns:
    - `serial_no` (क्रम संख्या)
    - `epic_no` (मतदाता संख्या)
    - `name` (निर्वाचक का नाम)
    - `relation_type` (पति/पिता/माता)
    - `relation_name` (पति/पिता का नाम)
    - `house_no` (मकान संख्या)
    - `age` (आयु)
    - `gender` (लिंग)

- **Annotated Image**: `ocr_output/detected_blocks_<pdf_name>.png`
  - Visual representation of detected voter boxes

## Features

- **Automatic Layout Detection**: Uses PaddleOCR PP-Structure to automatically detect voter boxes/table cells
- **High Accuracy Hindi OCR**: Recognizes Devanagari script with high accuracy
- **No Manual Segmentation**: Fully automatic, no need to manually define regions
- **Preprocessing**: Uses OpenCV for image enhancement (grayscale, thresholding)
- **Fallback Support**: Falls back to regular OCR if structure detection fails

## Configuration

The script uses settings from `config.py`:
- `SKIP_FIRST_N_PAGES`: Skip metadata pages (default: 2)
- `SKIP_LAST_N_PAGES`: Skip summary page (default: 1)

## Troubleshooting

- **PaddleOCR not found**: Install using `pip install paddleocr paddlepaddle`
- **Model download**: PaddleOCR will automatically download models on first run
- **Memory issues**: Reduce image DPI in `config.py` or process fewer pages at once
- **No blocks detected**: Check PDF quality, may need to adjust preprocessing parameters
- **Check logs**: Review `voter_extraction.log` for detailed error messages

## Differences from Tesseract Pipeline

- **Automatic Detection**: No need for manual grid detection
- **Better Layout Understanding**: PP-Structure understands document structure
- **Higher Accuracy**: Better Hindi text recognition
- **Faster Processing**: Optimized for batch processing

## Customization

To modify parsing logic, edit the `parse_voter_fields()` function in `extract_voter_data_paddleocr.py`. The function uses regex patterns to match Hindi keywords:
- `क्रम संख्या` → serial_no
- `निर्वाचक का नाम` → name
- `पति/पिता का नाम` → relation_name
- `आयु` → age
- `लिंग` → gender
- `मतदाता संख्या` → epic_no
