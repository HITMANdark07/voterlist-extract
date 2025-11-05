# Indian Election Voter Data Extraction

Extract structured voter information from scanned Indian election PDF files using OCR.

## Quick Start

### 1. Install System Dependencies

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng poppler-utils
```

### 2. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pdf2image pytesseract Pillow pandas python-dateutil opencv-python numpy

# Optional: For improved text detection (EasyOCR)
pip install easyocr
```

### 3. Verify Installation

```bash
python verify_setup.py
```

### 4. Extract Data

```bash
# Place PDFs in input_pdfs/
cp /path/to/pdfs/*.pdf input_pdfs/

# Run extraction
python extract_voter_data.py

# Check results
ls output_csv/
```

## Project Structure

```
ocr-poc/
├── config.py              # Configuration settings
├── pdf_converter.py        # PDF → Image conversion
├── grid_detector.py        # Grid detection & block extraction
├── block_splitter.py      # Split blocks into regions (EasyOCR + fallback)
├── ocr_detector.py        # EasyOCR text detection (optional)
├── ocr_processor.py       # Image → Text (OCR with Tesseract)
├── text_parser.py          # Text → Structured data
├── data_saver.py          # Data → CSV/Excel
├── extract_voter_data.py  # Main orchestrator
├── verify_setup.py        # Setup verification
├── test_easyocr_detection.py  # Test EasyOCR detection
├── input_pdfs/            # Place PDFs here
├── output_csv/            # Results appear here
└── temp_images/           # Temporary files
```

## Configuration

Edit `config.py` to adjust settings:

```python
IMAGE_DPI = 400              # Image quality (300-600)
OCR_LANGUAGE = "hin+eng"      # OCR languages
SKIP_FIRST_N_PAGES = 2       # Skip metadata pages
SKIP_LAST_N_PAGES = 1        # Skip summary page
USE_MULTIPROCESSING = False  # Enable for batch processing
```

## Output Format

CSV columns: `serial_no`, `epic_no`, `name`, `relation_type`, `relation_name`, `house_no`, `age`, `gender`

## EasyOCR Detection (Optional)

The system uses a hybrid approach for text region detection:
- **Primary**: EasyOCR detection (if installed) for better accuracy
- **Fallback**: Hardcoded percentage-based splitting

EasyOCR provides better detection of text regions, especially for serial numbers and EPIC numbers. If EasyOCR is not installed, the system automatically falls back to hardcoded region splitting.

To test EasyOCR detection:
```bash
python test_easyocr_detection.py
```

## Troubleshooting

- **Tesseract not found**: Install Tesseract with Hindi language pack
- **No text extracted**: Increase `IMAGE_DPI` in `config.py` or check PDF quality
- **Missing dependencies**: Install all packages: `pip install pdf2image pytesseract Pillow pandas python-dateutil opencv-python numpy`
- **EasyOCR not available**: System will automatically use hardcoded fallback. Install with `pip install easyocr` for improved detection
- **Grid detection fails**: Check PDF quality, may need to adjust thresholds in `grid_detector.py`
- **Check logs**: Review `voter_extraction.log` for details

## Module Usage

Use modules independently:

```python
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from ocr_processor import perform_ocr
from text_parser import parse_single_block
from data_saver import save_voters_to_csv

images = pdf_to_images("input_pdfs/booth_001.pdf")
all_voters = []
for page_num, image in images:
    # Detect grid and extract blocks
    blocks = detect_voter_blocks(image)
    # OCR and parse each block
    for block in blocks:
        text = perform_ocr(block)
        voter = parse_single_block(text)
        if voter:
            all_voters.append(voter)
save_voters_to_csv(all_voters, "booth_001")
```

## Performance

- **Speed**: ~40-60 seconds per 15-page PDF
- **Accuracy**: 85-95% (depends on PDF quality)
- **Multiprocessing**: Enable in `config.py` for multiple PDFs

## License

Open source - provided as-is for data extraction purposes.
