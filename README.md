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
pip install -r requirements.txt
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
├── ocr_processor.py        # Image → Text (OCR)
├── text_parser.py          # Text → Structured data
├── data_saver.py          # Data → CSV/Excel
├── extract_voter_data.py  # Main orchestrator
├── verify_setup.py        # Setup verification
├── test_single_image.py   # Test OCR on image
├── convert_pdf_page.py    # Convert PDF page
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

## Troubleshooting

- **Tesseract not found**: Install Tesseract with Hindi language pack
- **No text extracted**: Increase `IMAGE_DPI` in `config.py` or check PDF quality
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Check logs**: Review `voter_extraction.log` for details

## Module Usage

Use modules independently:

```python
from pdf_converter import pdf_to_images
from ocr_processor import perform_ocr
from text_parser import extract_voter_blocks
from data_saver import save_voters_to_csv

images = pdf_to_images("input_pdfs/booth_001.pdf")
for page_num, image in images:
    text = perform_ocr(image)
    voters = extract_voter_blocks(text)
    save_voters_to_csv(voters, "booth_001")
```

## Performance

- **Speed**: ~40-60 seconds per 15-page PDF
- **Accuracy**: 85-95% (depends on PDF quality)
- **Multiprocessing**: Enable in `config.py` for multiple PDFs

## License

Open source - provided as-is for data extraction purposes.
