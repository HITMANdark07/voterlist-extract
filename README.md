# Indian Election Voter Data Extraction

Extract structured voter information from scanned Indian election PDF files using OCR.

## Features

- **Grid Detection**: Automatically detects voter grids in PDF pages
- **Box Detection**: Detects and segments boxes within grids (serial numbers, EPIC, photo)
- **Smart Segmentation**: Splits grids into 60% (details) and 40% (EPIC) sections
- **OCR Processing**: Uses Tesseract OCR with Hindi + English support
- **Image Saving**: Saves all processed images for verification
- **CSV Export**: Exports structured voter data to CSV

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
python main.py

# Check results
ls output_csv/
```

## Project Structure

```
ocr-poc/
├── main.py                      # Main extraction script
├── config.py                    # Configuration settings
├── pdf_converter.py             # PDF → Image conversion
├── grid_detector.py             # Grid detection & extraction
├── box_detector.py              # Box detection & segmentation
├── ocr_processor.py            # Low-level OCR functions (Tesseract wrapper)
├── tesseract_ocr.py            # Tesseract OCR extraction from grid segments
├── paddleocr.py                # PaddleOCR extraction from grid segments
├── ocr_factory.py              # OCR engine factory (switches between engines)
├── text_parser.py              # Text → Structured data
├── data_saver.py               # Data → CSV/Excel
├── test_box_detection.py       # Test box detection pipeline
├── verify_setup.py             # Setup verification
├── requirements.txt            # Python dependencies
├── input_pdfs/                 # Place PDFs here
├── output_csv/                 # Results appear here
│   └── images/                 # Processed images saved here
└── temp_images/                # Temporary files
```

## Configuration

Edit `config.py` to adjust settings:

```python
IMAGE_DPI = 400              # Image quality (300-600)
OCR_ENGINE = "tesseract"     # OCR engine: "tesseract" or "paddleocr"
OCR_LANGUAGE = "hin+eng"     # OCR languages (for Tesseract)
SKIP_FIRST_N_PAGES = 2       # Skip metadata pages
SKIP_LAST_N_PAGES = 1        # Skip summary page
TEST_MODE = True             # Process only first page (for testing)
```

## Pipeline Flow

1. **PDF Conversion**: Converts PDF pages to images
2. **Grid Detection**: Detects individual voter grids using contour detection
3. **Box Detection**: Detects boxes inside each grid (serial numbers, photo)
4. **Segmentation**: 
   - Colors detected boxes white
   - Splits grid into 60% left (details) and 40% right (EPIC)
5. **OCR Processing**: 
   - Extracts serial numbers from serial boxes
   - Extracts EPIC from right half
   - Extracts details from left half
6. **Data Parsing**: Parses OCR text into structured voter data
7. **CSV Export**: Saves to CSV with all voter information

## Output Format

CSV columns: `serial_no`, `epic_no`, `name`, `relation_type`, `relation_name`, `house_no`, `age`, `gender`

## Output Structure

```
output_csv/
├── voters_{pdf_name}.csv
└── images/
    └── {pdf_name}/
        └── page_{page_num}/
            ├── grid_{idx}_before_ocr/    # Images before OCR
            │   ├── 00_original_grid.jpg
            │   ├── 01_grid_with_white_boxes.jpg
            │   ├── 02_left_half_60_percent.jpg
            │   ├── 03_right_half_40_percent.jpg
            │   ├── 04_boxes_visualization.jpg
            │   └── detected_boxes/        # Individual box images
            └── voter_{idx}/               # Images after OCR
                ├── serial_box_01.jpg
                ├── serial_box_02.jpg
                ├── left_half_details.jpg
                ├── right_half_epic.jpg
                ├── photo_box.jpg
                └── metadata.txt
```

## Testing

Test the box detection pipeline:

```bash
python test_box_detection.py
```

This will save segmented images to `temp_images/box_detection/` for inspection.

## OCR Engine Selection

The system supports two OCR engines. Switch between them in `config.py`:

```python
OCR_ENGINE = "tesseract"  # or "paddleocr"
```

### Tesseract OCR (Default)
- **Pros**: Fast, lightweight, good for English + Hindi
- **Cons**: May struggle with complex layouts
- **Setup**: `brew install tesseract tesseract-lang` (macOS) or `sudo apt-get install tesseract-ocr tesseract-ocr-hin` (Linux)

### PaddleOCR
- **Pros**: Better accuracy for complex layouts, better Hindi support
- **Cons**: Slower, requires more memory
- **Setup**: `pip install paddleocr paddlepaddle`

The system will automatically use the selected engine. Both engines extract text from the same grid segments.

## Troubleshooting

- **Tesseract not found**: Install Tesseract with Hindi language pack
- **No text extracted**: Increase `IMAGE_DPI` in `config.py` or check PDF quality
- **Missing dependencies**: Install all packages: `pip install -r requirements.txt`
- **Grid detection fails**: Check PDF quality, may need to adjust thresholds in `grid_detector.py`
- **Box detection issues**: Check `output_csv/images/` to see how boxes are detected
- **Check logs**: Review `voter_extraction.log` for details

## Module Usage

Use modules independently:

```python
from pdf_converter import pdf_to_images
from grid_detector import detect_voter_blocks
from box_detector import process_grid
from ocr_factory import extract_text_from_grid_segments
from text_parser import parse_from_regions
from data_saver import save_voters_to_csv

images = pdf_to_images("input_pdfs/booth_001.pdf")
all_voters = []
for page_num, image in images:
    blocks = detect_voter_blocks(image)
    for block in blocks:
        grid_data = process_grid(block)
        # Extract text using OCR (uses OCR_ENGINE from config)
        ocr_results = extract_text_from_grid_segments(grid_data, page_num, 0)
        # Parse voter data
        voter_data = parse_from_regions(
            ocr_results['serial_text'],
            ocr_results['epic_text'],
            ocr_results['details_text']
        )
        if voter_data:
            voters.append(voter_data)
save_voters_to_csv(all_voters, "booth_001")
```

## Performance

- **Speed**: ~40-60 seconds per 15-page PDF
- **Accuracy**: 85-95% (depends on PDF quality)
- **Image Saving**: All processed images are saved for verification

## License

Open source - provided as-is for data extraction purposes.
