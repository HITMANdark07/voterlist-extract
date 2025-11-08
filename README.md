# Voter List Extraction and OCR Processing

Process scanned Indian election PDF files with image extraction and OCR. The pipeline:
- Crops page 1 by detected page contour
- Detects voter grids on pages 3 to n-1
- Saves each detected voter block as an image
- Runs OCR on all extracted images using PaddleOCRVL

## Features

- **Page 1 Cropping**: Detects main page contour and saves cropped page
- **Grid Detection**: Automatically detects voter grids in relevant pages
- **Image Extraction**: Saves extracted images in organized directory structure
- **OCR Processing**: Runs PaddleOCRVL on all extracted images
- **Structured Output**: Saves OCR results as JSON and Markdown files

## Quick Start

### 1. Install System Dependencies

Poppler is required for PDF → image conversion (used by `pdf2image` library).

**Windows:**
- Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
- Extract and add the `bin` folder to your system PATH
- Or install via conda: `conda install -c conda-forge poppler`

**macOS:**
```bash
brew install poppler
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install poppler-utils
```

### 2. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PaddlePaddle GPU (required for PaddleOCR)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# Install PaddleOCR with doc-parser support
python -m pip install -U "paddleocr[doc-parser]"

# Install safetensors (Windows-specific wheel)
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl

# Install other dependencies
pip install -r requirements.txt
```

### 3. Extract Images from PDFs

```bash
# Place PDFs in input_pdfs/
cp /path/to/pdfs/*.pdf input_pdfs/

# Run image extraction
python extract_voters.py

# Check extracted images
open output_images/
```

### 4. Run OCR Processing

```bash
# Process all extracted images with OCR
python ocr_processor.py

# OCR results will be saved alongside images
# Each image will have corresponding .json and .md files
```

## Project Structure

```
voterlist-extract/
├── extract_voters.py       # Image extraction from PDFs
├── ocr_processor.py       # OCR processing on extracted images
├── config.py               # Configuration settings
├── pdf_converter.py        # PDF → Image conversion helpers
├── grid_detector.py        # Grid detection (voter blocks)
├── requirements.txt        # Python dependencies
├── input_pdfs/             # Place PDFs here
├── output_images/          # Image and OCR results
└── temp_images/            # Temporary files (debug)
```

## Configuration

Edit `config.py` to adjust settings:

```python
IMAGE_DPI = 400          # Image quality (300-600)
INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "output_images"
```

## Processing Flow

### Step 1: Image Extraction (`extract_voters.py`)

1. **Page 1**: Detect page contour and save cropped image as:
   - `output_images/{pdf_name}/page_1.jpg`
2. **Pages 3 to n-1**: Detect voter grids and save each grid as:
   - `output_images/{pdf_name}/voters/voter_001.jpg`, `voter_002.jpg`, ...

### Step 2: OCR Processing (`ocr_processor.py`)

1. Scans all images in `output_images/` directory
2. Processes each image with PaddleOCRVL
3. Saves OCR results (JSON and Markdown) alongside each image

## Output Structure

```
output_images/
  {pdf_name}/
    page_1.jpg
    page_1.json          # OCR result (JSON)
    page_1.md            # OCR result (Markdown)
    voters/
      voter_001.jpg
      voter_001.json     # OCR result (JSON)
      voter_001.md       # OCR result (Markdown)
      voter_002.jpg
      voter_002.json
      voter_002.md
      ...
```

## Troubleshooting

- **PDF conversion fails**: Ensure Poppler is installed (`pdftoppm` available). The `pdf2image` library requires poppler to convert PDFs to images.
- **No grids detected**: Adjust thresholds in `grid_detector.py`
- **OpenCV issues**: Ensure `opencv-python-headless` is installed, or use `opencv-python` if you need GUI windows
- **PaddleOCR/PaddlePaddle issues**: Ensure CUDA is properly installed for GPU support
- **OCR processing fails**: Make sure PaddleOCRVL is properly installed and GPU drivers are configured
- **Check logs**: See `pdf_processing.log` for extraction and `ocr_processing.log` for OCR processing

## License

Open source - provided as-is for image processing purposes.
