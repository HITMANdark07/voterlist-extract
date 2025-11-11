# Voter List Extraction and OCR Processing

Process scanned Indian election PDF files with image extraction and OCR. The pipeline:
- Crops page 1 by detected page contour
- Detects voter grids on pages 3 to n-1
- Saves each detected voter block as an image
- Cleans voter images by removing borders and boxes
- Runs OCR on cleaned images using Surya OCR

## Features

- **Page 1 Cropping**: Detects main page contour and saves cropped page
- **Grid Detection**: Automatically detects voter grids in relevant pages
- **Image Extraction**: Saves extracted images in organized directory structure
- **Image Cleaning**: Removes outer borders and inner boxes from voter images
- **OCR Processing**: Runs Surya OCR on cleaned images with multiprocessing support
- **Structured Output**: Saves OCR results as text files

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

# Install dependencies
pip install -r requirements.txt

# Install Surya OCR (if not already in requirements.txt)
pip install surya-ocr
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

### 4. Clean Voter Images

```bash
# Remove borders and boxes from extracted voter images
python clean_grid.py

# Cleaned images will be saved to "voter split" directory
```

### 5. Run OCR Processing

```bash
# Process all cleaned images with OCR
python ocr_split_images.py

# OCR results will be saved as .txt files in ocr_results directory
```

## Project Structure

```
voterlist-extract/
├── extract_voters.py       # Image extraction from PDFs
├── clean_grid.py           # Clean voter images (remove borders/boxes)
├── ocr_split_images.py     # OCR processing on cleaned images
├── config.py               # Configuration settings
├── pdf_converter.py        # PDF → Image conversion helpers
├── grid_detector.py        # Grid detection (voter blocks)
├── requirements.txt        # Python dependencies
├── input_pdfs/             # Place PDFs here
├── output_images/          # Extracted images
├── voter split/            # Cleaned voter images
└── ocr_results/            # OCR text results
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

### Step 2: Image Cleaning (`clean_grid.py`)

1. Processes all voter images from `output_images/{pdf_name}/voters/`
2. Removes outer border rectangles
3. Removes inner boxes and form elements
4. Saves cleaned images to `voter split/{pdf_name}/`

### Step 3: OCR Processing (`ocr_split_images.py`)

1. Scans all cleaned images in `voter split/` directory
2. Processes each image with Surya OCR (multiprocessing supported)
3. Saves OCR results as text files in `ocr_results/{pdf_name}/`

## Output Structure

```
output_images/
  {pdf_name}/
    page_1.jpg
    voters/
      voter_001.jpg
      voter_002.jpg
      ...

voter split/
  {pdf_name}/
    voter_001.jpg        # Cleaned image
    voter_002.jpg
    ...

ocr_results/
  {pdf_name}/
    voter_001.txt        # OCR text result
    voter_002.txt
    ...
```

## Troubleshooting

- **PDF conversion fails**: Ensure Poppler is installed (`pdftoppm` available). The `pdf2image` library requires poppler to convert PDFs to images.
- **No grids detected**: Adjust thresholds in `grid_detector.py`
- **OpenCV issues**: Ensure `opencv-python-headless` is installed, or use `opencv-python` if you need GUI windows
- **OCR processing fails**: Make sure Surya OCR is properly installed. For GPU support, ensure CUDA is properly configured.
- **Memory issues during OCR**: Use `num_workers=1` for sequential processing to reduce memory usage.
- **Check logs**: See `pdf_processing.log` for extraction, `ocr_split_processing.log` for OCR processing, and `contour_boxes_extraction.log` for image cleaning

## License

Open source - provided as-is for image processing purposes.
