# Voter Page Image Processing (No OCR)

Process scanned Indian election PDF files into images only (no OCR). The pipeline:
- Crops page 1 by detected page contour
- Detects voter grids on pages 3 to n-1
- Saves each detected voter block as an image

## Features

- **Page 1 Cropping**: Detects main page contour and saves cropped page
- **Grid Detection**: Automatically detects voter grids in relevant pages
- **Image Outputs Only**: No OCR, no text parsing, no CSV export

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

### 3. Run Processing

```bash
# Place PDFs in input_pdfs/
cp /path/to/pdfs/*.pdf input_pdfs/

# Run
python extract_voters.py

# Check results
open output_images/
```

## Project Structure

```
voterlist-extract/
├── extract_voters.py       # Main processing script
├── config.py               # Configuration settings
├── pdf_converter.py        # PDF → Image conversion helpers
├── grid_detector.py        # Grid detection (voter blocks)
├── requirements.txt        # Python dependencies
├── input_pdfs/             # Place PDFs here
├── output_images/          # Image results
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

1. **Page 1**: Detect page contour and save cropped image as:
   - `output_images/{pdf_name}/page_1.jpg`
2. **Pages 3 to n-1**: Detect voter grids and save each grid as:
   - `output_images/{pdf_name}/voters/voter_001.jpg`, `voter_002.jpg`, ...

Notes:
- There are no per-page folders for voters; images are numbered sequentially.
- This repository currently does not perform OCR or CSV export.

## Output Structure

```
output_images/
  {pdf_name}/
    page_1.jpg
    voters/
      voter_001.jpg
      voter_002.jpg
      voter_003.jpg
      ...
```

## Troubleshooting

- **PDF conversion fails**: Ensure Poppler is installed (`pdftoppm` available). The `pdf2image` library requires poppler to convert PDFs to images.
- **No grids detected**: Adjust thresholds in `grid_detector.py`
- **OpenCV issues**: Ensure `opencv-python-headless` is installed, or use `opencv-python` if you need GUI windows
- **PaddleOCR/PaddlePaddle issues**: Ensure CUDA is properly installed for GPU support
- **Check logs**: See `pdf_processing.log`

## License

Open source - provided as-is for image processing purposes.
