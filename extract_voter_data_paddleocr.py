#!/usr/bin/env python3
"""
PaddleOCR-based Voter Data Extraction
======================================
Replaces Tesseract + OpenCV + regex pipeline with PaddleOCR PP-Structure
for automatic layout detection and Hindi text recognition.
"""

import logging
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import pandas as pd
from paddleocr import PaddleOCR

# Import existing PDF converter (keeping pdf2image as requested)
from pdf_converter import pdf_to_images
from config import SKIP_FIRST_N_PAGES, SKIP_LAST_N_PAGES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voter_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PPStructureV3 requires additional dependencies - make it optional
try:
    from paddleocr import PPStructureV3
    HAS_PPSTRUCTUREV3 = True
except ImportError:
    HAS_PPSTRUCTUREV3 = False
    logger.warning("PPStructureV3 not available. Install with: pip install 'paddlex[ocr]'")

# Output directory
OCR_OUTPUT_DIR = Path("ocr_output")
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    Converts to grayscale, applies threshold, and cleans up box lines.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array (OpenCV format)
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance text
    # This helps with varying lighting conditions
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Optional: Morphological operations to clean up noise
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def detect_voter_blocks_with_paddleocr(image: Image.Image) -> List[Dict]:
    """
    Use PaddleOCR PPStructureV3 to detect voter boxes/table cells automatically.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of detected blocks with text and bounding boxes
    """
    # Convert PIL to numpy array for processing
    img_array = np.array(image)
    
    # Try PP-StructureV3 first if available (requires paddlex[ocr])
    # Otherwise fall back to regular OCR immediately
    if not HAS_PPSTRUCTUREV3:
        logger.info("PPStructureV3 not available, using regular OCR")
        return detect_blocks_with_regular_ocr(image)
    
    # Initialize PP-StructureV3 (layout detection + OCR)
    try:
        # Try with just device parameter
        structure_engine = PPStructureV3(device='cpu')
        
        # Run structure analysis
        # PPStructureV3 can accept PIL Image directly or numpy array
        # Try PIL Image first, fallback to numpy array
        try:
            result = structure_engine(image)
        except (TypeError, AttributeError) as e:
            # If PIL Image doesn't work, convert to numpy array
            logger.debug(f"Trying numpy array format: {e}")
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            result = structure_engine(img_bgr)
        
        # Extract blocks from result
        # PPStructureV3 returns a dict with structure:
        # - 'layout_res': layout detection results
        # - 'ocr_res': OCR results for each detected region
        blocks = []
        
        # Handle different result formats
        if isinstance(result, dict):
            # Extract layout regions and OCR results
            layout_res = result.get('layout_res', [])
            ocr_res = result.get('ocr_res', {})
            
            # Process each layout region
            for region in layout_res:
                if not isinstance(region, dict):
                    continue
                
                # Get bounding box
                bbox = region.get('bbox', [])
                region_type = region.get('type', 'unknown')
                
                # Get OCR result for this region (if available)
                text_lines = []
                region_id = region.get('id', None)
                
                if region_id and region_id in ocr_res:
                    ocr_result = ocr_res[region_id]
                    if isinstance(ocr_result, list):
                        for line in ocr_result:
                            if isinstance(line, list) and len(line) >= 2:
                                text_info = line[1]
                                if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                    text_lines.append(str(text_info[0]))
                                elif isinstance(text_info, str):
                                    text_lines.append(text_info)
                            elif isinstance(line, dict):
                                text_lines.append(line.get('text', ''))
                
                combined_text = ' '.join(text_lines).strip()
                
                if combined_text or bbox:
                    blocks.append({
                        'bbox': bbox,
                        'text': combined_text,
                        'text_lines': text_lines,
                        'type': region_type
                    })
        
        elif isinstance(result, list):
            # Fallback: handle as list format (similar to old PPStructure)
            for item in result:
                if not isinstance(item, dict):
                    continue
                
                bbox = item.get('bbox', [])
                ocr_result = item.get('res', [])
                text_lines = []
                
                if ocr_result:
                    for line in ocr_result:
                        if isinstance(line, list) and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) > 0:
                                text_lines.append(str(text_info[0]))
                            elif isinstance(text_info, str):
                                text_lines.append(text_info)
                        elif isinstance(line, dict):
                            text_lines.append(line.get('text', ''))
                
                combined_text = ' '.join(text_lines).strip()
                
                if combined_text or bbox:
                    blocks.append({
                        'bbox': bbox,
                        'text': combined_text,
                        'text_lines': text_lines,
                        'type': item.get('type', 'unknown')
                    })
        
        logger.info(f"Detected {len(blocks)} text blocks with PP-StructureV3")
        return blocks
        
    except Exception as e:
        logger.error(f"Error in PP-Structure detection: {e}", exc_info=True)
        # Fallback to regular OCR if structure detection fails
        logger.info("Falling back to regular OCR mode")
        return detect_blocks_with_regular_ocr(image)


def detect_blocks_with_regular_ocr(image: Image.Image) -> List[Dict]:
    """
    Fallback: Use regular PaddleOCR if structure detection fails.
    
    Args:
        image: PIL Image object
        
    Returns:
        List of detected blocks with text and bounding boxes
    """
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='hi')
        
        img_array = np.array(image)
        result = ocr_engine.ocr(img_array, cls=True)
        
        blocks = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) > 1:
                    bbox = line[0]  # Bounding box coordinates
                    text_info = line[1]
                    text = text_info[0] if isinstance(text_info, tuple) else text_info
                    
                    if text.strip():
                        blocks.append({
                            'bbox': bbox,
                            'text': text,
                            'text_lines': [text]
                        })
        
        logger.info(f"Detected {len(blocks)} text blocks with regular OCR")
        return blocks
        
    except Exception as e:
        logger.error(f"Error in regular OCR: {e}")
        return []


def parse_voter_fields(text: str) -> Optional[Dict[str, any]]:
    """
    Parse structured voter information from OCR text using Hindi keywords.
    
    Fields to extract:
    - क्रम संख्या (Serial Number)
    - निर्वाचक का नाम (Voter Name)
    - पति/पिता का नाम (Husband/Father Name)
    - आयु (Age)
    - लिंग (Gender)
    - मतदाता संख्या (Voter ID/EPIC Number)
    
    Args:
        text: OCR text from a voter block
        
    Returns:
        Dictionary with parsed voter information or None
    """
    if not text or not text.strip():
        return None
    
    voter = {
        'serial_no': None,      # क्रम संख्या
        'epic_no': '',          # मतदाता संख्या
        'name': '',             # निर्वाचक का नाम
        'relation_type': '',    # पति/पिता
        'relation_name': '',    # पति/पिता का नाम
        'house_no': '',         # मकान संख्या (optional)
        'age': None,            # आयु
        'gender': ''            # लिंग
    }
    
    # Clean text - normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Extract EPIC/Voter ID number (मतदाता संख्या)
    # Pattern: UIM followed by digits, or other ID patterns like ABC1234567
    epic_patterns = [
        r'\b(UIM\d{7,10})\b',           # UIMxxxxxxxx format
        r'\b([A-Z]{2,3}\d{7,10})\b',    # General format
        r'मतदाता\s*संख्या\s*[:：]?\s*([A-Z0-9]{8,12})',  # With Hindi label
    ]
    
    for pattern in epic_patterns:
        epic_match = re.search(pattern, text, re.IGNORECASE)
        if epic_match:
            voter['epic_no'] = epic_match.group(1).upper()
            break
    
    # Must have EPIC number to proceed
    if not voter['epic_no']:
        return None
    
    # Extract serial number (क्रम संख्या)
    serial_patterns = [
        r'क्रम\s*संख्या\s*[:：]?\s*(\d+)',
        r'क्र\.\s*सं\.\s*[:：]?\s*(\d+)',
        r'^(\d{1,4})\s',  # Leading digits (1-4 digits)
    ]
    
    for pattern in serial_patterns:
        serial_match = re.search(pattern, text)
        if serial_match:
            try:
                voter['serial_no'] = int(serial_match.group(1))
                break
            except ValueError:
                continue
    
    # Extract voter name (निर्वाचक का नाम)
    name_patterns = [
        r'निर्वाचक\s*का\s*नाम\s*[:：]?\s*([^\n:पआलम]+?)(?:\s*(?:पति|पिता|माता|आयु|लिंग|मकान)|$)',
        r'नाम\s*[:：]?\s*([^\n:पआलम]+?)(?:\s*(?:पति|पिता|माता|आयु|लिंग|मकान)|$)',
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, text)
        if name_match:
            name = name_match.group(1).strip()
            # Clean up name - remove common OCR artifacts
            name = re.sub(r'[^\u0900-\u097F\s\w]', '', name)
            if name and len(name) > 2:  # Valid name should be at least 3 chars
                voter['name'] = name
                break
    
    # Extract relation type and name (पति/पिता का नाम)
    # Check for पति (husband)
    if re.search(r'पति\s*का\s*नाम', text):
        voter['relation_type'] = 'पति'
        relation_patterns = [
            r'पति\s*का\s*नाम\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
            r'पति\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
        ]
        for pattern in relation_patterns:
            relation_match = re.search(pattern, text)
            if relation_match:
                relation_name = relation_match.group(1).strip()
                relation_name = re.sub(r'[^\u0900-\u097F\s\w]', '', relation_name)
                if relation_name and len(relation_name) > 2:
                    voter['relation_name'] = relation_name
                    break
    
    # Check for पिता (father)
    elif re.search(r'पिता\s*का\s*नाम', text):
        voter['relation_type'] = 'पिता'
        relation_patterns = [
            r'पिता\s*का\s*नाम\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
            r'पिता\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
        ]
        for pattern in relation_patterns:
            relation_match = re.search(pattern, text)
            if relation_match:
                relation_name = relation_match.group(1).strip()
                relation_name = re.sub(r'[^\u0900-\u097F\s\w]', '', relation_name)
                if relation_name and len(relation_name) > 2:
                    voter['relation_name'] = relation_name
                    break
    
    # Check for माता (mother)
    elif re.search(r'माता\s*का\s*नाम', text):
        voter['relation_type'] = 'माता'
        relation_patterns = [
            r'माता\s*का\s*नाम\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
            r'माता\s*[:：]?\s*([^\n:आलम]+?)(?:\s*(?:आयु|लिंग|मकान)|$)',
        ]
        for pattern in relation_patterns:
            relation_match = re.search(pattern, text)
            if relation_match:
                relation_name = relation_match.group(1).strip()
                relation_name = re.sub(r'[^\u0900-\u097F\s\w]', '', relation_name)
                if relation_name and len(relation_name) > 2:
                    voter['relation_name'] = relation_name
                    break
    
    # Extract age (आयु)
    age_patterns = [
        r'आयु\s*[:：]?\s*(\d+)',
        r'उम्र\s*[:：]?\s*(\d+)',
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, text)
        if age_match:
            try:
                age = int(age_match.group(1))
                if 18 <= age <= 120:  # Reasonable age range
                    voter['age'] = age
                    break
            except ValueError:
                continue
    
    # Extract gender (लिंग)
    if re.search(r'लिंग\s*[:：]?\s*(?:महिला|स्त्री|female)', text, re.IGNORECASE):
        voter['gender'] = 'महिला'
    elif re.search(r'लिंग\s*[:：]?\s*(?:पुरुष|male)', text, re.IGNORECASE):
        voter['gender'] = 'पुरुष'
    elif re.search(r'\b(?:महिला|स्त्री|female)\b', text, re.IGNORECASE):
        voter['gender'] = 'महिला'
    elif re.search(r'\b(?:पुरुष|male)\b', text, re.IGNORECASE):
        voter['gender'] = 'पुरुष'
    
    # Extract house number (मकान संख्या) - optional field
    house_patterns = [
        r'मकान\s*संख्या\s*[:：]?\s*(\d+)',
        r'संख्या\s*[:：]?\s*(\d+)',
    ]
    
    for pattern in house_patterns:
        house_match = re.search(pattern, text)
        if house_match:
            voter['house_no'] = house_match.group(1)
            break
    
    # Only return if we have essential fields (EPIC and name)
    if voter['epic_no'] and voter['name']:
        return voter
    
    return None


def draw_detected_blocks(image: Image.Image, blocks: List[Dict]) -> Image.Image:
    """
    Draw bounding boxes on image to visualize detected blocks.
    
    Args:
        image: PIL Image object
        blocks: List of detected blocks with bbox coordinates
        
    Returns:
        Annotated PIL Image
    """
    img_array = np.array(image).copy()
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes
    for block in blocks:
        bbox = block.get('bbox', [])
        
        if not bbox:
            continue
        
        try:
            # Convert bbox to numpy array of points
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], (list, tuple, np.ndarray)):
                    # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    pts = np.array(bbox, dtype=np.int32)
                    if len(pts) >= 4:
                        cv2.polylines(img_array, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                elif len(bbox) == 4:
                    # Format: [x, y, w, h] or [x1, y1, x2, y2]
                    if isinstance(bbox[0], (int, float)):
                        # Try [x, y, w, h] format first
                        x, y, w, h = map(int, bbox[:4])
                        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                        cv2.polylines(img_array, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        except Exception as e:
            logger.debug(f"Error drawing bbox: {e}")
            continue
    
    # Convert back to RGB for PIL
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_array)


def extract_voter_data_from_image(image: Image.Image, page_num: int) -> Tuple[List[Dict], Image.Image]:
    """
    Extract voter data from a single image using PaddleOCR PP-Structure.
    
    Args:
        image: PIL Image object
        page_num: Page number for logging
        
    Returns:
        Tuple of (list of voter dictionaries, annotated image)
    """
    logger.info(f"Processing page {page_num} with PaddleOCR...")
    
    # Step 1: Detect blocks using PP-Structure
    blocks = detect_voter_blocks_with_paddleocr(image)
    
    if not blocks:
        logger.warning(f"No blocks detected on page {page_num}")
        return [], image
    
    logger.info(f"Page {page_num}: Detected {len(blocks)} text blocks")
    
    # Step 2: Parse each block for voter information
    voters = []
    for block_idx, block in enumerate(blocks):
        try:
            text = block.get('text', '')
            if not text.strip():
                continue
            
            # Parse voter fields from text
            voter_data = parse_voter_fields(text)
            
            if voter_data:
                voters.append(voter_data)
            else:
                logger.debug(f"Page {page_num}, Block {block_idx + 1}: Could not parse voter data")
                
        except Exception as e:
            logger.debug(f"Error processing block {block_idx + 1} on page {page_num}: {e}")
            continue
    
    # Step 3: Create annotated image
    annotated_image = draw_detected_blocks(image, blocks)
    
    logger.info(f"Page {page_num}: Extracted {len(voters)} voters from {len(blocks)} blocks")
    return voters, annotated_image


def process_pdf(pdf_path: str) -> bool:
    """
    Process a single PDF file and extract voter data using PaddleOCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_name = Path(pdf_path).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing PDF with PaddleOCR: {pdf_name}")
        logger.info(f"{'='*60}")
        
        # Step 1: Convert PDF to images (using existing pdf2image converter)
        page_images = pdf_to_images(pdf_path)
        
        if not page_images:
            logger.warning(f"No pages to process in {pdf_name}")
            return False
        
        # Step 2: Extract voter data from all pages
        all_voters = []
        annotated_images = []
        
        for page_num, image in page_images:
            voters, annotated_img = extract_voter_data_from_image(image, page_num)
            all_voters.extend(voters)
            annotated_images.append((page_num, annotated_img))
        
        if not all_voters:
            logger.warning(f"No voter data extracted from {pdf_name}")
            return False
        
        # Step 3: Save to CSV
        # Use exact filename as requested, or append PDF name if multiple PDFs
        output_csv_path = OCR_OUTPUT_DIR / "voter_list_extracted.csv"
        
        df = pd.DataFrame(all_voters)
        
        # Sort by serial number if available
        if 'serial_no' in df.columns and df['serial_no'].notna().any():
            df = df.sort_values('serial_no')
        
        # Reorder columns
        column_order = ['serial_no', 'epic_no', 'name', 'relation_type', 
                       'relation_name', 'house_no', 'age', 'gender']
        df = df[[col for col in column_order if col in df.columns]]
        
        # Save CSV with UTF-8 encoding
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"✅ Saved {len(df)} voters to: {output_csv_path}")
        
        # Step 4: Save annotated image (from first page)
        if annotated_images:
            annotated_path = OCR_OUTPUT_DIR / "detected_blocks.png"
            annotated_images[0][1].save(annotated_path)
            logger.info(f"✅ Saved annotated image to: {annotated_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PaddleOCR-based Indian Election Voter Data Extraction")
    print("="*70 + "\n")
    
    # Ensure output directory exists
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    input_dir = Path("input_pdfs")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_dir}")
        print(f"\n❌ No PDF files found in '{input_dir}' directory.")
        print(f"Please place your PDF files in the '{input_dir}' folder and run again.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process PDFs
    successful = 0
    for pdf_path in pdf_files:
        if process_pdf(str(pdf_path)):
            successful += 1
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(pdf_files) - successful}")
    print(f"Output location: {OCR_OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
