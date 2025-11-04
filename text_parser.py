#!/usr/bin/env python3
"""
Text Parser Module
==================
Parses OCR text and extracts structured voter information using regex patterns.
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize OCR text output."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Hindi characters
    text = text.strip()
    return text


def extract_number(text: str) -> Optional[int]:
    """Extract first number from text."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def parse_from_regions(serial_text: str, epic_text: str, details_text: str) -> Optional[Dict[str, any]]:
    """
    Parse voter information from separate OCR regions.
    
    Args:
        serial_text: OCR text from serial number region
        epic_text: OCR text from EPIC number region
        details_text: OCR text from details region
        
    Returns:
        Dictionary with voter information or None
    """
    try:
        voter = {
            'serial_no': None,
            'epic_no': '',
            'name': '',
            'relation_type': '',
            'relation_name': '',
            'house_no': '',
            'age': None,
            'gender': ''
        }
        
        # Extract serial number from serial_text
        serial_no = extract_number(serial_text)
        if serial_no:
            voter['serial_no'] = serial_no
        
        # Extract EPIC number from epic_text
        epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
        epic_match = re.search(epic_pattern, epic_text)
        if epic_match:
            voter['epic_no'] = epic_match.group(1)
        else:
            logger.debug("No EPIC number found in EPIC region")
            return None
        
        # Parse details from details_text
        details = extract_voter_info(details_text, voter['epic_no'])
        
        if details:
            # Merge details with serial and EPIC
            voter['name'] = details.get('name', '')
            voter['relation_type'] = details.get('relation_type', '')
            voter['relation_name'] = details.get('relation_name', '')
            voter['house_no'] = details.get('house_no', '')
            voter['age'] = details.get('age')
            voter['gender'] = details.get('gender', '')
            
            # Use serial_no from serial_text if available, otherwise from details
            if voter['serial_no'] is None:
                voter['serial_no'] = details.get('serial_no')
        
        # Only return if we have at least EPIC and name
        if voter['epic_no'] and voter['name']:
            return voter
        
        return None
        
    except Exception as e:
        logger.debug(f"Error parsing from regions: {e}")
        return None


def parse_single_block(ocr_text: str) -> Optional[Dict[str, any]]:
    """
    Parse a single voter block (for use with grid-detected blocks).
    
    Args:
        ocr_text: OCR text from a single voter block
        
    Returns:
        Dictionary with voter information or None
    """
    if not ocr_text or not ocr_text.strip():
        return None
    
    # Find EPIC number in the block
    epic_pattern = r'\b([A-Z]{2,3}[A-Z0-9]{7,10})\b'
    epic_match = re.search(epic_pattern, ocr_text)
    
    if not epic_match:
        logger.debug("No EPIC number found in block")
        return None
    
    epic_no = epic_match.group(1)
    
    # Parse voter info from the entire block text
    return extract_voter_info(ocr_text, epic_no)


def extract_voter_blocks(ocr_text: str) -> List[Dict[str, any]]:
    """
    Extract structured voter data from OCR text using regex patterns.
    This is the legacy method for full-page OCR.
    
    Args:
        ocr_text: Raw OCR text output from full page
        
    Returns:
        List of dictionaries containing voter information
    """
    voters = []
    
    # Split text into potential blocks (looking for serial numbers as delimiters)
    # Pattern to identify voter blocks - looking for serial numbers followed by EPIC/UIM codes
    lines = ocr_text.split('\n')
    
    # Try to find voter blocks by looking for EPIC numbers and associated data
    # EPIC pattern: UIM followed by digits, or other ID patterns
    epic_pattern = r'([A-Z]{2,3}[A-Z0-9]{7,10})'
    
    # Find all EPIC numbers in the text
    text_cleaned = ' '.join(lines)
    
    # Split by EPIC numbers to create blocks
    epic_matches = list(re.finditer(epic_pattern, text_cleaned))
    
    for i, match in enumerate(epic_matches):
        try:
            epic_no = match.group(1)
            
            # Get context around the EPIC number (previous 200 and next 200 chars)
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(text_cleaned), match.end() + 200)
            block_text = text_cleaned[start_pos:end_pos]
            
            voter_data = extract_voter_info(block_text, epic_no)
            
            if voter_data:
                voters.append(voter_data)
                
        except Exception as e:
            logger.debug(f"Error extracting voter block: {e}")
            continue
    
    logger.info(f"Extracted {len(voters)} voter records from text")
    return voters


def extract_voter_info(block_text: str, epic_no: str) -> Optional[Dict[str, any]]:
    """
    Extract individual voter information from a text block.
    
    Args:
        block_text: Text block containing voter information
        epic_no: EPIC number already identified
        
    Returns:
        Dictionary with voter information or None
    """
    try:
        voter = {
            'serial_no': None,
            'epic_no': epic_no,
            'name': '',
            'relation_type': '',
            'relation_name': '',
            'house_no': '',
            'age': None,
            'gender': ''
        }
        
        # Extract serial number (usually 4 digits)
        serial_match = re.search(r'\b(\d{4})\b', block_text)
        if serial_match:
            voter['serial_no'] = int(serial_match.group(1))
        
        # Extract voter name (निर्वाचक का नाम)
        name_patterns = [
            r'निर्वाचक का नाम\s*[:：]\s*([^\n:]+?)(?:पति|पिता|माता|EPIC|\n)',
            r'नाम\s*[:：]\s*([^\n:]+?)(?:पति|पिता|माता|EPIC|\n)',
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, block_text)
            if name_match:
                voter['name'] = clean_text(name_match.group(1))
                break
        
        # Extract relation type and name
        # पति का नाम (Husband's name)
        if re.search(r'पति\s*का\s*नाम', block_text):
            voter['relation_type'] = 'पति'
            relation_match = re.search(r'पति\s*का\s*नाम\s*[:：]\s*([^\n:]+?)(?:मकान|आयु|EPIC|\n)', block_text)
            if relation_match:
                voter['relation_name'] = clean_text(relation_match.group(1))
        
        # पिता का नाम (Father's name)
        elif re.search(r'पिता\s*का\s*नाम', block_text):
            voter['relation_type'] = 'पिता'
            relation_match = re.search(r'पिता\s*का\s*नाम\s*[:：]\s*([^\n:]+?)(?:मकान|आयु|EPIC|\n)', block_text)
            if relation_match:
                voter['relation_name'] = clean_text(relation_match.group(1))
        
        # माता का नाम (Mother's name)
        elif re.search(r'माता\s*का\s*नाम', block_text):
            voter['relation_type'] = 'माता'
            relation_match = re.search(r'माता\s*का\s*नाम\s*[:：]\s*([^\n:]+?)(?:मकान|आयु|EPIC|\n)', block_text)
            if relation_match:
                voter['relation_name'] = clean_text(relation_match.group(1))
        
        # Extract house number (मकान संख्या)
        house_patterns = [
            r'मकान\s*संख्या\s*[:：]\s*(\d+)',
            r'संख्या\s*[:：]\s*(\d+)',
        ]
        for pattern in house_patterns:
            house_match = re.search(pattern, block_text)
            if house_match:
                voter['house_no'] = house_match.group(1)
                break
        
        # Extract age (आयु)
        age_patterns = [
            r'आयु\s*[:：]\s*(\d+)',
            r'उम्र\s*[:：]\s*(\d+)',
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, block_text)
            if age_match:
                voter['age'] = int(age_match.group(1))
                break
        
        # Extract gender (लिंग)
        if re.search(r'महिला|female', block_text, re.IGNORECASE):
            voter['gender'] = 'महिला'
        elif re.search(r'पुरुष|male', block_text, re.IGNORECASE):
            voter['gender'] = 'पुरुष'
        
        # Only return if we have at least EPIC and name
        if voter['epic_no'] and voter['name']:
            return voter
        
        return None
        
    except Exception as e:
        logger.debug(f"Error parsing voter info: {e}")
        return None

