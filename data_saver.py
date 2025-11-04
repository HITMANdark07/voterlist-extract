#!/usr/bin/env python3
"""
Data Saver Module
=================
Handles saving extracted voter data to CSV files.
"""

import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def save_voters_to_csv(voters: List[Dict[str, any]], pdf_name: str, output_dir: str = OUTPUT_DIR) -> Path:
    """
    Save extracted voter data to a CSV file.
    
    Args:
        voters: List of voter dictionaries
        pdf_name: Name of the source PDF (without extension)
        output_dir: Output directory path
        
    Returns:
        Path to the saved CSV file
    """
    try:
        if not voters:
            logger.warning(f"No voters to save for {pdf_name}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(voters)
        
        # Sort by serial number if available
        if 'serial_no' in df.columns and df['serial_no'].notna().any():
            df = df.sort_values('serial_no')
        
        # Reorder columns
        column_order = ['serial_no', 'epic_no', 'name', 'relation_type', 
                       'relation_name', 'house_no', 'age', 'gender']
        df = df[[col for col in column_order if col in df.columns]]
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = Path(output_dir) / f"voters_{pdf_name}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"✅ Saved {len(df)} voters to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving CSV for {pdf_name}: {e}")
        return None


def save_voters_to_excel(voters: List[Dict[str, any]], pdf_name: str, output_dir: str = OUTPUT_DIR) -> Path:
    """
    Save extracted voter data to an Excel file.
    
    Args:
        voters: List of voter dictionaries
        pdf_name: Name of the source PDF (without extension)
        output_dir: Output directory path
        
    Returns:
        Path to the saved Excel file
    """
    try:
        if not voters:
            logger.warning(f"No voters to save for {pdf_name}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(voters)
        
        # Sort by serial number if available
        if 'serial_no' in df.columns and df['serial_no'].notna().any():
            df = df.sort_values('serial_no')
        
        # Reorder columns
        column_order = ['serial_no', 'epic_no', 'name', 'relation_type', 
                       'relation_name', 'house_no', 'age', 'gender']
        df = df[[col for col in column_order if col in df.columns]]
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save to Excel
        output_path = Path(output_dir) / f"voters_{pdf_name}.xlsx"
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        logger.info(f"✅ Saved {len(df)} voters to Excel: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving Excel for {pdf_name}: {e}")
        return None


def get_dataframe_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get statistics about the extracted data.
    
    Args:
        df: DataFrame with voter data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_voters': len(df),
        'columns': list(df.columns),
        'missing_data': {}
    }
    
    for col in df.columns:
        missing = df[col].isna().sum()
        stats['missing_data'][col] = {
            'missing': missing,
            'percentage': (missing / len(df)) * 100 if len(df) > 0 else 0
        }
    
    return stats

