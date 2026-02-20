"""
Medical Report OCR Extraction Agent - Simplified Tesseract Version
Uses Tesseract OCR only (same as nutrition scanner)
Extracts structured data from medical report images/PDFs
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pytesseract
from PIL import Image, ImageEnhance

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️ PyPDF2 not available")

# Load report schemas
SCHEMAS_PATH = Path(__file__).parent.parent / 'report_schemas.json'
with open(SCHEMAS_PATH, 'r') as f:
    REPORT_SCHEMAS = json.load(f)


class ReportExtractor:
    """Base class for medical report extraction"""
    
    def __init__(self):
        self.extracted_data = {}
        self.raw_text = ""
        self.confidence_scores = {}
    
    def extract_from_pdf(self, pdf_path: str, report_type: str) -> Dict[str, Any]:
        """
        Main extraction pipeline with detailed logging
        
        Args:
            pdf_path: Path to PDF file
            report_type: Type of report (lipid_profile, cbc, etc.)
            
        Returns:
            Dictionary with extracted data, form mappings, and metadata
        """
        try:
            print(f"\n{'='*60}")
            print(f"🔍 EXTRACTING REPORT: {os.path.basename(pdf_path)}")
            print(f"📋 Report Type: {report_type}")
            print(f"{'='*60}")
            
            # Step 1: Extract text from PDF using OCR
            self.raw_text = self._extract_text_from_pdf(pdf_path)
            
            print(f"\n📄 OCR RESULTS:")
            print(f"   Text length: {len(self.raw_text)} characters")
            print(f"   First 300 chars:\n{self.raw_text[:300]}")
            
            if not self.raw_text or len(self.raw_text.strip()) < 10:
                print(f"❌ ERROR: No text extracted from PDF!")
                return {
                    'success': False,
                    'error': 'Could not extract text from PDF. Please ensure it is a readable document.',
                    'report_type': report_type,
                    'form_data': {},
                    'complete_data': {},
                    'raw_text': self.raw_text
                }
            
            # Step 2: Extract structured data based on report type
            if report_type not in REPORT_SCHEMAS:
                raise ValueError(f"Unknown report type: {report_type}")
            
            schema = REPORT_SCHEMAS[report_type]
            self.extracted_data = self._extract_fields(self.raw_text, schema)
            
            print(f"\n✅ EXTRACTION COMPLETE:")
            print(f"   Fields extracted: {len(self.extracted_data)}")
            for field, value in self.extracted_data.items():
                conf = self.confidence_scores.get(field, 0)
                print(f"   - {field}: {value} (confidence: {conf:.2f})")
            
            # Step 3: Map to form fields
            form_data = self._map_to_form_fields(self.extracted_data, schema['form_mappings'])
            
            print(f"\n📋 FORM FIELD MAPPINGS:")
            print(f"   Mapped fields: {len(form_data)}")
            for field, value in form_data.items():
                print(f"   - {field} → {value}")
            
            # Step 4: Prepare complete response
            return {
                'success': True,
                'report_type': report_type,
                'form_data': form_data,  # Data for auto-filling assessment form
                'complete_data': self.extracted_data,  # ALL extracted fields (including unused)
                'raw_text': self.raw_text,  # Full OCR text for debugging
                'confidence_scores': self.confidence_scores,
                'metadata': {
                    'schema_version': '1.0',
                    'extraction_method': 'tesseract_ocr',
                    'file_name': os.path.basename(pdf_path),
                    'fields_extracted': len(self.extracted_data),
                    'fields_mapped': len(form_data)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'report_type': report_type,
                'form_data': {},
                'complete_data': {},
                'raw_text': self.raw_text
            }
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using Tesseract OCR
        
        Strategy (same as nutrition_analyzer.py):
        1. Convert PDF to image (using pdf2image or PyMuPDF)
        2. Preprocess image (grayscale + contrast)
        3. Run Tesseract OCR
        
        Multiple fallback methods to ensure it works!
        """
        print(f"🔍 PROCESSING PDF: {os.path.basename(pdf_path)}")
        
        image = None
        
        # Method 1: Try pdf2image (requires poppler)
        if PDF2IMAGE_AVAILABLE and image is None:
            try:
                print("📄 Method 1: Trying pdf2image...")
                images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
                image = images[0]
                print(f"✅ pdf2image SUCCESS: {image.size}")
            except Exception as e:
                print(f"⚠️ pdf2image failed: {str(e)}")
                image = None
        
        # Method 2: Try PyMuPDF (no poppler needed!)
        if PYMUPDF_AVAILABLE and image is None:
            try:
                print("📄 Method 2: Trying PyMuPDF...")
                import io
                doc = fitz.open(pdf_path)
                page = doc[0]  # First page
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                doc.close()
                print(f"✅ PyMuPDF SUCCESS: {image.size}")
            except Exception as e:
                print(f"⚠️ PyMuPDF failed: {str(e)}")
                image = None
        
        # Method 3: Try opening as image (for image files renamed as PDF)
        if image is None:
            try:
                print("� Method 3: Trying direct image open...")
                image = Image.open(pdf_path)
                print(f"✅ Direct image open SUCCESS: {image.size}")
            except Exception as e:
                print(f"⚠️ Direct image open failed: {str(e)}")
                error_msg = (
                    f"Cannot extract from PDF. All methods failed!\n"
                    f"Install PyMuPDF: pip install PyMuPDF\n"
                    f"OR install pdf2image + poppler"
                )
                print(f"❌ ERROR: {error_msg}")
                raise Exception(error_msg)
        
        # Preprocess image (EXACTLY like nutrition_analyzer.py)
        print("🔧 Preprocessing image...")
        image = image.convert('L')  # Grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        print("✅ Image preprocessed")
        
        # Tesseract OCR (EXACTLY like nutrition_analyzer.py)
        print("🔤 Running Tesseract OCR...")
        text = pytesseract.image_to_string(image)
        print(f"✅ OCR COMPLETE: Extracted {len(text)} characters")
        if len(text) > 0:
            print(f"📄 First 200 chars: {text[:200]}")
        else:
            print("⚠️ WARNING: No text extracted!")
        
        return text
    
    def _extract_fields(self, text: str, schema: Dict) -> Dict[str, Any]:
        """
        Extract all fields from text based on schema patterns
        
        Args:
            text: OCR extracted text
            schema: Report schema with extraction patterns
            
        Returns:
            Dictionary of extracted values
        """
        extracted = {}
        text_lower = text.lower()
        
        for field_name, patterns in schema['extraction_patterns'].items():
            value, confidence = self._extract_field_value(text_lower, patterns)
            
            if value is not None:
                extracted[field_name] = value
                self.confidence_scores[field_name] = confidence
        
        return extracted
    
    def _extract_field_value(self, text: str, patterns: List[str]) -> Tuple[Optional[float], float]:
        """
        Extract numeric value for a field using pattern matching
        
        Args:
            text: Text to search in (lowercase)
            patterns: List of possible field name patterns
            
        Returns:
            Tuple of (value, confidence_score)
        """
        for pattern in patterns:
            # Create regex pattern to find the field and its value
            # Looks for pattern like: "HDL: 45" or "HDL 45" or "HDL - 45 mg/dL"
            regex_pattern = rf'{re.escape(pattern)}\s*[:=-]?\s*(\d+\.?\d*)'
            
            match = re.search(regex_pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    confidence = 0.9 if pattern == patterns[0] else 0.7  # Higher confidence for primary pattern
                    return value, confidence
                except ValueError:
                    continue
        
        return None, 0.0
    
    def _map_to_form_fields(self, extracted_data: Dict, mappings: Dict) -> Dict[str, float]:
        """
        Map extracted data to assessment form field names
        
        Args:
            extracted_data: All extracted fields
            mappings: Schema mappings (extracted_field -> form_field)
            
        Returns:
            Dictionary with form field names as keys
        """
        form_data = {}
        
        for extracted_field, form_field in mappings.items():
            if extracted_field in extracted_data:
                form_data[form_field] = extracted_data[extracted_field]
        
        return form_data
    
    def get_ai_context_data(self) -> Dict[str, Any]:
        """
        Prepare complete report data for Gemini AI enhancement
        
        Returns:
            Dictionary with all extracted data + metadata for AI context
        """
        return {
            'extracted_values': self.extracted_data,
            'confidence_scores': self.confidence_scores,
            'raw_text_sample': self.raw_text[:500] if len(self.raw_text) > 500 else self.raw_text,
            'field_count': len(self.extracted_data),
            'high_confidence_fields': [
                field for field, conf in self.confidence_scores.items() if conf >= 0.8
            ],
            'low_confidence_fields': [
                field for field, conf in self.confidence_scores.items() if conf < 0.6
            ]
        }


class LipidProfileExtractor(ReportExtractor):
    """Specialized extractor for Lipid Profile reports"""
    
    def validate_extracted_data(self) -> Dict[str, str]:
        """Validate extracted lipid values against normal ranges"""
        warnings = {}
        schema = REPORT_SCHEMAS['lipid_profile']
        
        for field, value in self.extracted_data.items():
            if field in schema['normal_ranges']:
                ranges = schema['normal_ranges'][field]
                if value < ranges['min'] or value > ranges['max']:
                    warnings[field] = f"Value {value} outside normal range ({ranges['min']}-{ranges['max']})"
        
        return warnings


class CBCExtractor(ReportExtractor):
    """Specialized extractor for CBC (Complete Blood Count) reports"""
    
    def validate_extracted_data(self, gender: Optional[str] = None) -> Dict[str, str]:
        """Validate extracted CBC values against normal ranges"""
        warnings = {}
        schema = REPORT_SCHEMAS['cbc']
        
        for field, value in self.extracted_data.items():
            if field in schema['normal_ranges']:
                ranges = schema['normal_ranges'][field]
                
                # Handle gender-specific ranges
                if isinstance(ranges, dict) and 'male' in ranges:
                    if gender:
                        ranges = ranges.get(gender.lower(), ranges['male'])
                    else:
                        # Use male ranges as default if gender unknown
                        ranges = ranges['male']
                
                if value < ranges['min'] or value > ranges['max']:
                    warnings[field] = f"Value {value} outside normal range ({ranges['min']}-{ranges['max']})"
        
        return warnings


class HbA1cExtractor(ReportExtractor):
    """Specialized extractor for HbA1c reports"""
    
    def interpret_hba1c(self, value: float) -> str:
        """Interpret HbA1c value"""
        if value < 5.7:
            return "Normal"
        elif value < 6.5:
            return "Prediabetes"
        else:
            return "Diabetes"


def extract_report(pdf_path: str, report_type: str, gender: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to extract data from medical reports
    
    Args:
        pdf_path: Path to PDF file
        report_type: Type of report (lipid_profile, cbc, hba1c, etc.)
        gender: Patient gender (for CBC validation)
        
    Returns:
        Complete extraction result with form data and metadata
    """
    # Select appropriate extractor
    if report_type == 'lipid_profile':
        extractor = LipidProfileExtractor()
    elif report_type == 'cbc':
        extractor = CBCExtractor()
    elif report_type == 'hba1c':
        extractor = HbA1cExtractor()
    else:
        # Use base extractor for other report types
        extractor = ReportExtractor()
    
    # Extract data
    result = extractor.extract_from_pdf(pdf_path, report_type)
    
    # Add validation warnings if available
    if result['success']:
        if isinstance(extractor, LipidProfileExtractor):
            result['warnings'] = extractor.validate_extracted_data()
        elif isinstance(extractor, CBCExtractor):
            result['warnings'] = extractor.validate_extracted_data(gender)
        
        # Add AI context data
        result['ai_context'] = extractor.get_ai_context_data()
    
    return result


def get_available_report_types() -> List[Dict[str, str]]:
    """Get list of supported report types"""
    return [
        {
            'value': report_type,
            'display_name': schema['display_name'],
            'description': schema['description']
        }
        for report_type, schema in REPORT_SCHEMAS.items()
    ]


if __name__ == "__main__":
    # Test the extractor
    print("Available report types:")
    for rt in get_available_report_types():
        print(f"  - {rt['display_name']}: {rt['description']}")
    
    # Example usage
    # result = extract_report('sample_lipid_profile.pdf', 'lipid_profile')
    # print(json.dumps(result, indent=2))
