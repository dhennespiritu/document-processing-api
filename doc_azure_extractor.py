import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from azure.core.exceptions import AzureError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import os
import re
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
current_file = Path(__file__)
project_root = current_file.parent
env_file = project_root / ".env"
load_dotenv(env_file)

# Initialize Azure Document Intelligence client
doc_ai_endpoint = os.getenv('AZURE_DOCAI_API_ENDPOINT')
doc_ai_key = os.getenv('AZURE_DOCAI_API_KEY')
azure_client = DocumentIntelligenceClient(endpoint=doc_ai_endpoint,
                                          credential=AzureKeyCredential(doc_ai_key))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentTypeResult:
    document_type: str
    confidence_score: float
    specialized_model: str
    indicators_found: List[str]
    extraction_strategy: Dict[str, Any]


@dataclass
class AzureExtractionResult:
    file_path: str
    content: str
    document_info: Dict[str, Any]
    extraction_method: str
    azure_model_used: str
    pages_processed: int
    paragraphs_found: int
    text_length: int
    tables_found: int
    tables_data: List[Dict[str, Any]]
    key_value_pairs_found: int
    key_value_pairs: Dict[str, str]
    confidence_scores: Dict[str, float]
    meta_data: Dict[str, Any]
    document_type_result: Optional[DocumentTypeResult] = None


class DocumentTypeDetector:
    """Enhanced document type detection using content analysis and pattern matching"""
    
    def __init__(self):
        self.document_patterns = self._initialize_patterns()
        self.specialized_models = self._initialize_specialized_models()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for different document types"""
        return {
            'invoice': {
                'keywords': [
                    'invoice', 'bill to', 'ship to', 'invoice number', 'invoice date',
                    'due date', 'subtotal', 'tax', 'total amount', 'amount due',
                    'payment terms', 'remit to', 'vendor', 'customer', 'qty', 'quantity',
                    'unit price', 'line total', 'net amount', 'gross amount'
                ],
                'patterns': [
                    r'invoice\s*#?\s*:?\s*\d+',
                    r'inv\s*#?\s*:?\s*\d+',
                    r'bill\s*to\s*:',
                    r'ship\s*to\s*:',
                    r'total\s*amount\s*:?\s*\$?[\d,]+\.?\d*',
                    r'amount\s*due\s*:?\s*\$?[\d,]+\.?\d*',
                    r'tax\s*:?\s*\$?[\d,]+\.?\d*'
                ],
                'structure_indicators': ['itemized_list', 'billing_address', 'totals_section'],
                'weight': 1.0
            },
            'receipt': {
                'keywords': [
                    'receipt', 'thank you', 'transaction', 'purchase', 'sale',
                    'cash', 'credit', 'debit', 'change', 'tender', 'merchant',
                    'terminal', 'card number', 'authorization', 'ref number'
                ],
                'patterns': [
                    r'receipt\s*#?\s*:?\s*\d+',
                    r'transaction\s*#?\s*:?\s*\d+',
                    r'auth\s*:?\s*\d+',
                    r'terminal\s*:?\s*\d+',
                    r'change\s*:?\s*\$?[\d,]+\.?\d*'
                ],
                'structure_indicators': ['timestamp', 'merchant_info', 'payment_method'],
                'weight': 0.9
            },
            'bank_statement': {
                'keywords': [
                    'statement', 'account statement', 'bank statement', 'balance',
                    'beginning balance', 'ending balance', 'deposit', 'withdrawal',
                    'transaction', 'account number', 'routing number', 'statement period',
                    'available balance', 'pending', 'overdraft', 'fee', 'interest'
                ],
                'patterns': [
                    r'account\s*#?\s*:?\s*\d{4,}',
                    r'routing\s*#?\s*:?\s*\d{9}',
                    r'statement\s*period\s*:?',
                    r'beginning\s*balance\s*:?\s*\$?[\d,]+\.?\d*',
                    r'ending\s*balance\s*:?\s*\$?[\d,]+\.?\d*'
                ],
                'structure_indicators': ['transaction_list', 'account_summary', 'date_range'],
                'weight': 1.0
            },
            'contract': {
                'keywords': [
                    'contract', 'agreement', 'party', 'parties', 'whereas', 'therefore',
                    'terms and conditions', 'effective date', 'termination', 'obligations',
                    'rights', 'liability', 'indemnification', 'governing law',
                    'signature', 'witness', 'notary'
                ],
                'patterns': [
                    r'this\s*agreement',
                    r'party\s*of\s*the\s*first\s*part',
                    r'whereas\s*,',
                    r'effective\s*date\s*:?',
                    r'governing\s*law\s*:?'
                ],
                'structure_indicators': ['signature_block', 'clauses', 'parties_section'],
                'weight': 0.95
            },
            'insurance_document': {
                'keywords': [
                    'policy', 'premium', 'deductible', 'coverage', 'claim',
                    'insured', 'insurer', 'beneficiary', 'policy number',
                    'effective period', 'liability', 'coverage limit',
                    'exclusions', 'endorsement'
                ],
                'patterns': [
                    r'policy\s*#?\s*:?\s*[A-Z0-9]+',
                    r'premium\s*:?\s*\$?[\d,]+\.?\d*',
                    r'deductible\s*:?\s*\$?[\d,]+\.?\d*',
                    r'coverage\s*limit\s*:?',
                    r'effective\s*period\s*:?'
                ],
                'structure_indicators': ['policy_details', 'coverage_table', 'terms_conditions'],
                'weight': 0.9
            },
            'tax_document': {
                'keywords': [
                    'tax', 'w-2', 'w-4', '1099', 'tax return', 'irs', 'withholding',
                    'taxable income', 'deductions', 'credits', 'refund', 'owed',
                    'federal', 'state', 'social security', 'medicare', 'ein', 'ssn'
                ],
                'patterns': [
                    r'form\s*\d{4}[a-z]*',
                    r'tax\s*year\s*:?\s*\d{4}',
                    r'ein\s*:?\s*\d{2}-\d{7}',
                    r'ssn\s*:?\s*\d{3}-\d{2}-\d{4}',
                    r'withholding\s*:?\s*\$?[\d,]+\.?\d*'
                ],
                'structure_indicators': ['tax_forms', 'income_sections', 'calculation_areas'],
                'weight': 1.0
            },
            'medical_record': {
                'keywords': [
                    'patient', 'diagnosis', 'treatment', 'medication', 'prescription',
                    'doctor', 'physician', 'hospital', 'clinic', 'medical record',
                    'symptoms', 'allergies', 'vital signs', 'blood pressure',
                    'temperature', 'pulse', 'medical history'
                ],
                'patterns': [
                    r'patient\s*id\s*:?\s*\d+',
                    r'dob\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                    r'bp\s*:?\s*\d+/\d+',
                    r'temp\s*:?\s*\d+\.?\d*',
                    r'diagnosis\s*:?'
                ],
                'structure_indicators': ['patient_info', 'vitals_section', 'notes_section'],
                'weight': 0.95
            },
            'legal_document': {
                'keywords': [
                    'plaintiff', 'defendant', 'court', 'case', 'docket', 'filing',
                    'motion', 'brief', 'memorandum', 'order', 'judgment', 'appeal',
                    'counsel', 'attorney', 'jurisdiction', 'statute', 'law'
                ],
                'patterns': [
                    r'case\s*#?\s*:?\s*\d+-[A-Z]+-\d+',
                    r'docket\s*#?\s*:?\s*\d+',
                    r'vs\.?\s*[A-Z]',
                    r'court\s*of\s*[A-Z]',
                    r'honorable\s*[A-Z]'
                ],
                'structure_indicators': ['case_header', 'legal_citations', 'signature_block'],
                'weight': 0.9
            }
        }
    
    def _initialize_specialized_models(self) -> Dict[str, str]:
        """Map document types to specialized Azure models"""
        return {
            'invoice': 'prebuilt-invoice',
            'receipt': 'prebuilt-receipt',
            'bank_statement': 'prebuilt-document',  # No specialized model yet
            'contract': 'prebuilt-contract',
            'insurance_document': 'prebuilt-document',
            'tax_document': 'prebuilt-tax.us.w2',  # Example for W-2
            'medical_record': 'prebuilt-document',
            'legal_document': 'prebuilt-document',
            'business_card': 'prebuilt-businessCard',
            'id_document': 'prebuilt-idDocument',
            'default': 'prebuilt-layout'
        }
    
    def analyze_text_content(self, text: str) -> Dict[str, float]:
        """Analyze text content for document type indicators"""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, config in self.document_patterns.items():
            score = 0.0
            indicators_found = []
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] 
                                if keyword.lower() in text_lower)
            keyword_score = (keyword_matches / len(config['keywords'])) * 0.4
            
            # Check patterns
            pattern_matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1
                    indicators_found.append(f"Pattern: {pattern}")
            
            pattern_score = (pattern_matches / len(config['patterns'])) * 0.6
            
            # Combined score with weight
            combined_score = (keyword_score + pattern_score) * config['weight']
            scores[doc_type] = {
                'score': combined_score,
                'keyword_matches': keyword_matches,
                'pattern_matches': pattern_matches,
                'indicators': indicators_found
            }
        
        return scores
    
    def analyze_structure(self, result) -> Dict[str, Any]:
        """Analyze document structure for additional clues"""
        structure_info = {
            'has_tables': bool(getattr(result, 'tables', None)),
            'table_count': len(result.tables) if getattr(result, 'tables', None) else 0,
            'has_key_value_pairs': bool(getattr(result, 'key_value_pairs', None)),
            'kv_count': len(result.key_value_pairs) if getattr(result, 'key_value_pairs', None) else 0,
            'page_count': len(result.pages) if getattr(result, 'pages', None) else 0,
            'has_form_fields': False,  # Can be enhanced based on form field detection
        }
        
        # Enhanced structure analysis
        if structure_info['has_tables'] and structure_info['table_count'] > 2:
            structure_info['likely_financial'] = True
        
        if structure_info['has_key_value_pairs'] and structure_info['kv_count'] > 5:
            structure_info['likely_form'] = True
        
        return structure_info
    
    def detect_document_type(self, text_content: str, azure_result=None) -> DocumentTypeResult:
        """Enhanced document type detection using content and structure analysis"""
        
        # Analyze text content
        content_scores = self.analyze_text_content(text_content)
        
        # Analyze structure if available
        structure_info = {}
        if azure_result:
            structure_info = self.analyze_structure(azure_result)
        
        # Find best match
        best_type = 'unknown'
        best_score = 0.0
        best_indicators = []
        
        for doc_type, score_info in content_scores.items():
            current_score = score_info['score']
            
            # Apply structure bonuses
            if doc_type == 'invoice' and structure_info.get('has_tables'):
                current_score += 0.1
            elif doc_type == 'bank_statement' and structure_info.get('likely_financial'):
                current_score += 0.15
            elif doc_type in ['contract', 'legal_document'] and structure_info.get('page_count', 0) > 3:
                current_score += 0.1
            
            if current_score > best_score:
                best_score = current_score
                best_type = doc_type
                best_indicators = score_info['indicators']
        
        # Determine specialized model
        specialized_model = self.specialized_models.get(best_type, self.specialized_models['default'])
        
        # Create extraction strategy
        extraction_strategy = self._create_extraction_strategy(best_type, structure_info)
        
        return DocumentTypeResult(
            document_type=best_type,
            confidence_score=min(best_score, 1.0),  # Cap at 1.0
            specialized_model=specialized_model,
            indicators_found=best_indicators,
            extraction_strategy=extraction_strategy
        )
    
    def _create_extraction_strategy(self, doc_type: str, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create extraction strategy based on document type"""
        strategies = {
            'invoice': {
                'focus_areas': ['vendor_info', 'line_items', 'totals', 'due_date'],
                'key_fields': ['invoice_number', 'invoice_date', 'total_amount', 'vendor_name'],
                'table_processing': 'detailed',
                'kv_extraction': True
            },
            'receipt': {
                'focus_areas': ['merchant_info', 'items', 'payment_method', 'totals'],
                'key_fields': ['merchant_name', 'transaction_date', 'total', 'payment_method'],
                'table_processing': 'simple',
                'kv_extraction': True
            },
            'bank_statement': {
                'focus_areas': ['account_info', 'transactions', 'balances', 'summary'],
                'key_fields': ['account_number', 'statement_date', 'beginning_balance', 'ending_balance'],
                'table_processing': 'detailed',
                'kv_extraction': False
            },
            'contract': {
                'focus_areas': ['parties', 'terms', 'signatures', 'dates'],
                'key_fields': ['contract_date', 'parties', 'effective_date', 'termination_date'],
                'table_processing': 'minimal',
                'kv_extraction': True
            }
        }
        
        return strategies.get(doc_type, {
            'focus_areas': ['general_content'],
            'key_fields': [],
            'table_processing': 'standard',
            'kv_extraction': True
        })


class AzureExtractor:
    def __init__(self, azure_client):
        self.azure_client = azure_client
        self.document_type_detector = DocumentTypeDetector()
        self.document_types = {
            '.pdf': 'document',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.tiff': 'image',
            '.tif': 'image',
            '.bmp': 'image',
            '.docx': 'document',
            '.doc': 'document'
        }

    def detect_document_type(self, file_path: Path, initial_extraction: bool = False) -> Dict[str, Any]:
        """Enhanced document type detection with optional initial extraction"""
        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        doc_info = {
            'file_extension': file_ext,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'document_category': self.document_types.get(file_ext, 'unknown')
        }

        # If initial extraction is requested, do a quick analysis
        if initial_extraction and self.azure_client:
            try:
                # Quick extraction with basic model for content analysis
                content_type = self.get_content_type(file_ext)
                with open(file_path, "rb") as file:
                    poller = self.azure_client.begin_analyze_document(
                        model_id="prebuilt-read",  # Fast model for initial analysis
                        body=file,
                        content_type=content_type
                    )
                    result = poller.result()
                
                # Use content for enhanced detection
                text_content = result.content if result.content else ""
                doc_type_result = self.document_type_detector.detect_document_type(text_content, result)
                
                doc_info.update({
                    'detected_type': doc_type_result.document_type,
                    'detection_confidence': doc_type_result.confidence_score,
                    'recommended_model': doc_type_result.specialized_model,
                    'indicators_found': doc_type_result.indicators_found,
                    'extraction_strategy': doc_type_result.extraction_strategy
                })
                
            except Exception as e:
                logger.warning(f"Initial extraction failed, falling back to basic detection: {e}")
                # Fallback to basic detection
                doc_info.update(self._basic_document_detection(file_ext))
        else:
            # Basic detection without content analysis
            doc_info.update(self._basic_document_detection(file_ext))

        table_likely_formats = {'.pdf', '.docx', '.doc'}
        doc_info['likely_has_tables'] = file_ext in table_likely_formats

        return doc_info
    
    def _basic_document_detection(self, file_ext: str) -> Dict[str, Any]:
        """Fallback basic document detection"""
        if file_ext == '.pdf':
            recommended_model = 'prebuilt-layout'
        elif file_ext in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}:
            recommended_model = 'prebuilt-read'
        else:
            recommended_model = 'prebuilt-document'
            
        return {
            'detected_type': 'unknown',
            'detection_confidence': 0.5,
            'recommended_model': recommended_model,
            'indicators_found': [],
            'extraction_strategy': {}
        }

    @staticmethod
    def get_content_type(file_ext: str) -> str:
        """Automatically determine content type based on file extension."""
        if file_ext == '.pdf':
            return "application/pdf"
        elif file_ext in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}:
            return f"image/{file_ext.replace('.', '')}"
        elif file_ext in {'.doc', '.docx'}:
            return "application/octet-stream"
        else:
            return "application/octet-stream"

    def extract_tables(self, result) -> List[Dict[str, Any]]:
        tables = []
        if not getattr(result, 'tables', None):
            return tables

        for table_idx, table in enumerate(result.tables):
            table_data = {
                'table_id': table_idx,
                'row_count': table.row_count,
                'column_count': table.column_count,
                'cells': [],
                'structured_data': []
            }

            if table.cells:
                for cell in table.cells:
                    cell_data = {
                        'row': cell.row_index,
                        'column': cell.column_index,
                        'content': cell.content or '',
                        'confidence': getattr(cell, 'confidence', None),
                        'is_header': getattr(cell, 'kind', '') == 'columnHeader'
                    }
                    table_data['cells'].append(cell_data)

            if table_data['cells']:
                structured = [[''] * table.column_count for _ in range(table.row_count)]
                for cell in table_data['cells']:
                    if cell['row'] < table.row_count and cell['column'] < table.column_count:
                        structured[cell['row']][cell['column']] = cell['content']
                table_data['structured_data'] = structured
                if structured:
                    table_data['headers'] = structured[0]
                    table_data['data_rows'] = structured[1:] if len(structured) > 1 else []

            tables.append(table_data)
        return tables

    def extract_key_value_pairs(self, result) -> Dict[str, str]:
        kv_pairs = {}
        if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
            for kv in result.key_value_pairs:
                if kv.key and kv.value:
                    key = kv.key.content if hasattr(kv.key, 'content') else str(kv.key)
                    value = kv.value.content if hasattr(kv.value, 'content') else str(kv.value)
                    kv_pairs[key] = value
        return kv_pairs

    def calculate_confidence_scores(self, result) -> Dict[str, float]:
        confidence_data = {}
        text_confidences, table_confidences = [], []

        if hasattr(result, 'pages') and result.pages:
            for page in result.pages:
                if hasattr(page, 'words') and page.words:
                    for word in page.words:
                        if getattr(word, 'confidence', None) is not None:
                            text_confidences.append(word.confidence)

        if hasattr(result, 'tables') and result.tables:
            for table in result.tables:
                if getattr(table, 'cells', None):
                    for cell in table.cells:
                        if getattr(cell, 'confidence', None) is not None:
                            table_confidences.append(cell.confidence)

        if text_confidences:
            confidence_data['text_avg_confidence'] = sum(text_confidences) / len(text_confidences)
            confidence_data['text_min_confidence'] = min(text_confidences)
            confidence_data['text_confidence_count'] = len(text_confidences)

        if table_confidences:
            confidence_data['table_avg_confidence'] = sum(table_confidences) / len(table_confidences)
            confidence_data['table_min_confidence'] = min(table_confidences)

        if text_confidences and table_confidences:
            all_confidences = text_confidences + table_confidences
            confidence_data['overall_confidence'] = sum(all_confidences) / len(all_confidences)
        elif text_confidences:
            confidence_data['overall_confidence'] = confidence_data['text_avg_confidence']
        elif table_confidences:
            confidence_data['overall_confidence'] = confidence_data['table_avg_confidence']

        return confidence_data

    def extract_with_azure(self, file_path: Path, use_enhanced_detection: bool = True) -> AzureExtractionResult:
        if not self.azure_client:
            raise ValueError("Azure client not initialized")

        try:
            # Enhanced document type detection
            doc_info = self.detect_document_type(file_path, initial_extraction=use_enhanced_detection)
            logger.info(f"Document info: {doc_info}")

            content_type = self.get_content_type(doc_info['file_extension'])
            
            # Use the recommended model from enhanced detection
            model_to_use = doc_info.get('recommended_model', 'prebuilt-layout')

            with open(file_path, "rb") as file:
                poller = self.azure_client.begin_analyze_document(
                    model_id=model_to_use,
                    body=file,
                    content_type=content_type
                )
                result = poller.result()

            # If we used basic detection initially, now do enhanced detection with full content
            doc_type_result = None
            if use_enhanced_detection:
                text_content = result.content if result.content else ""
                doc_type_result = self.document_type_detector.detect_document_type(text_content, result)
                
                # Update doc_info with refined detection results
                doc_info.update({
                    'detected_type': doc_type_result.document_type,
                    'detection_confidence': doc_type_result.confidence_score,
                    'final_model_used': model_to_use,
                    'indicators_found': doc_type_result.indicators_found,
                    'extraction_strategy': doc_type_result.extraction_strategy
                })

            # Extract content
            content_parts = []
            if result.content:
                content_parts.append(result.content)
            if result.paragraphs:
                for p in result.paragraphs:
                    if p.content and p.content.strip() and p.content not in content_parts:
                        content_parts.append(p.content)
            extracted_text = "\n".join(content_parts)

            # Extract tables and key-values
            tables = self.extract_tables(result)
            kv_pairs = self.extract_key_value_pairs(result)
            confidence_scores = self.calculate_confidence_scores(result)

            # Build full content for chunking
            full_content_for_chunking = extracted_text

            if tables:
                for i, table in enumerate(tables):
                    full_content_for_chunking += f"\n[TABLE {i+1}: {table['row_count']}x{table['column_count']}]"
                    if table.get('headers'):
                        full_content_for_chunking += "\nHeaders: " + ", ".join(table['headers'])
                    for row in table.get('data_rows', []):
                        full_content_for_chunking += "\n" + "\t".join(row)

            if kv_pairs:
                full_content_for_chunking += "\n\nKEY-VALUE PAIRS:\n" + "\n".join([f"{k}: {v}" for k, v in kv_pairs.items()])

            # Build metadata
            metadata = {
                'document_info': doc_info,
                'extraction_method': 'azure_document_intelligence_enhanced',
                'azure_model_used': model_to_use,
                'pages_processed': len(result.pages) if getattr(result, "pages", None) else 0,
                'paragraphs_found': len(result.paragraphs) if getattr(result, "paragraphs", None) else 0,
                'text_length': len(extracted_text),
                'tables_found': len(tables),
                'key_value_pairs_found': len(kv_pairs),
            }

            # Build result object
            result_obj = AzureExtractionResult(
                file_path=str(file_path),
                content=full_content_for_chunking,
                document_info=doc_info,
                extraction_method="azure_document_intelligence_enhanced",
                azure_model_used=model_to_use,
                pages_processed=len(result.pages) if result.pages else 0,
                paragraphs_found=len(result.paragraphs) if result.paragraphs else 0,
                text_length=len(extracted_text),
                tables_found=len(tables),
                tables_data=tables,
                key_value_pairs_found=len(kv_pairs),
                key_value_pairs=kv_pairs,
                confidence_scores=confidence_scores,
                meta_data=metadata,
                document_type_result=doc_type_result
            )

            logger.info(f"Azure extraction complete: {len(full_content_for_chunking)} chars, "
                        f"{len(tables)} tables, {len(kv_pairs)} key-value pairs, "
                        f"detected type: {doc_info.get('detected_type', 'unknown')} "
                        f"(confidence: {doc_info.get('detection_confidence', 0):.2f})")

            return result_obj

        except AzureError as e:
            logger.error(f"Azure API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected extraction error: {e}")
            raise


# # For testing
# def test():
#     extractor = AzureExtractor(azure_client)

#     processed_doc = extractor.extract_with_azure(
#         Path(r"C:\Users\dv146ms\OneDrive - EY\00 - RFP\GPT\sample documents\Rouse Hill - Country Road - ASIC On-File Report Current - COUNTRY ROAD CLOTHING PTY. LTD. ACN 005 419 447.pdf").resolve(),
#         use_enhanced_detection=True
#     )

#     print("=== DOCUMENT TYPE DETECTION RESULTS ===")
#     if processed_doc.document_type_result:
#         print(f"Detected Type: {processed_doc.document_type_result.document_type}")
#         print(f"Confidence: {processed_doc.document_type_result.confidence_score:.2f}")
#         print(processed_doc.content)

# result = test()
# print(result)