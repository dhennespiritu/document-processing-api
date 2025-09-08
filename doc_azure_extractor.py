import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from azure.core.exceptions import AzureError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import os
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

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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


class AzureExtractor:

    def __init__(self, azure_client):
        self.azure_client = azure_client
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

    def detect_document_type(self, file_path: Path) -> Dict[str, Any]:
        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        doc_info = {
            'file_extension': file_ext,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'document_category': self.document_types.get(file_ext, 'unknown')
        }

        # Recommend Azure model
        if doc_info['document_category'] == 'image':
            doc_info['recommended_model'] = 'prebuilt-read'
        elif file_ext == '.pdf':
            doc_info['recommended_model'] = 'prebuilt-layout'
        else:
            doc_info['recommended_model'] = 'prebuilt-document'

        table_likely_formats = {'.pdf', '.docx', '.doc'}
        doc_info['likely_has_tables'] = file_ext in table_likely_formats

        return doc_info

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

    def extract_with_azure(self, file_path: Path) -> AzureExtractionResult:
        if not self.azure_client:
            raise ValueError("Azure client not initialized")

        try:
            doc_info = self.detect_document_type(file_path)
            logger.info(f"Document info: {doc_info}")

            content_type = self.get_content_type(doc_info['file_extension'])

            with open(file_path, "rb") as file:
                poller = self.azure_client.begin_analyze_document(
                    model_id=doc_info['recommended_model'],
                    body=file,
                    content_type=content_type
                )
                result = poller.result()

            # 1. Extract text
            content_parts = []
            if result.content:
                content_parts.append(result.content)
            if result.paragraphs:
                for p in result.paragraphs:
                    if p.content and p.content.strip() and p.content not in content_parts:
                        content_parts.append(p.content)
            extracted_text = "\n".join(content_parts)

            # 2. Tables and key-values
            tables = self.extract_tables(result)
            kv_pairs = self.extract_key_value_pairs(result)
            confidence_scores = self.calculate_confidence_scores(result)

            # 3. Build a single string with tables and key-values for chunking
            full_content_for_chunking = extracted_text  # start with main text

            if tables:
                for i, table in enumerate(tables):
                    full_content_for_chunking += f"\n[TABLE {i+1}: {table['row_count']}x{table['column_count']}]"
                    if table.get('headers'):
                        full_content_for_chunking += "\nHeaders: " + ", ".join(table['headers'])
                    for row in table.get('data_rows', []):
                        full_content_for_chunking += "\n" + "\t".join(row)

            if kv_pairs:
                full_content_for_chunking += "\n\nKEY-VALUE PAIRS:\n" + "\n".join([f"{k}: {v}" for k, v in kv_pairs.items()])


            # 4. Build metadata dict (keeps the original structure you used earlier)
            metadata = {
                'document_info': doc_info,
                'extraction_method': 'azure_document_intelligence',
                'azure_model_used': doc_info['recommended_model'],
                'pages_processed': len(result.pages) if getattr(result, "pages", None) else 0,
                'paragraphs_found': len(result.paragraphs) if getattr(result, "paragraphs", None) else 0,
                'text_length': len(extracted_text),
                'tables_found': len(tables),
                'key_value_pairs_found': len(kv_pairs),
            }


            # 5. Build data class result

            result_obj = AzureExtractionResult(
                file_path=str(file_path),
                content=full_content_for_chunking,
                document_info=doc_info,
                extraction_method="azure_document_intelligence",
                azure_model_used=doc_info['recommended_model'],
                pages_processed=len(result.pages) if result.pages else 0,
                paragraphs_found=len(result.paragraphs) if result.paragraphs else 0,
                text_length=len(extracted_text),
                tables_found=len(tables),
                tables_data=tables,
                key_value_pairs_found=len(kv_pairs),
                key_value_pairs=kv_pairs,
                confidence_scores=confidence_scores,
                meta_data=metadata
        )

            logger.info(f"Azure extraction complete: {len(full_content_for_chunking)} chars, "
                        f"{len(tables)} tables, {len(kv_pairs)} key-value pairs")

            return result_obj

        except AzureError as e:
            logger.error(f"Azure API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected extraction error: {e}")
            raise



# # For testing
def test():
    extractor = AzureExtractor(azure_client)

    processed_doc = extractor.extract_with_azure(
        Path(r"C:\Users\dv146ms\Downloads\Invoice-000sample.pdf").resolve()
    )

    print(processed_doc.content)
    print(processed_doc.confidence_scores)
    print(processed_doc.meta_data)

if __name__ == "__main__":
    result = test()
    print("Extraction complete.")
