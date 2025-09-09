"""
Document Chunking Module
A simplified, modular approach to document chunking with dynamic strategy selection
based on document classification.
"""

import re
import tiktoken
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from utils import load_chat_model

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class DocumentType(Enum):
    """Document types supported by Microsoft Doc AI."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    FORM = "form"
    LETTER = "letter"
    RESUME = "resume"
    RESEARCH_PAPER = "research_paper"
    MANUAL = "manual"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    MARKDOWN = "markdown"
    GENERAL = "general"


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    SLIDING_WINDOW = "sliding_window"
    AI_ENHANCED = "ai_enhanced"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    source_document: str
    chunk_index: int
    total_chunks: int
    character_count: int
    token_count: int
    strategy_used: str
    document_type: Optional[str] = None
    section_title: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_document(self) -> Document:
        """Convert chunk to LangChain Document."""
        return Document(
            page_content=self.content,
            metadata=self.metadata.__dict__
        )


# ============================================================================
# BASE CHUNKER
# ============================================================================

class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))


# ============================================================================
# CHUNKER IMPLEMENTATIONS
# ============================================================================

class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""
    
    def chunk(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        return splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """Semantic chunking based on paragraph boundaries."""
    
    def chunk(self, text: str) -> List[str]:
        """Split text semantically by paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
            
            if self.count_tokens(potential_chunk) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                sentences = re.split(r'[.!?]\s+', current_chunk)
                overlap = sentences[-1] if sentences else ""
                current_chunk = overlap + "\n\n" + paragraph
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class MarkdownChunker(BaseChunker):
    """Chunker specialized for Markdown documents."""
    
    def chunk(self, text: str) -> List[str]:
        """Split markdown text preserving structure."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunker for dense overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, step_size: int = 256):
        super().__init__(chunk_size, chunk_size - step_size)
        self.step_size = step_size
    
    def chunk(self, text: str) -> List[str]:
        """Create overlapping chunks with sliding window."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for start in range(0, len(tokens), self.step_size):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_content = self.encoding.decode(chunk_tokens).strip()
            
            if chunk_content:
                chunks.append(chunk_content)
            
            if end >= len(tokens):
                break
        
        return chunks


class AIEnhancedChunker(BaseChunker):
    """AI-enhanced semantic chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.llm = load_chat_model("azure_openai/gpt-4o")
        self.base_chunker = SemanticChunker(chunk_size, chunk_overlap)
    
    def chunk(self, text: str) -> List[str]:
        """Use AI to identify optimal chunk boundaries."""
        # For simplicity, use semantic chunking as base
        # In production, you could use AI to identify better boundaries
        return self.base_chunker.chunk(text)
    
    async def enhance_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Enhance chunks with AI-generated metadata."""
        enhanced = []
        for chunk in chunks[:5]:  # Limit AI calls for performance
            try:
                prompt = f"""
                Extract 3-5 keywords from this text:
                {chunk[:500]}
                
                Return only keywords separated by commas.
                """
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                keywords = [k.strip() for k in response.content.split(',')]
                enhanced.append({"content": chunk, "keywords": keywords})
            except Exception as e:
                logger.warning(f"AI enhancement failed: {e}")
                enhanced.append({"content": chunk, "keywords": []})
        
        # Add remaining chunks without enhancement
        for chunk in chunks[5:]:
            enhanced.append({"content": chunk, "keywords": []})
        
        return enhanced


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """Select optimal chunking strategy based on document type and content."""
    
    # Mapping of document types to preferred strategies
    STRATEGY_MAP = {
        DocumentType.INVOICE: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE],
        DocumentType.RECEIPT: [ChunkingStrategy.FIXED_SIZE],
        DocumentType.CONTRACT: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.SLIDING_WINDOW],
        DocumentType.REPORT: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE],
        DocumentType.FORM: [ChunkingStrategy.FIXED_SIZE],
        DocumentType.LETTER: [ChunkingStrategy.SEMANTIC],
        DocumentType.RESUME: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE],
        DocumentType.RESEARCH_PAPER: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.AI_ENHANCED],
        DocumentType.MANUAL: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.SLIDING_WINDOW],
        DocumentType.PRESENTATION: [ChunkingStrategy.FIXED_SIZE],
        DocumentType.SPREADSHEET: [ChunkingStrategy.FIXED_SIZE],
        DocumentType.MARKDOWN: [ChunkingStrategy.MARKDOWN, ChunkingStrategy.SEMANTIC],
        DocumentType.GENERAL: [ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE],
    }
    
    @classmethod
    def select_strategies(cls, doc_type: DocumentType, content: str = "") -> List[ChunkingStrategy]:
        """Get ordered list of strategies to try for a document type."""
        strategies = cls.STRATEGY_MAP.get(doc_type, [ChunkingStrategy.SEMANTIC])
        
        # Add markdown strategy if markdown detected
        if "# " in content[:500] or "## " in content[:500]:
            if ChunkingStrategy.MARKDOWN not in strategies:
                strategies.insert(0, ChunkingStrategy.MARKDOWN)
        
        # Always have fixed_size as final fallback
        if ChunkingStrategy.FIXED_SIZE not in strategies:
            strategies.append(ChunkingStrategy.FIXED_SIZE)
        
        return strategies
    
    def classify_document_type(text: str) -> DocumentType:
        """
        Dynamically classify document type based on content analysis.
        
        Args:
            text: Document text to analyze
            
        Returns:
            DocumentType: The classified document type
        """
        if not text or not text.strip():
            return DocumentType.GENERAL
        
        # Convert to lowercase for pattern matching
        content = text.lower()
        first_500_chars = content[:500]
        
        # Markdown detection (check first)
        markdown_patterns = [
            r'^#+ ',  # Headers
            r'\*\*.*\*\*',  # Bold text
            r'\[.*\]\(.*\)',  # Links
            r'```',  # Code blocks
            r'\| .* \|'  # Tables
        ]
        if any(re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns):
            return DocumentType.MARKDOWN
        
        # Invoice patterns
        invoice_keywords = ['invoice', 'bill to', 'ship to', 'total amount', 'tax', 'subtotal', 'due date', 'invoice number']
        invoice_count = sum(1 for keyword in invoice_keywords if keyword in first_500_chars)
        if invoice_count >= 3:
            return DocumentType.INVOICE
        
        # Receipt patterns
        receipt_keywords = ['receipt', 'purchased', 'thank you', 'total:', 'cash', 'credit card', 'change']
        receipt_count = sum(1 for keyword in receipt_keywords if keyword in first_500_chars)
        if receipt_count >= 2 and ('receipt' in first_500_chars or 'thank you' in first_500_chars):
            return DocumentType.RECEIPT
        
        # Contract patterns
        contract_keywords = ['agreement', 'contract', 'party', 'terms', 'conditions', 'whereas', 'hereby', 'shall']
        contract_count = sum(1 for keyword in contract_keywords if keyword in first_500_chars)
        legal_phrases = ['terms and conditions', 'party of the first part', 'binding agreement', 'null and void']
        legal_count = sum(1 for phrase in legal_phrases if phrase in content)
        if contract_count >= 3 or legal_count >= 1:
            return DocumentType.CONTRACT
        
        # Research paper patterns
        research_keywords = ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references', 'doi:', 'journal']
        research_count = sum(1 for keyword in research_keywords if keyword in first_500_chars)
        academic_patterns = [r'\b\d{4}\b.*et al\.', r'doi:', r'journal of', r'proceedings of']
        academic_count = sum(1 for pattern in academic_patterns if re.search(pattern, content))
        if research_count >= 3 or academic_count >= 1:
            return DocumentType.RESEARCH_PAPER
        
        # Resume/CV patterns
        resume_keywords = ['experience', 'education', 'skills', 'objective', 'employment', 'qualifications', 'references']
        resume_count = sum(1 for keyword in resume_keywords if keyword in first_500_chars)
        resume_patterns = [r'\d{4}\s*-\s*\d{4}', r'\d{4}\s*-\s*present', r'@\w+\.\w+']  # Date ranges, email
        resume_pattern_count = sum(1 for pattern in resume_patterns if re.search(pattern, content))
        if resume_count >= 2 and resume_pattern_count >= 1:
            return DocumentType.RESUME
        
        # Letter patterns
        letter_keywords = ['dear', 'sincerely', 'yours truly', 'best regards', 'kind regards']
        letter_count = sum(1 for keyword in letter_keywords if keyword in content[:200] or keyword in content[-200:])
        if letter_count >= 2:
            return DocumentType.LETTER
        
        # Form patterns
        form_keywords = ['application', 'form', 'please fill', 'signature', 'date:', 'name:', 'address:']
        form_count = sum(1 for keyword in form_keywords if keyword in first_500_chars)
        form_patterns = [r'_+', r'\[\s*\]', r'☐', r'□']  # Blank lines, checkboxes
        form_pattern_count = sum(1 for pattern in form_patterns if re.search(pattern, content))
        if form_count >= 2 or form_pattern_count >= 3:
            return DocumentType.FORM
        
        # Manual patterns
        manual_keywords = ['manual', 'instructions', 'step', 'procedure', 'warning', 'caution', 'installation']
        manual_count = sum(1 for keyword in manual_keywords if keyword in first_500_chars)
        numbered_steps = len(re.findall(r'^\d+\.', content, re.MULTILINE))
        if manual_count >= 2 or numbered_steps >= 3:
            return DocumentType.MANUAL
        
        # Presentation patterns
        presentation_keywords = ['slide', 'presentation', 'agenda', 'overview', 'next slide']
        presentation_count = sum(1 for keyword in presentation_keywords if keyword in first_500_chars)
        if presentation_count >= 1:
            return DocumentType.PRESENTATION
        
        # Report patterns (check after more specific types)
        report_keywords = ['report', 'summary', 'analysis', 'findings', 'recommendations', 'executive summary']
        report_count = sum(1 for keyword in report_keywords if keyword in first_500_chars)
        if report_count >= 2:
            return DocumentType.REPORT
        
        # Spreadsheet patterns (for text exports)
        spreadsheet_indicators = content.count('\t') > 10 or content.count(',') > content.count(' ') / 4
        if spreadsheet_indicators:
            return DocumentType.SPREADSHEET
        
        # Default fallback
        return DocumentType.GENERAL


# ============================================================================
# DOCUMENT CHUNKER (MAIN CLASS)
# ============================================================================

class DocumentChunker:
    """Main document chunking manager with dynamic strategy selection."""
    
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.SEMANTIC: SemanticChunker,
            ChunkingStrategy.MARKDOWN: MarkdownChunker,
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
            ChunkingStrategy.AI_ENHANCED: AIEnhancedChunker,
        }
    
    def chunk_document(
        self,
        text: str,
        doc_type: Optional[DocumentType] = None,
        document_id: str = "unknown",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[TextChunk]:
        """
        Chunk a document using appropriate strategies.
        
        Args:
            text: Document text to chunk
            doc_type: Type of document (from Microsoft Doc AI classification)
            document_id: Unique identifier for the document
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        if doc_type is None:
            doc_type = StrategySelector.classify_document_type(text)

        logger.info(f"Document Type: {doc_type}")
        
        # Adjust chunk size based on max tokens
        effective_chunk_size = min(chunk_size, self.max_tokens - 1000)
        
        # Get strategies to try
        strategies = StrategySelector.select_strategies(doc_type, text)
        
        # Try each strategy
        all_chunks = []
        remaining_text = text
        
        for strategy in strategies:
            if not remaining_text.strip():
                break
            
            # Get chunker
            chunker_class = self.chunkers.get(strategy)
            if not chunker_class:
                continue
            
            # Initialize chunker with appropriate parameters
            if strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunker = chunker_class(
                    chunk_size=min(512, effective_chunk_size),
                    step_size=min(256, effective_chunk_size // 2)
                )
            else:
                chunker = chunker_class(effective_chunk_size, chunk_overlap)
            
            try:
                # Chunk the remaining text
                raw_chunks = chunker.chunk(remaining_text)
                
                # Process chunks
                strategy_chunks = []
                reconstructed = ""
                
                for chunk_content in raw_chunks:
                    if not chunk_content.strip():
                        continue
                    
                    # Ensure chunk is within token limits
                    if chunker.count_tokens(chunk_content) > self.max_tokens:
                        # Truncate if necessary
                        tokens = self.encoding.encode(chunk_content)
                        chunk_content = self.encoding.decode(tokens[:self.max_tokens - 100])
                    
                    strategy_chunks.append(chunk_content)
                    reconstructed += chunk_content + " "
                
                # Check coverage
                original_tokens = set(remaining_text.lower().split())
                reconstructed_tokens = set(reconstructed.lower().split())
                coverage = len(original_tokens & reconstructed_tokens) / len(original_tokens) if original_tokens else 0
                
                if coverage > 0.9:  # Good coverage
                    # Create TextChunk objects
                    for i, content in enumerate(strategy_chunks):
                        metadata = ChunkMetadata(
                            chunk_id=f"{document_id}_chunk_{len(all_chunks) + i}",
                            source_document=document_id,
                            chunk_index=len(all_chunks) + i,
                            total_chunks=0,  # Will update later
                            character_count=len(content),
                            token_count=chunker.count_tokens(content),
                            strategy_used=strategy.value,
                            document_type=doc_type.value
                        )
                        all_chunks.append(TextChunk(content=content, metadata=metadata))
                    
                    remaining_text = ""  # All text processed
                    logger.info(f"Successfully chunked with {strategy.value} strategy")
                    break
                else:
                    # Try to identify what was missed
                    missed_tokens = original_tokens - reconstructed_tokens
                    if len(missed_tokens) > 10:  # Significant content missed
                        # Find the missing content
                        remaining_text = self._extract_missing_content(remaining_text, reconstructed)
                        logger.warning(f"{strategy.value} missed content, trying fallback for remainder")
                        
                        # Add what we got
                        for i, content in enumerate(strategy_chunks):
                            metadata = ChunkMetadata(
                                chunk_id=f"{document_id}_chunk_{len(all_chunks) + i}",
                                source_document=document_id,
                                chunk_index=len(all_chunks) + i,
                                total_chunks=0,
                                character_count=len(content),
                                token_count=chunker.count_tokens(content),
                                strategy_used=strategy.value,
                                document_type=doc_type.value
                            )
                            all_chunks.append(TextChunk(content=content, metadata=metadata))
                    else:
                        remaining_text = ""
                        break
                        
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                continue
        
        # Handle any remaining text with fixed-size fallback
        if remaining_text.strip():
            logger.info("Processing remaining text with fixed-size fallback")
            fallback_chunker = FixedSizeChunker(effective_chunk_size, chunk_overlap)
            fallback_chunks = fallback_chunker.chunk(remaining_text)
            
            for i, content in enumerate(fallback_chunks):
                if content.strip():
                    metadata = ChunkMetadata(
                        chunk_id=f"{document_id}_chunk_{len(all_chunks) + i}",
                        source_document=document_id,
                        chunk_index=len(all_chunks) + i,
                        total_chunks=0,
                        character_count=len(content),
                        token_count=fallback_chunker.count_tokens(content),
                        strategy_used="fallback_fixed_size",
                        document_type=doc_type.value
                    )
                    all_chunks.append(TextChunk(content=content, metadata=metadata))
        
        # Update total chunks count
        for chunk in all_chunks:
            chunk.metadata.total_chunks = len(all_chunks)
        
        return all_chunks
    
    def _extract_missing_content(self, original: str, reconstructed: str) -> str:
        """Extract content that was missed during chunking."""
        # Simple approach: find the last matching sentence and return everything after
        sentences = re.split(r'[.!?]\s+', original)
        reconstructed_lower = reconstructed.lower()
        
        last_matched_idx = -1
        for i, sentence in enumerate(sentences):
            if sentence.lower() in reconstructed_lower:
                last_matched_idx = i
        
        if last_matched_idx >= 0 and last_matched_idx < len(sentences) - 1:
            # Return everything after the last matched sentence
            remaining_sentences = sentences[last_matched_idx + 1:]
            return '. '.join(remaining_sentences)
        
        # If no clear boundary found, return the last 20% of the original
        split_point = int(len(original) * 0.8)
        return original[split_point:]
    
    def chunk_documents(
        self,
        documents: List[Document],
        doc_types: Optional[List[DocumentType]] = None,
        **kwargs
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of LangChain Document objects
            doc_types: List of document types (parallel to documents)
            **kwargs: Additional arguments for chunking
        
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        for i, doc in enumerate(documents):
            if not doc.page_content.strip():
                continue
            
            doc_type = doc_types[i] if doc_types and i < len(doc_types) else DocumentType.GENERAL
            document_id = doc.metadata.get('source', f'doc_{i}')
            
            chunks = self.chunk_document(
                text=doc.page_content,
                doc_type=doc_type,
                document_id=document_id,
                **kwargs
            )
            
            # Add original document metadata to chunks
            for chunk in chunks:
                chunk.metadata.__dict__.update(doc.metadata)
            
            all_chunks.extend(chunks)
        
        return all_chunks
