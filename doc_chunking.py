
import re
import tiktoken
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import asyncio
import logging
from pathlib import Path
from utils import load_chat_model

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    AI_POWERED = "ai_powered"
    MARKDOWN = "markdown"
    CODE = "code"
    SLIDING_WINDOW = "sliding_window"
    PARAGRAPH = "paragraph"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    source_document: str
    chunk_index: int
    total_chunks: int
    character_count: int
    token_count: int
    chunk_type: str
    semantic_score: Optional[float] = None
    keywords: List[str] = field(default_factory=list)
    section_title: Optional[str] = None


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


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split text into chunks."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _create_chunk_metadata(
        self, 
        chunk_content: str, 
        document_id: str, 
        chunk_index: int, 
        total_chunks: int,
        chunk_type: str = "text",
        **kwargs
    ) -> ChunkMetadata:
        """Create metadata for a chunk."""
        return ChunkMetadata(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            source_document=document_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            character_count=len(chunk_content),
            token_count=self.count_tokens(chunk_content),
            chunk_type=chunk_type,
            **kwargs
        )


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split text into fixed-size chunks."""
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        
        for i, chunk_content in enumerate(raw_chunks):
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, i, len(raw_chunks), "fixed_size"
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class TokenBasedChunker(BaseChunker):
    """Token-based chunking for precise token control."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split text based on token count."""
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        
        for i, chunk_content in enumerate(raw_chunks):
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, i, len(raw_chunks), "token_based"
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on paragraph and sentence boundaries."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split text semantically by paragraphs and sentences."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if (len(current_chunk) + len(paragraph) > self.chunk_size and 
                current_chunk.strip()):
                
                chunks.append(current_chunk.strip())
                # Keep overlap from end of previous chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Convert to TextChunk objects
        text_chunks = []
        for i, chunk_content in enumerate(chunks):
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, i, len(chunks), "semantic"
            )
            text_chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return text_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find sentence boundary for clean overlap
        overlap_start = len(text) - self.chunk_overlap
        sentences = re.split(r'[.!?]\s+', text[overlap_start:])
        
        if len(sentences) > 1:
            return sentences[-2] + ". " if len(sentences) > 2 else sentences[0]
        else:
            return text[-self.chunk_overlap:]


class MarkdownChunker(BaseChunker):
    """Chunker specialized for Markdown documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split markdown text preserving structure."""
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        
        for i, chunk_content in enumerate(raw_chunks):
            # Extract section title if present
            section_title = self._extract_section_title(chunk_content)
            
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, i, len(raw_chunks), "markdown",
                section_title=section_title
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks
    
    def _extract_section_title(self, chunk: str) -> Optional[str]:
        """Extract the main heading from a markdown chunk."""
        lines = chunk.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                return line.strip().lstrip('# ').strip()
        return None


class CodeChunker(BaseChunker):
    """Chunker specialized for code documents."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Split code text preserving function/class boundaries."""
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        
        for i, chunk_content in enumerate(raw_chunks):
            metadata = self._create_chunk_metadata(
                chunk_content, document_id, i, len(raw_chunks), "code"
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class AIPoweredChunker(BaseChunker):
    """AI-powered chunking using Azure OpenAI for intelligent content analysis."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, model: str = "gpt-4o"):
        super().__init__(chunk_size, chunk_overlap)
        self.model = model
        self.llm = self._init_azure_openai()
    
    def _init_azure_openai(self):
        """Initialize Azure OpenAI client."""
        return load_chat_model("azure_openai/gpt-4o")
    
    async def chunk_text_async(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Asynchronously split text using AI analysis."""
        # First, get initial chunks using semantic chunking
        semantic_chunker = SemanticChunker(self.chunk_size, self.chunk_overlap)
        initial_chunks = semantic_chunker.chunk_text(text, document_id)
        
        # Enhance chunks with AI analysis
        enhanced_chunks = []
        for chunk in initial_chunks:
            enhanced_chunk = await self._enhance_chunk_with_ai(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Synchronous wrapper for async chunking."""
        return asyncio.run(self.chunk_text_async(text, document_id))
    
    async def _enhance_chunk_with_ai(self, chunk: TextChunk) -> TextChunk:
        """Enhance chunk metadata using AI analysis."""
        try:
            analysis_prompt = f"""
            Analyze this text chunk and provide:
            1. Key keywords (max 5)
            2. A semantic coherence score (0-1)
            3. The main topic/theme
            
            Text: {chunk.content[:500]}...
            
            Respond in this format:
            Keywords: keyword1, keyword2, keyword3
            Coherence: 0.85
            Theme: main topic description
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis = self._parse_ai_analysis(response.content)
            
            # Update chunk metadata
            chunk.metadata.keywords = analysis.get('keywords', [])
            chunk.metadata.semantic_score = analysis.get('coherence', 0.5)
            chunk.metadata.chunk_type = "ai_enhanced"
            
        except Exception as e:
            logger.warning(f"AI analysis failed for chunk {chunk.metadata.chunk_id}: {e}")
        
        return chunk
    
    def _parse_ai_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse AI analysis response."""
        result = {}
        
        for line in analysis_text.split('\n'):
            if line.startswith('Keywords:'):
                keywords = [k.strip() for k in line.replace('Keywords:', '').split(',')]
                result['keywords'] = keywords
            elif line.startswith('Coherence:'):
                try:
                    coherence = float(line.replace('Coherence:', '').strip())
                    result['coherence'] = coherence
                except ValueError:
                    result['coherence'] = 0.5
        
        return result


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunker for dense overlapping chunks."""
    
    def __init__(self, chunk_size: int = 512, step_size: int = 256):
        super().__init__(chunk_size, chunk_size - step_size)
        self.step_size = step_size
    
    def chunk_text(self, text: str, document_id: str = "unknown") -> List[TextChunk]:
        """Create overlapping chunks with sliding window."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                # Estimate total chunks
                total_chunks = (len(text) - 1) // self.step_size + 1
                
                metadata = self._create_chunk_metadata(
                    chunk_content, document_id, chunk_index, total_chunks, "sliding_window"
                )
                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1
            
            start += self.step_size
            
            # Break if we've reached the end
            if end >= len(text):
                break
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks


class ChunkingManager:
    """Manager class for different chunking strategies."""
    
    def __init__(self):
        self.chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.SEMANTIC: SemanticChunker,
            ChunkingStrategy.AI_POWERED: AIPoweredChunker,
            ChunkingStrategy.MARKDOWN: MarkdownChunker,
            ChunkingStrategy.CODE: CodeChunker,
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
        }
    
    def chunk_text(
        self, 
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        document_id: str = "unknown",
        **kwargs
    ) -> List[TextChunk]:
        """Chunk text using the specified strategy."""
        chunker_class = self.chunkers.get(strategy)
        if not chunker_class:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        chunker = chunker_class(**kwargs)
        return chunker.chunk_text(text, document_id)
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        **kwargs
    ) -> List[TextChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for i, doc in enumerate(documents):
            document_id = doc.metadata.get('source', f'doc_{i}')
            chunks = self.chunk_text(doc.page_content, strategy, document_id, **kwargs)
            
            # Add original document metadata to chunks
            for chunk in chunks:
                chunk.metadata.__dict__.update(doc.metadata)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_optimal_strategy(self, text: str, content_type: str = "text") -> ChunkingStrategy:
        """Suggest optimal chunking strategy based on content."""
        if content_type.lower() == "markdown" or "# " in text[:200]:
            return ChunkingStrategy.MARKDOWN
        elif content_type.lower() in ["python", "code"] or "def " in text[:200]:
            return ChunkingStrategy.CODE
        elif len(text) > 5000:  # Large documents benefit from AI analysis
            return ChunkingStrategy.AI_POWERED
        else:
            return ChunkingStrategy.SEMANTIC


# Example usage and testing
# async def main():
#     """Example usage of the chunking system."""
#     manager = ChunkingManager()
    
#     sample_text = """
#     # Introduction to Machine Learning
    
#     Machine learning is a subset of artificial intelligence that focuses on algorithms
#     that can learn and make decisions from data without being explicitly programmed.
    
#     ## Types of Machine Learning
    
#     There are three main types of machine learning:
    
#     1. Supervised Learning: Uses labeled data to train models
#     2. Unsupervised Learning: Finds patterns in unlabeled data  
#     3. Reinforcement Learning: Learns through interaction with an environment
    
#     ## Applications
    
#     Machine learning has numerous applications across various industries including
#     healthcare, finance, transportation, and entertainment.
#     """
    
#     chunks = manager.chunk_text(sample_text, ChunkingStrategy.AI_POWERED, "sample_doc")
#     print(chunks


# if __name__ == "__main__":
#     asyncio.run(main())