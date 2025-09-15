"""
Fixed Document Retrieval System for Vector Database

This module provides document retrieval that works specifically with your database schema
where embeddings are stored as JSON strings, not vector types.
"""

import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from collections import Counter

from utils import load_chat_model
from doc_embedding import EmbeddingGenerator
from dbase_store import VectorDatabase, DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of search strategies available."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    ENHANCED = "enhanced"

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    max_results: int = 10
    min_similarity_threshold: float = 0.6
    search_type: SearchType = SearchType.SEMANTIC
    enable_query_enhancement: bool = True
    llm_model: str = "azure_openai/gpt-4o"  # Corrected model name
    source_filter: Optional[str] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    context_window_size: int = 3
    # New parameters for keyword and hybrid search
    keyword_weight: float = 0.3  # Weight for keyword scores in hybrid search
    semantic_weight: float = 0.7  # Weight for semantic scores in hybrid search
    min_keyword_matches: int = 1  # Minimum keyword matches required
    use_stemming: bool = True  # Use word stemming for keyword matching
    fuzzy_matching: bool = False  # Enable fuzzy keyword matching
    fuzzy_threshold: float = 0.8  # Threshold for fuzzy matching

@dataclass
class SearchResult:
    """Represents a single search result with metadata."""
    document_id: str
    content: str
    similarity_score: float
    source: str
    chunk_index: int
    metadata: Dict[str, Any]
    timestamp: datetime
    context_chunks: List[str] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    keyword_score: float = 0.0  # New field for keyword relevance
    hybrid_score: float = 0.0  # New field for combined score

    def to_dict(self, full_content: bool =False, preview_chars: int = 500) -> Dict[str,Any]:

        content_out = self.content if full_content else self.content[:preview_chars]
        if not full_content and len(self.content) > preview_chars:
            content_out += "..."

        return {
            "document_id": self.document_id,
            "content": content_out,
            "similarity_score": round(self.similarity_score,4),
            "keyword_score": round(self.keyword_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata" : self.metadata,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "context_chunks": self.context_chunks,
            "highlights": self.highlights
        }

@dataclass
class RetrievalResults:
    """Container for retrieval results and metadata."""
    query: str
    enhanced_query: Optional[str]
    results: List[SearchResult]
    total_results: int
    search_time: float
    search_type: SearchType

    def get_top_k(self, k: int) -> List[SearchResult]:
        """Get top k results."""
        return self.results[:k]

    def get_by_threshold(self, threshold: float) -> List[SearchResult]:
        """Get results above similarity threshold."""
        return [r for r in self.results if r.similarity_score >= threshold]
    
    def to_dict(self, full_content: bool = False, preview_chars: int = 500) -> Dict[str,Any]:
        return {
            "query": self.query,
            "enhanced_query": self.enhanced_query,
            "total_results": self.total_results,
            "search_time": round(self.search_time, 3),
            "search_type": self.search_type.value,
            "results": [
                r.to_dict(full_content=full_content, preview_chars=preview_chars)
                for r in self.results
            ]
        }
    
    def to_json(self, full_content: bool = False, preview_chars: int =500, indent: int = 2) -> str:
        return json.dumps(self.to_dict(full_content=full_content, preview_chars=preview_chars), indent=indent)

class QueryEnhancer:
    """Enhances user queries using LLM for better retrieval."""
    
    def __init__(self, model: str = "azure_openai/gpt-4o"):
        self.llm = load_chat_model(model)

    async def enhance_query(self, query: str) -> str:
        """Enhance user query for better semantic search."""
        try:
            from langchain_core.messages import HumanMessage
            
            enhancement_prompt = f"""
            Enhance this search query to find more relevant documents by adding synonyms and related terms.
            Keep it concise but comprehensive.

            Original query: "{query}"

            Enhanced query:"""
            
            response = await self.llm.ainvoke([HumanMessage(content=enhancement_prompt)])
            enhanced_query = response.content.strip()
            
            # Clean up the response
            if "Enhanced query:" in enhanced_query:
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
            
            logger.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}. Using original query.")
            return query

class KeywordProcessor:
    """Processes keywords for matching and scoring."""
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
            'should', 'now', 'to', 'from', 'with', 'without', 'and', 'or',
            'but', 'if', 'for', 'nor', 'not', 'no', 'yes', 'of', 'by'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Remove stop words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Apply stemming if enabled
        if self.use_stemming:
            keywords = [self._simple_stem(w) for w in keywords]
        
        return keywords
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemming by removing common suffixes."""
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'er', 'est', 'tion', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def calculate_keyword_score(self, query_keywords: List[str], document_keywords: List[str]) -> float:
        """Calculate keyword relevance score using TF-IDF-like approach."""
        if not query_keywords or not document_keywords:
            return 0.0
        
        # Count keyword frequencies
        doc_counter = Counter(document_keywords)
        query_set = set(query_keywords)
        
        # Calculate matches
        matches = sum(1 for kw in query_keywords if kw in doc_counter)
        
        # Calculate frequency-weighted score
        total_score = 0.0
        for keyword in query_keywords:
            if keyword in doc_counter:
                # TF component (normalized)
                tf = doc_counter[keyword] / len(document_keywords)
                # Simple IDF simulation (rarer keywords get higher weight)
                idf_weight = 1.0 / (1.0 + np.log(1 + doc_counter[keyword]))
                total_score += tf * idf_weight
        
        # Normalize by query length
        normalized_score = total_score / len(query_keywords) if query_keywords else 0.0
        
        # Boost score based on match percentage
        match_percentage = matches / len(query_keywords)
        final_score = normalized_score * (1 + match_percentage)
        
        return min(1.0, final_score)  # Cap at 1.0

class DocumentRetriever:
    """Main class for document retrieval from vector database."""
    
    def __init__(self, config: RetrievalConfig = None, db_config: DatabaseConfig = None):
        self.config = config or RetrievalConfig()
        self.db_config = db_config or DatabaseConfig()
        
        # Initialize components
        self.vector_db = VectorDatabase(self.db_config)
        self.embedding_generator = EmbeddingGenerator()
        self.query_enhancer = QueryEnhancer(self.config.llm_model) if self.config.enable_query_enhancement else None
        self.keyword_processor = KeywordProcessor(use_stemming=self.config.use_stemming)
        
    async def initialize(self):
        """Initialize the retrieval system."""
        await self.vector_db.initialize()
        logger.info("Document retriever initialized successfully")
    
    async def close(self):
        """Close database connections."""
        await self.vector_db.close()
        logger.info("Document retriever closed")
    
    async def search(self, query: str, **kwargs) -> RetrievalResults:
        """
        Main search method that routes to appropriate search type.
        
        Args:
            query: User search query
            **kwargs: Additional search parameters to override config
            
        Returns:
            RetrievalResults object with search results and metadata
        """
        start_time = datetime.now()
        
        # Merge kwargs with config
        search_config = self._merge_config(kwargs)
        enhanced_query = None
        
        try:
            # Enhance query if enabled
            if search_config.enable_query_enhancement and self.query_enhancer:
                enhanced_query = await self.query_enhancer.enhance_query(query)
                search_query = enhanced_query
            else:
                search_query = query
            
            # Route to appropriate search method based on search type
            if search_config.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(search_query, search_config)
            elif search_config.search_type == SearchType.KEYWORD:
                results = await self._keyword_search(search_query, search_config)
            elif search_config.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(search_query, search_config)
            else:  # ENHANCED - uses all methods with optimizations
                results = await self._enhanced_search(search_query, search_config)
            
            # Add context chunks if enabled
            if search_config.context_window_size > 0:
                results = await self._add_context_chunks(results, search_config.context_window_size)
            
            # Generate highlights
            results = self._generate_highlights(query, results)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return RetrievalResults(
                query=query,
                enhanced_query=enhanced_query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                search_type=search_config.search_type
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _keyword_search(self, query: str, config: RetrievalConfig) -> List[SearchResult]:
        """
        Perform keyword-based search using full-text search capabilities.
        """
        # Extract keywords from query
        query_keywords = self.keyword_processor.extract_keywords(query)
        logger.info(f"Extracted keywords: {query_keywords}")
        
        if not query_keywords:
            logger.warning("No keywords extracted from query")
            return []
        
        # Build search conditions
        conditions = []
        params = []
        param_count = 0
        
        # Create keyword search condition using PostgreSQL's text search
        # We'll use ILIKE for simple keyword matching
        keyword_conditions = []
        for keyword in query_keywords[:10]:  # Limit to first 10 keywords
            param_count += 1
            keyword_conditions.append(f"LOWER(content) LIKE ${param_count}")
            params.append(f'%{keyword.lower()}%')
        
        if keyword_conditions:
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # Add source filter if specified
        if config.source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(config.source_filter)
        
        # Add metadata filters
        if config.metadata_filters:
            for key, value in config.metadata_filters.items():
                key_esc = str(key).replace("'", "''")
                param_count += 1
                conditions.append(f"(metadata::jsonb ->> '{key_esc}') = ${param_count}")
                params.append(str(value))
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Query with keyword matching
        query_sql = f"""
        SELECT id, content, metadata::text AS metadata, source, chunk_index, timestamp,
               embedding::text AS embedding
        FROM documents
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT {config.max_results * 3}
        """
        
        logger.info(f"Executing keyword search query with {len(query_keywords)} keywords")
        
        async with self.vector_db.get_connection() as conn:
            rows = await conn.fetch(query_sql, *params)
            logger.info(f"Retrieved {len(rows)} documents from keyword search")
            
            results = []
            
            for row in rows:
                try:
                    # Parse metadata
                    metadata_raw = row['metadata']
                    if isinstance(metadata_raw, (bytes, bytearray, memoryview)):
                        metadata_raw = bytes(metadata_raw).decode("utf-8", "ignore")
                    
                    if isinstance(metadata_raw, str) and metadata_raw.strip():
                        try:
                            metadata = json.loads(metadata_raw)
                        except Exception:
                            metadata = {}
                    else:
                        metadata = {}
                    
                    # Extract document keywords for scoring
                    doc_keywords = self.keyword_processor.extract_keywords(row['content'])
                    
                    # Calculate keyword score
                    keyword_score = self.keyword_processor.calculate_keyword_score(
                        query_keywords, doc_keywords
                    )
                    
                    # Only include if meets minimum keyword match threshold
                    if keyword_score > 0:
                        result = SearchResult(
                            document_id=row['id'],
                            content=row['content'],
                            similarity_score=0.0,  # No semantic similarity in keyword search
                            keyword_score=keyword_score,
                            source=row['source'],
                            chunk_index=row['chunk_index'],
                            metadata=metadata,
                            timestamp=row['timestamp']
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.warning(f"Failed to process document {row.get('id', 'unknown')}: {e}")
                    continue
            
            # Sort by keyword score
            results.sort(key=lambda x: x.keyword_score, reverse=True)
            return results[:config.max_results]
    
    async def _hybrid_search(self, query: str, config: RetrievalConfig) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.
        """
        # Perform both searches in parallel
        semantic_task = self._semantic_search(query, config)
        keyword_task = self._keyword_search(query, config)
        
        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )
        
        # Create a mapping of document_id to results
        result_map: Dict[str, SearchResult] = {}
        
        # Process semantic results
        for result in semantic_results:
            result_map[result.document_id] = result
        
        # Merge keyword results
        for keyword_result in keyword_results:
            if keyword_result.document_id in result_map:
                # Document found in both searches - update scores
                existing = result_map[keyword_result.document_id]
                existing.keyword_score = keyword_result.keyword_score
            else:
                # Document only found in keyword search
                keyword_result.similarity_score = 0.0
                result_map[keyword_result.document_id] = keyword_result
        
        # Calculate hybrid scores
        for result in result_map.values():
            # Normalize scores to [0, 1] range if needed
            semantic_score = min(1.0, max(0.0, result.similarity_score))
            keyword_score = min(1.0, max(0.0, result.keyword_score))
            
            # Calculate weighted hybrid score
            result.hybrid_score = (
                config.semantic_weight * semantic_score +
                config.keyword_weight * keyword_score
            )
        
        # Convert to list and sort by hybrid score
        results = list(result_map.values())
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        logger.info(f"Hybrid search combined {len(semantic_results)} semantic and "
                   f"{len(keyword_results)} keyword results into {len(results)} unique results")
        
        return results[:config.max_results]
    
    async def _enhanced_search(self, query: str, config: RetrievalConfig) -> List[SearchResult]:
        """
        Enhanced search that intelligently combines multiple search strategies.
        This method analyzes the query to determine the best search approach.
        """
        # Analyze query characteristics
        query_length = len(query.split())
        has_technical_terms = any(term in query.lower() for term in 
                                 ['api', 'function', 'method', 'class', 'error', 'bug'])
        has_natural_language = query_length > 5
        
        # Determine optimal weights based on query analysis
        if has_technical_terms and query_length <= 3:
            # Short technical queries benefit more from keyword search
            config.keyword_weight = 0.6
            config.semantic_weight = 0.4
        elif has_natural_language:
            # Natural language queries benefit more from semantic search
            config.keyword_weight = 0.2
            config.semantic_weight = 0.8
        else:
            # Use default balanced weights
            config.keyword_weight = 0.3
            config.semantic_weight = 0.7
        
        logger.info(f"Enhanced search using weights - semantic: {config.semantic_weight}, "
                   f"keyword: {config.keyword_weight}")
        
        # Perform hybrid search with optimized weights
        results = await self._hybrid_search(query, config)
        
        # Post-process results for diversity
        results = self._ensure_result_diversity(results, max_per_source=3)
        
        return results
    
    def _ensure_result_diversity(self, results: List[SearchResult], max_per_source: int = 3) -> List[SearchResult]:
        """Ensure diversity in results by limiting documents from the same source."""
        source_counts: Dict[str, int] = {}
        diverse_results = []
        
        for result in results:
            count = source_counts.get(result.source, 0)
            if count < max_per_source:
                diverse_results.append(result)
                source_counts[result.source] = count + 1
        
        return diverse_results
    
    async def _semantic_search(self, query: str, config: RetrievalConfig) -> List[SearchResult]:
        """Perform semantic search using vector embeddings."""
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
        
        # Perform search using our custom method that handles JSON embeddings
        raw_results = await self._json_based_similarity_search(
            query_embedding=query_embedding,
            limit=config.max_results * 2,  # Get more to filter properly
            threshold=config.min_similarity_threshold,
            source_filter=config.source_filter,
            metadata_filter=config.metadata_filters
        )
        
        # Convert to SearchResult objects
        results = []
        for raw_result in raw_results:
            result = SearchResult(
                document_id=raw_result['id'],
                content=raw_result['content'],
                similarity_score=raw_result['similarity_score'],
                source=raw_result['source'],
                chunk_index=raw_result['chunk_index'],
                metadata=raw_result['metadata'],
                timestamp=raw_result['timestamp']
            )
            results.append(result)
        
        # Sort by similarity score and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:config.max_results]
    
    async def _json_based_similarity_search(self,
                                            query_embedding: List[float],
                                            limit: int = 20,
                                            threshold: float = 0.6,
                                            source_filter: Optional[str] = None,
                                            metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Similarity search that works with JSON-stored embeddings.
        This method retrieves documents and calculates similarity in Python.
        """
        # Build query conditions for filtering — avoid comparing vector column directly to a string
        conditions = ["COALESCE(embedding::text, '') <> ''"]  # <<-- safe: cast vector -> text first
        params = []
        param_count = 0

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)

        if metadata_filter:
            for key, value in metadata_filter.items():
                # embed the JSON key as a literal safely (escape single quotes)
                key_esc = str(key).replace("'", "''")
                param_count += 1
                conditions.append(f"(metadata::jsonb ->> '{key_esc}') = ${param_count}")
                params.append(str(value))

        where_clause = " AND ".join(conditions)

        # Force embedding/metadata to text on read so the driver never tries to parse to vector
        query_sql = f"""
        SELECT id, content, metadata::text AS metadata, source, chunk_index, timestamp, embedding::text AS embedding
        FROM documents
        WHERE {where_clause}
        ORDER BY timestamp DESC
        """

        logger.info(f"Executing query: {query_sql}")
        logger.info(f"With params: {params}")

        async with self.vector_db.get_connection() as conn:
            rows = await conn.fetch(query_sql, *params)
            logger.info(f"Retrieved {len(rows)} documents from database")

            results = []
            processed_count = 0

            for row in rows:
                try:
                    # Parse JSON embedding safely
                    embedding_raw = row['embedding']

                    # Normalize bytes/memoryview to str
                    if isinstance(embedding_raw, (bytes, bytearray, memoryview)):
                        embedding_raw = bytes(embedding_raw).decode("utf-8", "ignore")

                    if not embedding_raw:
                        # should be filtered out by WHERE, but guard anyway
                        continue

                    embedding_raw = embedding_raw.strip()

                    # Accept already-parsed list (some drivers may return lists)
                    if isinstance(embedding_raw, list):
                        stored_embedding = embedding_raw
                    else:
                        # Normalize some non-JSON formats (e.g. "(0.1,0.2)" or "0.1,0.2")
                        if not embedding_raw.lstrip().startswith("["):
                            cleaned = embedding_raw
                            if cleaned.startswith("(") and cleaned.endswith(")"):
                                cleaned = "[" + cleaned[1:-1] + "]"
                            elif "," in cleaned and "[" not in cleaned and "]" not in cleaned:
                                cleaned = "[" + cleaned + "]"
                            embedding_raw = cleaned

                        stored_embedding = json.loads(embedding_raw)

                    if not isinstance(stored_embedding, list) or not stored_embedding:
                        logger.warning(f"Invalid embedding format for document {row['id']}. Expected a non-empty list.")
                        continue

                    # Parse metadata safely
                    metadata_raw = row['metadata']
                    if isinstance(metadata_raw, (bytes, bytearray, memoryview)):
                        metadata_raw = bytes(metadata_raw).decode("utf-8", "ignore")

                    if isinstance(metadata_raw, str) and metadata_raw.strip():
                        try:
                            metadata = json.loads(metadata_raw)
                        except Exception:
                            metadata = {}
                    elif isinstance(metadata_raw, dict):
                        metadata = metadata_raw
                    else:
                        metadata = {}

                    # Cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    processed_count += 1

                    if similarity >= threshold:
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'metadata': metadata,
                            'source': row['source'],
                            'chunk_index': row['chunk_index'],
                            'timestamp': row['timestamp'],
                            'similarity_score': float(similarity)
                        })

                except Exception as e:
                    logger.warning(f"Failed to process embedding for document {row.get('id', 'unknown')}: {e}")
                    continue

            logger.info(f"Processed {processed_count} embeddings, found {len(results)} matches above threshold {threshold}")

            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:limit]


    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                logger.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Convert to numpy arrays for efficient calculation
            vec1_np = np.array(vec1, dtype=np.float32)
            vec2_np = np.array(vec2, dtype=np.float32)
            
            # Calculate dot product
            dot_product = np.dot(vec1_np, vec2_np)
            
            # Calculate norms
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure the result is in valid range
            similarity = max(-1.0, min(1.0, float(similarity)))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _add_context_chunks(self, results: List[SearchResult], window_size: int) -> List[SearchResult]:
        """Add surrounding chunks as context for each result."""
        for result in results:
            try:
                context_chunks = []
                
                # Get surrounding chunks from the same source
                async with self.vector_db.get_connection() as conn:
                    context_sql = """
                    SELECT content, chunk_index 
                    FROM documents 
                    WHERE source = $1 
                    AND chunk_index BETWEEN $2 AND $3
                    AND id != $4
                    ORDER BY chunk_index
                    """
                    
                    start_idx = max(0, result.chunk_index - window_size)
                    end_idx = result.chunk_index + window_size
                    
                    context_rows = await conn.fetch(
                        context_sql, 
                        result.source, 
                        start_idx, 
                        end_idx, 
                        result.document_id
                    )
                    
                    context_chunks = [row['content'] for row in context_rows]
                
                result.context_chunks = context_chunks
                
            except Exception as e:
                logger.warning(f"Failed to add context for {result.document_id}: {e}")
        
        return results
    
    def _generate_highlights(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Generate highlights from the query terms."""
        query_terms = query.lower().split()
        
        for result in results:
            try:
                highlights = []
                sentences = result.content.split('. ')
                
                for sentence in sentences[:10]:  # Check first 10 sentences
                    sentence_lower = sentence.lower()
                    if any(term in sentence_lower for term in query_terms):
                        # Truncate long sentences
                        highlight = sentence[:200] + "..." if len(sentence) > 200 else sentence
                        highlights.append(highlight)
                        
                        if len(highlights) >= 3:  # Limit to 3 highlights
                            break
                
                result.highlights = highlights
                
            except Exception as e:
                logger.warning(f"Failed to generate highlights for {result.document_id}: {e}")
        
        return results
    
    def _merge_config(self, kwargs: Dict) -> RetrievalConfig:
        """Merge kwargs with default config."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(kwargs)
        return RetrievalConfig(**config_dict)
    
    # Convenience methods
    async def search_by_source(self, query: str, source: str, **kwargs) -> RetrievalResults:
        """Search within a specific document source."""
        kwargs['source_filter'] = source
        return await self.search(query, **kwargs)
    
    async def search_keywords(self, query: str, **kwargs) -> RetrievalResults:
        """Perform keyword-only search."""
        kwargs['search_type'] = SearchType.KEYWORD
        return await self.search(query, **kwargs)
    
    async def search_hybrid(self, query: str, **kwargs) -> RetrievalResults:
        """Perform hybrid search combining semantic and keyword."""
        kwargs['search_type'] = SearchType.HYBRID
        return await self.search(query, **kwargs)
    
    async def get_similar_documents(self, document_id: str, limit: int = 5) -> List[SearchResult]:
        """Find documents similar to a given document."""
        # Get the document's content
        doc = await self.vector_db.get_document_by_id(document_id)
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        
        # Use the document content as query (first 500 chars)
        search_results = await self._semantic_search(
            doc['content'][:500],
            RetrievalConfig(max_results=limit + 1)  # +1 to exclude self
        )
        
        # Filter out the original document
        similar_docs = [r for r in search_results if r.document_id != document_id]
        return similar_docs[:limit]

# Convenience functions for quick usage
async def simple_search(query: str, max_results: int = 5, min_similarity: float = 0.5) -> RetrievalResults:
    """Simple search interface for quick queries."""
    config = RetrievalConfig(
        max_results=max_results,
        min_similarity_threshold=min_similarity,
        enable_query_enhancement=False,  # Disable for speed
        search_type=SearchType.SEMANTIC  # Default to semantic
    )
    retriever = DocumentRetriever(config)
    
    try:
        await retriever.initialize()
        results = await retriever.search(query)
        return results
    finally:
        await retriever.close()

async def keyword_search(
    query: str,
    max_results: int = 10,
    min_keyword_matches: int = 1
) -> RetrievalResults:
    """Keyword-based search for exact term matching."""
    config = RetrievalConfig(
        max_results=max_results,
        search_type=SearchType.KEYWORD,
        min_keyword_matches=min_keyword_matches,
        enable_query_enhancement=False
    )
    
    retriever = DocumentRetriever(config)
    
    try:
        await retriever.initialize()
        results = await retriever.search(query)
        return results
    finally:
        await retriever.close()

async def hybrid_search(
    query: str,
    max_results: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    min_similarity: float = 0.5,
) -> RetrievalResults:
    """Hybrid search combining semantic and keyword approaches."""
    config = RetrievalConfig(
        max_results=max_results,
        search_type=SearchType.HYBRID,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        min_similarity_threshold=min_similarity,
        enable_query_enhancement=False
    )
    
    retriever = DocumentRetriever(config)
    
    try:
        await retriever.initialize()
        results = await retriever.search(query)
        return results
    finally:
        await retriever.close()

async def enhanced_search(
    query: str,
    max_results: int = 10,
    min_similarity: float = 0.6,
    enable_context: bool = True,
    enable_enhancement: bool = True,
    search_type: SearchType = SearchType.ENHANCED
) -> RetrievalResults:
    """Enhanced search with full features."""
    config = RetrievalConfig(
        max_results=max_results,
        min_similarity_threshold=min_similarity,
        context_window_size=3 if enable_context else 0,
        enable_query_enhancement=enable_enhancement,
        search_type=search_type
    )
    
    retriever = DocumentRetriever(config)
    
    try:
        await retriever.initialize()
        results = await retriever.search(query)
        return results
    finally:
        await retriever.close()

