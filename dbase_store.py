"""
Database Operations Module for Vector Embeddings Storage and Retrieval

This module provides comprehensive database operations for storing and retrieving
vector embeddings using PostgreSQL with pgvector extension.

Features:
- Async connection pooling with asyncpg
- Batch insertions for performance
- Vector similarity search
- Duplicate detection
- Automatic indexing with HNSW and IVFFlat
- Connection health monitoring
- Transaction management
"""

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import asyncpg
import numpy as np
from asyncpg import Pool, Connection
from asyncpg.exceptions import DuplicateColumnError, DuplicateTableError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Vector index types supported by pgvector."""
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DATABASE", "")
    username: str = os.getenv("POSTGRES_USERNAME", "")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    # Connection pool settings
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: float = 30.0
    
    # Performance settings
    batch_size: int = 1000
    vector_dimension: int = 1536  # Default for OpenAI embeddings
    
    # Index settings
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    ivfflat_lists: int = 100


@dataclass
class VectorRecord:
    """Represents a vector record to be stored in the database."""
    id: Optional[str] = None
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    chunk_index: int = 0
    timestamp: Optional[datetime] = None
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
        
        if not self.id:
            self.id = f"{self.source}_{self.chunk_index}_{self.content_hash[:8]}"


class VectorDatabase:
    """
    Async PostgreSQL database operations for vector embeddings.
    
    Handles connection pooling, batch operations, indexing, and similarity search.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[Pool] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the database connection pool and create tables."""
        if self._initialized:
            return
            
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            # Initialize database schema
            await self._create_schema()
            await self._create_indexes()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager for database connections."""
        if not self.pool:
            await self.initialize()
        
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def _create_schema(self) -> None:
        """Create the database schema with pgvector extension."""
        async with self.get_connection() as conn:
            try:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create main documents table
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({dimension}),
                    metadata JSONB,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    content_hash TEXT NOT NULL,
                    UNIQUE(content_hash)
                );
                """.format(dimension=self.config.vector_dimension)
                
                await conn.execute(create_table_sql)
                
                # Create similarity search function
                similarity_function = """
                CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
                RETURNS float AS $$
                BEGIN
                    RETURN 1 - (a <=> b);
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """
                
                await conn.execute(similarity_function)
                
                logger.info("Database schema created successfully")
                
            except DuplicateTableError:
                logger.info("Tables already exist")
            except Exception as e:
                logger.error(f"Error creating schema: {e}")
                raise
    
    async def _create_indexes(self) -> None:
        """Create vector indexes for optimal performance."""
        async with self.get_connection() as conn:
            try:
                # Create HNSW index for fast similarity search
                hnsw_index = f"""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
                ON documents USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction});
                """
                await conn.execute(hnsw_index)
                
                # Create IVFFlat index as alternative
                ivfflat_index = f"""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self.config.ivfflat_lists});
                """
                await conn.execute(ivfflat_index)
                
                # Create indexes on commonly queried fields
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);")
                
                logger.info("Database indexes created successfully")
                
            except Exception as e:
                logger.error(f"Error creating indexes: {e}")
                raise
    
    async def insert_vectors(self, records: List[VectorRecord], 
                           ignore_duplicates: bool = True) -> Dict[str, int]:
        """
        Insert vector records in batches with duplicate detection.
        
        Args:
            records: List of VectorRecord objects to insert
            ignore_duplicates: If True, skip duplicate records based on content_hash
            
        Returns:
            Dictionary with insertion statistics
        """
        if not records:
            return {"inserted": 0, "skipped": 0, "errors": 0}
        
        stats = {"inserted": 0, "skipped": 0, "errors": 0}
        
        # Process in batches
        for i in range(0, len(records), self.config.batch_size):
            batch = records[i:i + self.config.batch_size]
            batch_stats = await self._insert_batch(batch, ignore_duplicates)
            
            stats["inserted"] += batch_stats["inserted"]
            stats["skipped"] += batch_stats["skipped"]
            stats["errors"] += batch_stats["errors"]
        
        logger.info(f"Insertion complete. Stats: {stats}")
        return stats
    
    async def _insert_batch(self, records: List[VectorRecord], 
                           ignore_duplicates: bool) -> Dict[str, int]:
        """Insert a single batch of records."""
        stats = {"inserted": 0, "skipped": 0, "errors": 0}
        
        if ignore_duplicates:
            # Check for existing records first
            existing_hashes = await self._get_existing_hashes(
                [record.content_hash for record in records]
            )
            records = [r for r in records if r.content_hash not in existing_hashes]
            stats["skipped"] = len(existing_hashes)
        
        if not records:
            return stats
        
        insert_sql = """
        INSERT INTO documents (id, content, embedding, metadata, source, chunk_index, timestamp, content_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (content_hash) DO NOTHING
        """
        
        try:
            async with self.transaction() as conn:
                for record in records:
                    try:
                        await conn.execute(
                            insert_sql,
                            record.id,
                            record.content,
                            json.dumps(record.embedding),
                            json.dumps(record.metadata),
                            record.source,
                            record.chunk_index,
                            record.timestamp,
                            record.content_hash
                        )
                        stats["inserted"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error inserting record {record.id}: {e}")
                        stats["errors"] += 1
                        
        except Exception as e:
            logger.error(f"Transaction error in batch insert: {e}")
            stats["errors"] += len(records)
        
        return stats
    
    async def _get_existing_hashes(self, hashes: List[str]) -> set:
        """Get existing content hashes from the database."""
        if not hashes:
            return set()
        
        async with self.get_connection() as conn:
            query = "SELECT content_hash FROM documents WHERE content_hash = ANY($1)"
            rows = await conn.fetch(query, hashes)
            return {row['content_hash'] for row in rows}
    
    async def similarity_search(self, 
                              query_embedding: List[float],
                              limit: int = 10,
                              threshold: float = 0.7,
                              source_filter: Optional[str] = None,
                              metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search using vector embeddings.
        
        Args:
            query_embedding: The query vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            source_filter: Filter by document source
            metadata_filter: Filter by metadata fields
            
        Returns:
            List of similar documents with scores
        """
        # Build query conditions
        conditions = ["cosine_similarity(embedding, $1) >= $2"]
        params = [query_embedding, threshold]
        param_count = 2
        
        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                param_count += 1
                conditions.append(f"metadata ->> '{key}' = ${param_count}")
                params.append(str(value))
        
        where_clause = " AND ".join(conditions)
        
        query_sql = f"""
        SELECT id, content, metadata, source, chunk_index, timestamp,
               cosine_similarity(embedding, $1) as similarity_score
        FROM documents
        WHERE {where_clause}
        ORDER BY similarity_score DESC
        LIMIT {limit}
        """
        
        async with self.get_connection() as conn:
            rows = await conn.fetch(query_sql, *params)
            
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'source': row['source'],
                    'chunk_index': row['chunk_index'],
                    'timestamp': row['timestamp'],
                    'similarity_score': float(row['similarity_score'])
                })
            
            return results
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        query_sql = """
        SELECT id, content, metadata, source, chunk_index, timestamp
        FROM documents
        WHERE id = $1
        """
        
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query_sql, doc_id)
            
            if row:
                return {
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'source': row['source'],
                    'chunk_index': row['chunk_index'],
                    'timestamp': row['timestamp']
                }
            return None
    
    async def delete_by_source(self, source: str) -> int:
        """Delete all documents from a specific source."""
        delete_sql = "DELETE FROM documents WHERE source = $1"
        
        async with self.get_connection() as conn:
            result = await conn.execute(delete_sql, source)
            # Extract number of deleted rows from result
            deleted_count = int(result.split()[-1])
            logger.info(f"Deleted {deleted_count} documents from source: {source}")
            return deleted_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats_sql = """
        SELECT 
            COUNT(*) as total_documents,
            COUNT(DISTINCT source) as unique_sources,
            AVG(LENGTH(content)) as avg_content_length,
            MAX(timestamp) as latest_document,
            MIN(timestamp) as earliest_document
        FROM documents
        """
        
        async with self.get_connection() as conn:
            row = await conn.fetchrow(stats_sql)
            
            return {
                'total_documents': row['total_documents'],
                'unique_sources': row['unique_sources'],
                'avg_content_length': float(row['avg_content_length']) if row['avg_content_length'] else 0,
                'latest_document': row['latest_document'],
                'earliest_document': row['earliest_document']
            }
    
    async def optimize_database(self) -> None:
        """Perform database maintenance operations."""
        async with self.get_connection() as conn:
            try:
                # Update table statistics
                await conn.execute("ANALYZE documents;")
                
                # Vacuum to reclaim space
                await conn.execute("VACUUM documents;")
                
                logger.info("Database optimization completed")
                
            except Exception as e:
                logger.error(f"Error during database optimization: {e}")
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database connection."""
        try:
            async with self.get_connection() as conn:
                # Test basic query
                result = await conn.fetchrow("SELECT 1 as test, NOW() as timestamp")
                
                # Check pgvector extension
                vector_check = await conn.fetchrow(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as has_vector"
                )
                
                return {
                    'status': 'healthy',
                    'timestamp': result['timestamp'],
                    'pgvector_enabled': vector_check['has_vector'],
                    'pool_size': self.pool.get_size() if self.pool else 0,
                    'pool_free': len(self.pool._queue._queue) if self.pool else 0
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
