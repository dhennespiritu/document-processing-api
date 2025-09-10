from doc_extraction import DocumentProcessor, TextCleaner
from doc_chunker import DocumentChunker
from doc_embedding import EmbeddingGenerator
from dbase_store import VectorDatabase, DatabaseConfig, VectorRecord
from doc_azure_extractor import AzureExtractor, azure_client
import logging
from pathlib import Path
import logging
import json
import asyncio

# Initialize VectorDB - Fix: instantiate properly
db_config = DatabaseConfig()  # Add parentheses
vector_db = VectorDatabase(db_config)

async def process_document(file_path: str) -> dict:
    """Process a document and store in vector database.
    
    Args:
        file_path: Path to the document to process
        
    Returns:
        Dictionary with processing results and stats
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = DocumentProcessor()
    cleaner = TextCleaner()
    chunker = DocumentChunker(max_tokens=8192)
    embedding_generator = EmbeddingGenerator()

    try:
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        processed_doc = processor.process_document(file_path)
        
        if not processed_doc.content:

            raise ValueError(f"Failed to process document: {processed_doc.errors}")
        
        source_file = processed_doc.source_file
        content = processed_doc.content
        doc_metadata = processed_doc.metadata
        
        logger.info(f"Successfully processed: {source_file}")
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Processing time: {processed_doc.processing_time:.2f} seconds")
        logger.info(f"Metadata: {doc_metadata}")
    
        # Apply advanced cleaning
        enhanced_content = cleaner.remove_headers_footers(content)
        enhanced_content = cleaner.normalize_spacing(enhanced_content)
        enhanced_content = cleaner.fix_hyphenation(enhanced_content)

        # Chunk document content
        logger.info('---Chunking document---')
        doc_chunks = chunker.chunk_document(
            text=enhanced_content,
            document_id=source_file,
            chunk_size=500,
            chunk_overlap=50
        )
        
        if not doc_chunks:
            logger.warning("No chunks were created from the document content.")
            doc_chunks = [enhanced_content]

        logger.info(f"Created {len(doc_chunks)} chunks.")

        # Get embeddings for each chunk and create records
        logger.info('---Embedding and preparing records---')
        records = []
        for i, chunk in enumerate(doc_chunks):
            # Check if chunk has a .content attribute, otherwise treat it as a string
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            doc_embeddings = embedding_generator.generate_embedding(chunk_content)
            
            # Create a VectorRecord for each chunk
            records.append(
                VectorRecord(
                    content=chunk_content,
                    embedding=doc_embeddings,
                    metadata=doc_metadata,
                    source=str(source_file),
                    chunk_index=i
                )
            )

        # Saving to vector database
        logger.info('---Saving chunks to vector database---')
        
        # Initialize vector database
        await vector_db.initialize()
        
        # Health check
        health = await vector_db.health_check()
        logger.info(f"Database health: {health}")
        
        # Insert all records
        stats = await vector_db.insert_vectors(records)
        logger.info(f"Insertion stats: {stats}")
        
        # Perform similarity search
        # For simplicity, we'll use the embedding of the first chunk for the search
        query_embedding = records[0].embedding
        results = await vector_db.similarity_search(
            json.dumps(query_embedding), 
            limit=5, 
            threshold=0.5
        )
        logger.info(f"Search results: {len(results)} documents found")
        
        # Get database statistics
        db_stats = await vector_db.get_stats()
        logger.info(f"Database stats: {db_stats}")
        
        # Optimize database
        await vector_db.optimize_database()
        
        # Return success info
        return {
            "status": "success",
            "source_file": str(source_file),
            "content_length": len(content),
            "processing_time": processed_doc.processing_time,
            "chunks_created": len(doc_chunks),
            "embedding_dimension": len(doc_embeddings) if doc_embeddings else 0,
            "insertion_stats": stats,
            "search_results_count": len(results),
            "database_stats": db_stats
        }
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise e
    finally:
        try:
            await vector_db.close()
        except Exception as e:
            logger.warning(f"Error closing database: {e}")


async def process_document_with_azure(file_path: str) -> dict:
    """Process a document with microsoft azure doc ai and store in vector database.
    
    Args:
        file_path: Path to the document to process
        
    Returns:
        Dictionary with processing results and stats
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = AzureExtractor(azure_client)
    cleaner = TextCleaner()
    chunker = DocumentChunker(max_tokens=8192)
    embedding_generator = EmbeddingGenerator()


    try:
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        processed_doc = processor.extract_with_azure(file_path)
        
        source_file = processed_doc.file_path
        content = processed_doc.content
        doc_metadata = processed_doc.meta_data


        if len(content)==0:
            raise ValueError(f"Failed to process document: {e}")
        
        logger.info(f"Successfully processed: {source_file}")
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Confidence score: {processed_doc.confidence_scores}")
        logger.info(f"Metadata: {doc_metadata}")
    
        # Apply advanced cleaning
        enhanced_content = cleaner.remove_headers_footers(content)
        enhanced_content = cleaner.normalize_spacing(enhanced_content)
        enhanced_content = cleaner.fix_hyphenation(enhanced_content)

        # Chunk document content
        logger.info('---Chunking document---')
        doc_chunks = chunker.chunk_document(
            text=enhanced_content,
            document_id=source_file,
            chunk_size=500,
            chunk_overlap=50
        )

        if not doc_chunks:
            logger.warning("No chunks were created from the document content.")
            doc_chunks = [enhanced_content]

        logger.info(f"Created {len(doc_chunks)} chunks.")

        # Get embeddings for each chunk and create records
        logger.info('---Embedding and preparing records---')
        records = []
        for i, chunk in enumerate(doc_chunks):
            # Check if chunk has a .content attribute, otherwise treat it as a string
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            doc_embeddings = embedding_generator.generate_embedding(chunk_content)
            
            # Create a VectorRecord for each chunk
            records.append(
                VectorRecord(
                    content=chunk_content,
                    embedding=doc_embeddings,
                    metadata=doc_metadata,
                    source=str(source_file),
                    chunk_index=i
                )
            )

        # Saving to vector database
        logger.info('---Saving chunks to vector database---')
        
        # Initialize vector database
        await vector_db.initialize()
        
        # Health check
        health = await vector_db.health_check()
        logger.info(f"Database health: {health}")
        
        # Insert all records
        stats = await vector_db.insert_vectors(records)
        logger.info(f"Insertion stats: {stats}")
        
        # Perform similarity search
        # For simplicity, we'll use the embedding of the first chunk for the search
        query_embedding = records[0].embedding
        results = await vector_db.similarity_search(
            json.dumps(query_embedding), 
            limit=5, 
            threshold=0.5
        )
        logger.info(f"Search results: {len(results)} documents found")
        
        # Get database statistics
        db_stats = await vector_db.get_stats()
        logger.info(f"Database stats: {db_stats}")
        
        # Optimize database
        await vector_db.optimize_database()
        
        # Return success info
        return {
            "status": "success",
            "source_file": str(source_file),
            "content_length": len(content),
            "chunks_created": len(doc_chunks),
            "embedding_dimension": len(doc_embeddings) if doc_embeddings else 0,
            "insertion_stats": stats,
            "search_results_count": len(results),
            "database_stats": db_stats
        }
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise e
    finally:
        try:
            await vector_db.close()
        except Exception as e:
            logger.warning(f"Error closing database: {e}")


# For testing
if __name__ == "__main__":
    document_path=r"C:\Users\dv146ms\Downloads\Invoice-000sample.pdf"
    asyncio.run(process_document_with_azure(document_path))
