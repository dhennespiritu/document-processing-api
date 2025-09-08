"""Embedding generation utilities for the ReAct agent.

This module provides functions to generate embeddings using various providers
including OpenAI, Azure OpenAI, and Anthropic (via sentence-transformers fallback).
Uses the existing load_chat_model function from utils.py for Azure OpenAI connections.
"""

import os
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from utils import load_chat_model

# Load environment variables
current_file = Path(__file__)
current_dir = current_file.parent
project_root = current_dir.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

embedding_model = os.getenv('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT')

def get_azure_openai_config():
    """Get Azure OpenAI configuration using the same logic as load_chat_model."""
    try:
        # Try to use load_chat_model to validate the configuration works
        test_model = load_chat_model("azure_openai/gpt-4o")  # Use a known chat model to test config
        print("✓ Azure OpenAI configuration validated via load_chat_model")
        
        # Return the configuration needed for embedding client
        return {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
        }
    except Exception as e:
        print(f"⚠ Could not validate config via load_chat_model: {e}")
        # Fallback to direct environment variables
        return {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
        }


class EmbeddingGenerator:
    """Generate embeddings using different providers."""
    
    def __init__(self, provider: str = "azure_openai", model: str = embedding_model):
        """Initialize the embedding generator.
        
        Args:
            provider: The embedding provider ("openai", "azure_openai", or "sentence_transformers")
            model: The model to use for embeddings
        """
        self.provider = provider
        self.model = model
        self.client = None
        
        # Initialize based on provider
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
                
        elif self.provider == "azure_openai":
            try:
                # Get Azure OpenAI configuration (optionally validated via load_chat_model)
                config = get_azure_openai_config()
                
                # Create the Azure OpenAI client specifically for embeddings
                from openai import AzureOpenAI
                self.client = AzureOpenAI(**config)
                
                
                print(f"✓ Azure OpenAI embedding client created for model: {self.model}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to setup Azure OpenAI client for embeddings: {e}")
                
        elif self.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.client = SentenceTransformer(self.model)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if self.provider in ["openai", "azure_openai"]:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
            
        elif self.provider == "sentence_transformers":
            embedding = self.client.encode(text, convert_to_tensor=False)
            return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            raise ValueError("All input texts are empty")
        
        if self.provider in ["openai", "azure_openai"]:
            response = self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts
            )
            return [data.embedding for data in response.data]
            
        elif self.provider == "sentence_transformers":
            embeddings = self.client.encode(non_empty_texts, convert_to_tensor=False)
            return embeddings.tolist()
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Convenience functions for quick usage
def generate_embedding(text: str, provider: str = "azure_openai", model: str = embedding_model) -> List[float]:
    """Generate a single embedding quickly.
    
    Args:
        text: Input text to embed
        provider: Embedding provider (defaults to azure_openai to match your setup)
        model: Model name
        
    Returns:
        Embedding vector as list of floats
    """
    generator = EmbeddingGenerator(provider=provider, model=model)
    return generator.generate_embedding(text)


def generate_embeddings(texts: List[str], provider: str = "azure_openai", model: str = embedding_model) -> List[List[float]]:
    """Generate multiple embeddings quickly.
    
    Args:
        texts: List of input texts to embed
        provider: Embedding provider (defaults to azure_openai to match your setup)
        model: Model name
        
    Returns:
        List of embedding vectors
    """
    generator = EmbeddingGenerator(provider=provider, model=model)
    return generator.generate_embeddings(texts)


def semantic_search(query: str, documents: List[str], top_k: int = 5, 
                   provider: str = "azure_openai", model: str = embedding_model) -> List[tuple]:
    """Perform semantic search on documents using embeddings.
    
    Args:
        query: Search query
        documents: List of documents to search through
        top_k: Number of top results to return
        provider: Embedding provider (defaults to azure_openai to match your setup)
        model: Model name
        
    Returns:
        List of (document_index, similarity_score, document_text) tuples
    """
    generator = EmbeddingGenerator(provider=provider, model=model)
    
    # Generate embeddings
    query_embedding = generator.generate_embedding(query)
    doc_embeddings = generator.generate_embeddings(documents)
    
    # Calculate similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = generator.cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity, documents[i]))
    
    # Sort by similarity and return top_k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Simple embedding generation using Azure OpenAI
#     text = "This is a sample text for embedding generation."
    
#     # Using Azure OpenAI (default, matching your setup)
#     try:
#         embedding = generate_embedding(text)
#         print(f"Azure OpenAI embedding dimension: {len(embedding)}")
#         print(f"First 5 values: {embedding[:5]}")
#     except Exception as e:
#         print(f"Azure OpenAI embedding failed: {e}")
    
#     # # Example 2: Semantic search with Azure OpenAI
#     # query = "machine learning algorithms"
#     # documents = [
#     #     "Deep learning is a subset of machine learning",
#     #     "Natural language processing uses various algorithms",
#     #     "Computer vision applications are growing rapidly",
#     #     "Neural networks are powerful machine learning models",
#     #     "Data science involves statistical analysis"
#     # ]
    
#     # try:
#     #     results = semantic_search(query, documents, top_k=3)
#     #     print("\nSemantic search results:")
#     #     for idx, score, doc in results:
#     #         print(f"Doc {idx}: {score:.3f} - {doc}")
#     # except Exception as e:
#     #     print(f"Semantic search failed: {e}")
    