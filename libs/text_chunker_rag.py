"""
Knowledge Graph RAG Module

This module provides functionality for overlapping text chunking and 
vector database integration for knowledge graph data. It supports
retrieval-augmented generation (RAG) alongside knowledge graph triplets.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import chromadb
from chromadb.config import Settings
from fuzzywuzzy import fuzz
import numpy as np
from vllm import LLM
from libs.query_llm import DeepSeekChat
from libs.kg_triplet_generator import KnowledgeGraphTripletGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class KnowledgeGraphRAG:
    """
    A class for chunking text with overlap and adding it to a vector database
    to support retrieval-augmented generation alongside knowledge graph triplets.
    """
    
    def __init__(
        self,
        db_path: str = "kg_rag_db",
        collection_name: str = "kg_chunks",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "intfloat/e5-mistral-7b-instruct",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Knowledge Graph RAG system.
        
        Args:
            db_path: Path to the vector database directory
            collection_name: Name of the collection in the vector database
            chunk_size: Maximum size of each text chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            embedding_model: Model to use for generating embeddings
            system_prompt: Optional custom system prompt for the KG triplet generator
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Initialize DeepSeek client for generation tasks
        self.deepseek_client = DeepSeekChat()
        
        # Initialize the knowledge graph triplet generator
        self.kg_generator = KnowledgeGraphTripletGenerator(system_prompt=system_prompt)
        
        # Initialize vLLM for embeddings generation
        try:
            logger.info(f"Initializing vLLM embedding model: {embedding_model}")
            self.embedding_model = LLM(
                model=embedding_model,
                task="embed",
                enforce_eager=True,
                dtype="float16",  # Add this to ensure proper initialization
                gpu_memory_utilization=0.8  # Adjust memory usage to avoid OOM
            )
            # Try to set forward context right away
            if hasattr(self.embedding_model, 'set_forward_context'):
                try:
                    self.embedding_model.set_forward_context()
                    logger.info("Forward context set successfully")
                except Exception as context_err:
                    logger.warning(f"Could not set forward context: {context_err}")
            
            logger.info("vLLM embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM embedding model: {e}")
            logger.warning("Using a fallback embedding mechanism")
            # Create a dummy embedding model attribute to indicate we need to use fallback
            self.embedding_model = None
        
        # Initialize the vector database
        self._initialize_vector_db()
    
    def _initialize_vector_db(self) -> None:
        """Initialize the vector database connection with better error handling."""
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            # Use persistent client with retries
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Try both collection access approaches
            try:
                # First try to get the existing collection
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Connected to existing collection '{self.collection_name}'")
            except Exception:
                # If it doesn't exist, create it
                logger.info(f"Creating new collection '{self.collection_name}'")
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=None  # Allow custom embedding dimensions
                )
        except Exception as e:
            logger.error(f"Failed to initialize persistent vector database: {e}")
            logger.warning("Creating collection in existing client as fallback")
            
            # Last resort - create collection in the existing client
            try:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=None
                )
            except Exception as final_err:
                logger.critical(f"All database initialization attempts failed: {final_err}")
                raise
    
    def _initialize_embedding_model(self):
        """
        Initialize the embedding model with better error handling and forward context setup.
        """
        try:
            logger.info(f"Initializing embedding model: {self.embedding_model_name}")
            self.embedding_model = LLM(
                model=self.embedding_model_name,
                task="embed",
                enforce_eager=True,
                dtype="float16",
                gpu_memory_utilization=0.8
            )
            
            # Try calling set_forward_context and handle if it doesn't exist
            if hasattr(self.embedding_model, 'set_forward_context'):
                try:
                    self.embedding_model.set_forward_context()
                    logger.info("Forward context set successfully")
                except Exception as e:
                    logger.warning(f"Failed to set forward context: {e}")
                    
            # Test the embedding generation to ensure it works
            test_embed = self.embedding_model.encode(["Test embedding"])
            embed_dim = len(test_embed[0].outputs.embedding)
            logger.info(f"Embedding model initialized successfully with dimension {embed_dim}")
            self.embed_dim = embed_dim
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
            return False

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text with reliable initialization."""
        if not text:
            return np.zeros(self.embed_dim if hasattr(self, 'embed_dim') else 4096).tolist()
        
        # Lazy initialization of embedding model if needed
        if self.embedding_model is None:
            success = self._initialize_embedding_model()
            if not success:
                return self._generate_fallback_embedding(text)
        
        try:
            # Ensure forward context is set (wrapping in try-except)
            try:
                if hasattr(self.embedding_model, 'set_forward_context'):
                    self.embedding_model.set_forward_context()
            except Exception as ctx_err:
                logger.debug(f"Forward context setting error (non-critical): {ctx_err}")
            
            # Generate embedding with vLLM
            output = self.embedding_model.encode([text])[0]
            embedding = output.outputs.embedding
            
            if embedding is not None and len(embedding) > 0:
                # Cache the embedding dimension for future reference
                if not hasattr(self, 'embed_dim'):
                    self.embed_dim = len(embedding)
                return embedding
        except Exception as e:
            logger.warning(f"vLLM embedding generation failed: {e}")
        
        # If we got here, we need to use the fallback
        return self._generate_fallback_embedding(text)

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic pseudo-random embedding based on the text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        logger.info("Using fallback embedding generation")
        import hashlib
        
        # Create a deterministic but unique hash for the text
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Generate 4096 float values (or the dimension we need)
        dim = self.embed_dim if hasattr(self, 'embed_dim') else 4096
        expanded = []
        
        for i in range(dim):
            byte_idx = i % 16
            bit_idx = (i // 16) % 8
            # Create a value between -1 and 1
            val = ((hash_bytes[byte_idx] >> bit_idx) & 1) * 2 - 1
            expanded.append(float(val))
            
        return expanded

    def chunk_text(self, text: str) -> List[str]:
        """Split input text into overlapping chunks while preserving formatting."""
        # For very small texts, just return the original
        if len(text) <= self.chunk_size * 4:  # Rough character estimate
            return [text]
        
        # Split text into sentences first to avoid breaking mid-sentence
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            # Estimate token count (rough approximation)
            sentence_size = len(sentence.split())
            
            # If adding this sentence exceeds chunk size and we already have content
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep overlap by retaining the last portion of the previous chunk
                words = current_chunk.split()
                if len(words) > self.chunk_overlap:
                    overlap_text = " ".join(words[-self.chunk_overlap:])
                    current_chunk = overlap_text + " " + sentence
                    current_size = self.chunk_overlap + sentence_size
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # Add separator if needed
                if current_chunk and not current_chunk.endswith((" ", "\n")):
                    current_chunk += " "
                current_chunk += sentence
                current_size += sentence_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def process_text(self, text: str, source_id: Optional[str] = None) -> List[str]:
        """
        Process text by chunking it and adding it to the vector database.
        
        Args:
            text: The text to process
            source_id: Optional source identifier for the text
            
        Returns:
            List of chunk IDs added to the database
        """
        # Generate a source ID if not provided
        if source_id is None:
            import hashlib
            source_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Chunk the text
        chunks = self.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Process each chunk
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Generate triplets for the chunk
            triplets = self.kg_generator.generate_triplets(chunk)
            
            # Generate embedding for the chunk
            embedding = self._generate_embedding(chunk)
            
            # Store the chunk, embedding, and triplets in the vector database
            metadata = {
                "source_id": source_id,
                "chunk_id": chunk_id,  # Ensure chunk_id is in metadata
                "chunk_index": i,
                "total_chunks": len(chunks),
                "triplets": json.dumps(triplets)
            }
            
            # Check if this chunk already exists in the collection
            try:
                existing = self.collection.get(ids=[chunk_id])
                if len(existing["ids"]) > 0:
                    # Update the existing record
                    self.collection.update(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[metadata]
                    )
                    logger.info(f"Updated chunk {chunk_id} in the database")
                else:
                    # Add a new record
                    self.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[metadata]
                    )
                    logger.info(f"Added chunk {chunk_id} to the database")
            except Exception as e:
                logger.error(f"Error adding chunk {chunk_id} to database: {e}")
        
        return chunk_ids
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a file by reading it, chunking the text, and adding it to the database.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            List of chunk IDs added to the database
        """
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Use the file path as the source ID
        source_id = os.path.basename(file_path)
        
        # Process the text
        return self.process_text(text, source_id)
    
    def query(self, query_text, n_results=5) -> List[Dict[str, Any]]:
        """
        Query the vector database for chunks relevant to the query text.
        
        Args:
            query_text: The query text
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing chunks and their triplets
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query_text)
        
        # Query the vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2  # Get more results to filter duplicates
        )
        
        # Extract the results
        chunks = results["documents"][0] if "documents" in results and len(results["documents"]) > 0 else []
        metadatas = results["metadatas"][0] if "metadatas" in results and len(results["metadatas"]) > 0 else []
        
        # Handle the case where there are no results
        if not chunks or not metadatas:
            return []
        
        # Process results
        unique_results = []
        unique_chunks = []
        
        for chunk, metadata in zip(chunks, metadatas):
            # Check if this chunk is too similar to any already-selected chunk
            if not any(fuzz.ratio(chunk, unique_chunk) > 95 for unique_chunk in unique_chunks):
                unique_chunks.append(chunk)
                
                # Get metadata values with fallbacks
                source_id = metadata.get("source_id", metadata.get("chunk_id", f"unknown_{len(unique_results)}"))
                chunk_index = metadata.get("chunk_index", 0)
                
                # Parse the triplets from metadata with error handling
                try:
                    triplets_str = metadata.get("triplets", "[]")
                    triplets = json.loads(triplets_str) if triplets_str else []
                except json.JSONDecodeError:
                    triplets = []
                    logger.warning(f"Could not parse triplets for chunk {source_id}")
                
                # Add to results with all required fields
                unique_results.append({
                    "chunk": chunk,
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "triplets": triplets
                })
                
                # Stop once we have enough results
                if len(unique_results) >= n_results:
                    break
        
        return unique_results

    def extract_all_triplets(self) -> List[List[str]]:
        """
        Extract all knowledge graph triplets from the database.
        
        Returns:
            A list of all triplets in the database
        """
        # Get all items from the collection
        all_items = self.collection.get()
        
        # Extract triplets from all metadatas
        all_triplets = []
        for metadata in all_items["metadatas"]:
            try:
                triplets = json.loads(metadata["triplets"])
                all_triplets.extend(triplets)
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Could not extract triplets from metadata: {e}")
        
        # Remove duplicates
        unique_triplets = []
        triplet_strs = set()
        
        for triplet in all_triplets:
            triplet_str = json.dumps(triplet)
            if triplet_str not in triplet_strs:
                triplet_strs.add(triplet_str)
                unique_triplets.append(triplet)
        
        return unique_triplets
    
    def save_all_triplets(self, output_file: str) -> None:
        """
        Save all knowledge graph triplets from the database to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        triplets = self.extract_all_triplets()
        self.kg_generator.save_triplets_to_json(triplets, output_file)
        logger.info(f"Saved {len(triplets)} unique triplets to {output_file}")
    
    def generate_rag_response(self, query: str, model_name: Optional[str] = None) -> str:
        """
        Generate a response to a query using RAG and the knowledge graph.
        
        Args:
            query: The user query
            model_name: Optional model name to use for generation
            
        Returns:
            The generated response
        """
        # Get relevant chunks from the vector database
        relevant_chunks = self.query(query)
        
        # Extract the text and triplets
        context_chunks = [result["chunk"] for result in relevant_chunks]
        context_triplets = []
        for result in relevant_chunks:
            context_triplets.extend(result["triplets"])
        
        # Format the context
        context = "\n\nBEGIN_CHUNK\n" + "\nEND_CHUNK\n\nBEGIN_CHUNK\n".join(context_chunks) + "\nEND_CHUNK"
        
        # Format the knowledge graph triplets
        kg_context = ""
        if context_triplets:
            triplet_strs = []
            for t in context_triplets:
                if len(t) == 3:
                    triplet_strs.append(f"[{t[0]}, {t[1]}, {t[2]}]")
            
            kg_context = "\n\nKNOWLEDGE GRAPH TRIPLETS:\n" + "\n".join(triplet_strs)
        
        # Create the prompt
        prompt = (
            "You are a knowledge assistant using both text context and knowledge graph triplets. "
            "Use the following information to answer the user query.\n\n"
            f"TEXT CONTEXT:{context}\n\n"
            f"{kg_context}\n\n"
            "----------\n\n"
            f"Query: {query}\n"
            "Answer (be concise and accurate):"
        )
        
        # Generate a response using DeepSeek
        model = model_name or "deepseek-ai/DeepSeek-V3"
        conversation = self.deepseek_client.conversation(
            system_prompt="You are a helpful assistant with access to a knowledge graph and text context."
        )
        conversation.add_user_message(prompt)
        response = conversation.get_response()
        
        return response


def process_directory(
    directory_path: str,
    output_file: str,
    db_path: str = "kg_rag_db",
    collection_name: str = "kg_chunks",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    file_extensions: List[str] = [".txt", ".md", ".csv", ".json"]
) -> None:
    """
    Process all files in a directory, extract knowledge graph triplets,
    and save them to a JSON file.
    
    Args:
        directory_path: Path to the directory containing text files
        output_file: Path to the output JSON file
        db_path: Path to the vector database directory
        collection_name: Name of the collection in the vector database
        chunk_size: Maximum size of each text chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        file_extensions: List of file extensions to process
    """
    # Initialize the KG RAG system
    kg_rag = KnowledgeGraphRAG(
        db_path=db_path,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Find all files in the directory with the specified extensions
    files_to_process = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            if ext.lower() in file_extensions:
                files_to_process.append(file_path)
    
    # Process each file
    for file_path in files_to_process:
        logger.info(f"Processing file: {file_path}")
        kg_rag.process_file(file_path)
    
    # Save all triplets to the output file
    kg_rag.save_all_triplets(output_file)
    logger.info("Processing complete.")