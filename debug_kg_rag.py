#!/usr/bin/env python3
"""
Knowledge Graph RAG System Debug Script with Multithreading

This script performs comprehensive testing of all components in the Knowledge Graph RAG system
with extensive debugging output at each step of the process, using multiple process threads
to query many chunks in bulk for improved performance.

Usage:
    python debug_kg_rag.py [--input-file INPUT_FILE] [--output-dir OUTPUT_DIR] [--verbose] [--threads NUM_THREADS]

If no input file is provided, sample test data will be generated automatically.
"""

import os
import sys
import argparse
import tempfile
import shutil
import json
import logging
import concurrent.futures
import time
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from fuzzywuzzy import fuzz

# Import the modules to test
from libs.kg_rag_integration import KGRAGIntegration
from libs.kg_triplet_generator import KnowledgeGraphTripletGenerator
from libs.query_llm import DeepSeekChat
from libs.text_chunker_rag import KnowledgeGraphRAG
from libs.graph_viz import KnowledgeGraphVisualizer

# Get default queries
with open("default_queries.json", "r") as f:
    DEFAULT_QUERIES = json.load(f)

# Configure detailed logging for debugging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m', # Bold Red
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['RESET']}"

# Set up logging configuration
def setup_logging(verbose=False, log_file="kg_rag_debug.log"):
    """Configure logging with appropriate level and formatters"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    )
    console_formatter = ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("kg_rag_debugger")

class KGRAGDebugger:
    """Debug helper for the Knowledge Graph RAG system with multithreading support"""
    
    def __init__(self, input_file=None, output_dir=None, db_path=None, use_temp=True, verbose=False, num_threads=None):
        """
        Initialize the debugging environment
        
        Args:
            input_file: Optional path to input file
            output_dir: Optional path to output directory
            db_path: Optional path for the vector database
            use_temp: Whether to use temporary directories
            verbose: Whether to enable verbose logging
            num_threads: Number of threads to use for parallel processing (default: CPU count)
        """
        self.logger = setup_logging(verbose=verbose)
        self.start_time = datetime.now()
        self.logger.info(f"Initializing KG RAG Debugger at {self.start_time}")
        
        # Set threading configuration
        self.num_threads = num_threads or os.cpu_count()
        self.logger.info(f"Using {self.num_threads} threads for parallel processing")
        
        # Set up directories
        if use_temp:
            self.base_dir = tempfile.mkdtemp()
            self.logger.info(f"Created temporary directory: {self.base_dir}")
        else:
            self.base_dir = os.path.abspath("kg_rag_debug")
            os.makedirs(self.base_dir, exist_ok=True)
            self.logger.info(f"Using directory: {self.base_dir}")
        
        # Set up database path
        if db_path:
            self.db_path = os.path.abspath(db_path)
            os.makedirs(self.db_path, exist_ok=True)
            self.logger.info(f"Using custom database path: {self.db_path}")
        else:
            self.db_path = os.path.join(self.base_dir, "vector_db")
            self.logger.info(f"Using default database path: {self.db_path}")
        
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(self.base_dir, "output")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Set up input file
        self.input_file = input_file
        if not self.input_file:
            self.input_file = self._create_sample_data()
        else:
            self.logger.info(f"Using provided input file: {self.input_file}")
        
        # Initialize components with extensive debugging
        self._initialize_components()
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        self.logger.info("Creating sample test data")
        
        sample_data = open("default_sample_data.txt", "r").read().strip()
        
        # Save to file
        data_file = os.path.join(self.base_dir, "sample_data.txt")
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(sample_data)
        
        self.logger.info(f"Sample data written to: {data_file}")
        return data_file
    
    def _initialize_components(self, overlap=250, chunk_size=500):
        """Initialize all components for testing with detailed logging"""
        self.logger.info("=======================================")
        self.logger.info("INITIALIZING KG RAG COMPONENTS")
        self.logger.info("=======================================")
        
        try:
            # Initialize DeepSeek client
            self.logger.info("Initializing DeepSeek client...")
            try:
                self.deepseek_client = DeepSeekChat()
                self.logger.info("✓ DeepSeek client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize DeepSeek client: {e}")
                # Create a mock client for testing without API access
                from unittest.mock import MagicMock
                self.deepseek_client = MagicMock()
                self.deepseek_client.chat.return_value = "Mock response for testing purposes"
                # For the conversation mock
                mock_conversation = MagicMock()
                mock_conversation.get_response.return_value = json.dumps([
                    ["Knowledge Graph", "is_a", "Graph Structure"],
                    ["RAG", "stands_for", "Retrieval-Augmented Generation"],
                    ["Knowledge Graph", "enhances", "RAG"]
                ])
                self.deepseek_client.conversation.return_value = mock_conversation
                self.logger.warning("⚠ Using mock DeepSeek client for testing")
            
            # Initialize triplet generator
            self.logger.info("Initializing triplet generator...")
            self.triplet_generator = KnowledgeGraphTripletGenerator()
            self.logger.info("✓ Triplet generator initialized successfully")
            
            # Initialize RAG system with detailed settings
            self.logger.info(f"Initializing RAG system...")
            self.logger.debug(f"  DB path: {self.db_path}")
            self.logger.debug(f"  Collection: debug_collection")
            self.logger.debug(f"  Chunk size: {chunk_size} tokens")
            self.logger.debug(f"  Chunk overlap: {overlap} tokens")
            
            # Custom initialization to handle the collection creation issue
            try:
                self.rag_system = KnowledgeGraphRAG(
                    db_path=self.db_path,
                    collection_name="debug_collection",
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
            except Exception as e:
                self.logger.warning(f"Error in standard initialization: {e}")
                self.logger.info("Attempting alternative initialization...")
                
                # Create ChromaDB client directly
                import chromadb
                os.makedirs(self.db_path, exist_ok=True)
                chroma_client = chromadb.PersistentClient(path=self.db_path)
                
                # Create the collection explicitly
                try:
                    collection = chroma_client.create_collection(name="debug_collection")
                    self.logger.info("Successfully created collection manually")
                    
                    # Now retry initialization
                    self.rag_system = KnowledgeGraphRAG(
                        db_path=self.db_path,
                        collection_name="debug_collection",
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                except Exception as create_err:
                    self.logger.error(f"Failed to create collection: {create_err}")
                    raise
            self.logger.info("✓ RAG system initialized successfully")
            
            # Initialize KG RAG Integration
            self.logger.info("Initializing KG RAG Integration...")
            self.kg_integration = KGRAGIntegration(
                rag_system=self.rag_system,
                triplet_generator=self.triplet_generator
            )
            self.logger.info("✓ KG RAG Integration initialized successfully")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.critical(f"ERROR during component initialization: {e}", exc_info=True)
            raise
    
    def test_triplet_generation(self):
        """Test triplet generation functionality with detailed debugging"""
        self.logger.info("=======================================")
        self.logger.info("TESTING TRIPLET GENERATION")
        self.logger.info("=======================================")
        
        # Read a small portion of the input file for quick testing
        self.logger.info(f"Reading input file: {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as f:
            text = f.read(1000)  # Just read the first 1000 chars
        
        self.logger.debug(f"Sample text for triplet generation:\n{text}")
        
        # Generate triplets with detailed logging
        self.logger.info("Generating triplets from sample text...")
        self.logger.debug("Using KnowledgeGraphTripletGenerator.generate_triplets()")
        
        try:
            start_time = datetime.now()
            triplets = self.triplet_generator.generate_triplets(text)
            duration = datetime.now() - start_time
            
            # Log the results
            if triplets:
                self.logger.info(f"✓ Successfully generated {len(triplets)} triplets in {duration.total_seconds():.2f}s:")
                for i, triplet in enumerate(triplets):
                    if len(triplet) == 3:
                        self.logger.info(f"  Triplet {i+1}: [{triplet[0]}, {triplet[1]}, {triplet[2]}]")
                    else:
                        self.logger.warning(f"  Triplet {i+1} has invalid format: {triplet}")
            else:
                self.logger.warning("⚠ No triplets were generated")
            
            # Save triplets to output file
            triplets_file = os.path.join(self.output_dir, "sample_triplets.json")
            self.logger.info(f"Saving triplets to: {triplets_file}")
            self.triplet_generator.save_triplets_to_json(triplets, triplets_file)
            
            # Print the saved file content for verification
            with open(triplets_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.logger.debug(f"Saved {len(saved_data)} triplets in JSON format")
            
            return triplets
        
        except Exception as e:
            self.logger.error(f"ERROR in triplet generation: {e}", exc_info=True)
            return []
    
    def test_text_chunking(self):
        """Test text chunking functionality with detailed debugging"""
        self.logger.info("=======================================")
        self.logger.info("TESTING TEXT CHUNKING")
        self.logger.info("=======================================")
        
        try:
            # Read the input file
            self.logger.info(f"Reading input file: {self.input_file}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            self.logger.debug(f"Text length: {len(text)} characters")
            
            # Chunk the text with detailed logging
            self.logger.info("Chunking text...")
            self.logger.debug(f"Using chunk_size={self.rag_system.chunk_size}, chunk_overlap={self.rag_system.chunk_overlap}")
            
            start_time = datetime.now()
            chunks = self.rag_system.chunk_text(text)
            duration = datetime.now() - start_time
            
            # Log the results
            self.logger.info(f"✓ Generated {len(chunks)} chunks in {duration.total_seconds():.2f}s:")
            for i, chunk in enumerate(chunks):
                words = len(chunk.split())
                self.logger.info(f"  Chunk {i+1}: {words} words, {len(chunk)} chars")
                self.logger.debug(f"  Preview: {chunk[:100]}...")
            
            # Check for overlap between chunks
            if len(chunks) > 1:
                self.logger.info("Analyzing chunk overlap...")

                configured_overlap = self.rag_system.chunk_overlap

                for i in range(len(chunks) - 1):
                    chunk1_tokens = chunks[i].split()
                    chunk2_tokens = chunks[i+1].split()

                    # How many tokens we *attempt* to overlap (could be smaller if the chunk is short)
                    actual_overlap_size = min(configured_overlap, len(chunk1_tokens), len(chunk2_tokens))

                    # Slice out the region of intended overlap
                    chunk1_overlap = chunk1_tokens[-actual_overlap_size:]
                    chunk2_overlap = chunk2_tokens[:actual_overlap_size]

                    # Count how many of those token positions match exactly
                    exact_matches = sum(
                        1 for w1, w2 in zip(chunk1_overlap, chunk2_overlap) if w1 == w2
                    )

                    self.logger.info(
                        f"  Overlap between chunks {i+1} and {i+2}: "
                        f"{exact_matches} / {actual_overlap_size} tokens match exactly"
                    )

                    if exact_matches > 0:
                        # Optional: show the first few matching tokens
                        matching_tokens = [
                            w1 for w1, w2 in zip(chunk1_overlap, chunk2_overlap) if w1 == w2
                        ]
                        self.logger.debug(
                            f"  Example matching tokens: {matching_tokens[:10]}"
                        )


            # Save a sample chunk to file for inspection
            if chunks:
                sample_chunk_file = os.path.join(self.output_dir, "sample_chunk.txt")
                with open(sample_chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunks[0])
                self.logger.info(f"Saved sample chunk to: {sample_chunk_file}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"ERROR in text chunking: {e}", exc_info=True)
            return []

    # Fix 1: Update the '_process_chunk' method to ensure source_id is correctly set
    def _process_chunk(self, chunk_id, chunk_text):
        """Process a single chunk and return the results (for parallel processing)"""
        try:
            # Generate triplets for this chunk
            # Note: We don't generate embeddings separately as it's handled internally by the rag_system
            triplets = self.triplet_generator.generate_triplets(chunk_text)
            
            return {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "triplets": triplets,
                "source_id": chunk_id,  # Explicitly add source_id matching chunk_id
                "status": "success"
            }
        except Exception as e:
            return {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_id": chunk_id,  # Still add source_id even on error
                "status": "error",
                "error": str(e)
            }

    

    # Fix 3: Update the '_process_single_query' method to handle missing fields
    def _process_single_query(self, query, n_results=3):
        """Process a single query (for parallel processing)"""
        try:
            start_time = time.time()
            results = self.rag_system.query(query, n_results=n_results)
            end_time = time.time()
            
            # Ensure each result has the required fields
            processed_results = []
            for result in results:
                # Create a safe copy with all required fields
                safe_result = {
                    "chunk": result.get("chunk", "No chunk content available"),
                    "source_id": result.get("source_id", f"query_{query.replace(' ', '_')[:10]}"),
                    "triplets": result.get("triplets", [])
                }
                
                # Copy any other fields from the original result
                for key, value in result.items():
                    if key not in safe_result:
                        safe_result[key] = value
                        
                processed_results.append(safe_result)
            
            return {
                "query": query,
                "results": processed_results,
                "time_taken": end_time - start_time,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")
            # Return a valid structure even on error to prevent downstream failures
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "results": [],  # Return empty results on error
                "time_taken": 0
            }

    # Fix 4: Update KnowledgeGraphRAG.query method to ensure it returns results with source_id
    # Add this method to the KnowledgeGraphRAG class
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
                    self.logger.warning(f"Could not parse triplets for chunk {source_id}")
                
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

    def test_file_processing(self):
        """Test processing a file through the RAG system with parallel chunk processing"""
        self.logger.info("=======================================")
        self.logger.info("TESTING FILE PROCESSING & VECTOR DB WITH PARALLEL PROCESSING")
        self.logger.info("=======================================")
        
        try:
            # REMOVE OR COMMENT OUT THIS CONDITIONAL to force using parallel implementation
            # if hasattr(self.rag_system, 'process_file') and callable(getattr(self.rag_system, 'process_file')):
            #    self.logger.info(f"Using original process_file method for {self.input_file}")
            #    chunk_ids = self.rag_system.process_file(self.input_file)
            #    self.logger.info(f"✓ Processed file into {len(chunk_ids)} chunks")
            #    return chunk_ids
            
            # Always use the parallel implementation
            self.logger.info(f"Reading file: {self.input_file}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Chunk the text
            self.logger.info("Chunking text for parallel processing...")
            chunks = self.rag_system.chunk_text(text)
            self.logger.info(f"Generated {len(chunks)} chunks for processing")
            
            # Prepare for parallel processing
            self.logger.info(f"Processing chunks in parallel using {self.num_threads} threads...")
            start_time = datetime.now()
            
            # Process chunks in parallel
            chunk_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Create a list of future objects
                futures = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"chunk_{i+1}"
                    futures.append(executor.submit(self._process_chunk, chunk_id, chunk))
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    chunk_results.append(result)
                    if result["status"] == "success":
                        self.logger.info(f"✓ Processed chunk {result['chunk_id']} successfully with {len(result['triplets'])} triplets")
                    else:
                        self.logger.error(f"✗ Failed to process chunk {result['chunk_id']}: {result['error']}")
            
            duration = datetime.now() - start_time
            
            # Add processed chunks to vector database
            self.logger.info("Adding processed chunks to vector database...")
            chunk_ids = []
            
            # Create a custom method to add chunks to the database with triplets
            def add_chunk_with_triplets(chunk_id, chunk_text, triplets):
                try:
                    # Use the rag_system's methods to add the chunk
                    # This assumes the collection and embeddings are handled internally
                    metadata = {
                        "chunk_id": chunk_id, 
                        "source_id": chunk_id,  # Ensure source_id is set in metadata
                        "triplets": json.dumps(triplets)
                    }
                    
                    # Add to database - we'll use the underlying collection API
                    embedding = self.rag_system._generate_embedding(chunk_text)
                    
                    self.rag_system.collection.add(
                        ids=[chunk_id],
                        documents=[chunk_text],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                    return True
                except Exception as e:
                    self.logger.error(f"Error adding chunk {chunk_id} to database: {e}")
                    return False
            
            # Process results and add to database
            for result in chunk_results:
                if result["status"] == "success":
                    chunk_id = result["chunk_id"]
                    chunk_text = result["text"]
                    triplets = result["triplets"]
                    
                    if add_chunk_with_triplets(chunk_id, chunk_text, triplets):
                        chunk_ids.append(chunk_id)
                        self.logger.debug(f"Added chunk {chunk_id} to database")
            
            self.logger.info(f"✓ Processed {len(chunk_ids)}/{len(chunks)} chunks in {duration.total_seconds():.2f}s")
            
            # Check database state
            self.logger.info("Checking vector database state...")
            try:
                # Get collection info
                collection_items = self.rag_system.collection.get()
                if collection_items and "ids" in collection_items:
                    self.logger.info(f"  Database contains {len(collection_items['ids'])} items")
                    
                    # Check item structure
                    if len(collection_items["ids"]) > 0:
                        self.logger.debug("  Database item structure:")
                        for key in collection_items:
                            self.logger.debug(f"    {key}: {type(collection_items[key])}")
                else:
                    self.logger.warning("  Database appears to be empty")
            except Exception as db_err:
                self.logger.error(f"  Error accessing database: {db_err}")
            
            # Extract triplets from database
            self.logger.info("Extracting all triplets from database...")
            start_time = datetime.now()
            all_triplets = self.rag_system.extract_all_triplets()
            duration = datetime.now() - start_time
            
            self.logger.info(f"✓ Extracted {len(all_triplets)} unique triplets in {duration.total_seconds():.2f}s")
            
            # Save all triplets
            all_triplets_file = os.path.join(self.output_dir, "all_database_triplets.json")
            self.rag_system.save_all_triplets(all_triplets_file)
            self.logger.info(f"Saved all triplets to: {all_triplets_file}")
            
            return chunk_ids
            
        except Exception as e:
            self.logger.error(f"ERROR in file processing: {e}", exc_info=True)
            return []

    
    def test_rag_query(self, batch_queries=None):
        """
        Test querying the RAG system with parallel processing
        
        Args:
            batch_queries: List of queries to process in parallel (optional)
        """
        self.logger.info("=======================================")
        self.logger.info("TESTING RAG QUERY WITH PARALLEL PROCESSING")
        self.logger.info("=======================================")
        
        try:
            # Define test queries if not provided
            if batch_queries is None:
                batch_queries = DEFAULT_QUERIES["test_queries"]
            
            self.logger.info(f"Processing {len(batch_queries)} queries in parallel...")
            for i, query in enumerate(batch_queries):
                self.logger.info(f"  Query {i+1}: '{query}'")
            
            # Process queries in parallel
            start_time = datetime.now()
            query_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Create a list of future objects
                futures = {executor.submit(self._process_single_query, query, 3): query for query in batch_queries}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    query = futures[future]
                    try:
                        result = future.result()
                        query_results.append(result)
                        if result["status"] == "success":
                            self.logger.info(f"✓ Query processed successfully: '{query}' ({result['time_taken']:.2f}s)")
                        else:
                            self.logger.error(f"✗ Query processing failed: '{query}' - {result['error']}")
                    except Exception as exc:
                        self.logger.error(f"✗ Error processing query '{query}': {exc}")
            
            duration = datetime.now() - start_time
            
            # Log results summary
            successful_queries = [r for r in query_results if r["status"] == "success"]
            self.logger.info(f"✓ Processed {len(successful_queries)}/{len(batch_queries)} queries in {duration.total_seconds():.2f}s")
            
            # Calculate average processing time
            if successful_queries:
                avg_time = sum(r["time_taken"] for r in successful_queries) / len(successful_queries)
                self.logger.info(f"  Average query processing time: {avg_time:.2f}s")
                self.logger.info(f"  Single-threaded estimated time: {avg_time * len(batch_queries):.2f}s")
                self.logger.info(f"  Speedup factor: {(avg_time * len(batch_queries)) / duration.total_seconds():.2f}x")
            
            # Save detailed results for inspection
            query_results_file = os.path.join(self.output_dir, "parallel_rag_query_results.json")
            
            serializable_results = []
            for result in query_results:
                if result["status"] == "success":
                    serializable_query_result = {
                        "query": result["query"],
                        "time_taken": result["time_taken"],
                        "results": []
                    }
                    
                    for chunk_result in result["results"]:
                        serializable_chunk = {
                            "source_id": chunk_result.get("source_id", "N/A"),
                            "chunk": chunk_result["chunk"],
                            "triplet_count": len(chunk_result["triplets"]),
                            "triplets": chunk_result["triplets"]
                        }
                        serializable_query_result["results"].append(serializable_chunk)
                    
                    serializable_results.append(serializable_query_result)
                else:
                    serializable_results.append({
                        "query": result["query"],
                        "status": "error",
                        "error": result["error"]
                    })
            
            # Save to file
            with open(query_results_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved parallel query results to: {query_results_file}")
            
            return query_results
            
        except Exception as e:
            self.logger.error(f"ERROR in parallel RAG query: {e}", exc_info=True)
            return []

    def test_knowledge_graph_building(self):
        """Test building the knowledge graph with detailed debugging"""
        self.logger.info("=======================================")
        self.logger.info("TESTING KNOWLEDGE GRAPH BUILDING")
        self.logger.info("=======================================")
        
        try:
            # Build the knowledge graph with detailed logging
            self.logger.info("Building knowledge graph...")
            self.logger.debug("Using kg_integration.build_knowledge_graph()")
            
            start_time = datetime.now()
            graph = self.kg_integration.build_knowledge_graph()
            duration = datetime.now() - start_time
            
            # Log graph statistics
            self.logger.info(f"✓ Knowledge graph built in {duration.total_seconds():.2f}s:")
            self.logger.info(f"  Nodes: {graph.number_of_nodes()}")
            self.logger.info(f"  Edges: {graph.number_of_edges()}")
            
            # Analyze node degree distribution
            if graph.number_of_nodes() > 0:
                self.logger.info("Analyzing node degree distribution:")
                degrees = [d for _, d in graph.degree()]
                avg_degree = sum(degrees) / len(degrees) if degrees else 0
                max_degree = max(degrees) if degrees else 0
                min_degree = min(degrees) if degrees else 0
                
                self.logger.info(f"  Average degree: {avg_degree:.2f}")
                self.logger.info(f"  Maximum degree: {max_degree}")
                self.logger.info(f"  Minimum degree: {min_degree}")
            
            # Log some sample nodes
            self.logger.info("Sample nodes and their connections:")
            sample_nodes = list(graph.nodes())[:min(5, graph.number_of_nodes())]
            
            for i, node in enumerate(sample_nodes):
                self.logger.info(f"  Node {i+1}: '{node}'")
                
                # Log connections for this node
                out_edges = list(graph.out_edges(node, data=True))[:3]
                in_edges = list(graph.in_edges(node, data=True))[:3]
                
                if out_edges:
                    self.logger.info(f"    Outgoing connections:")
                    for src, tgt, data in out_edges:
                        self.logger.info(f"      → '{tgt}' [{data.get('label', 'no-label')}]")
                
                if in_edges:
                    self.logger.info(f"    Incoming connections:")
                    for src, tgt, data in in_edges:
                        self.logger.info(f"      ← '{src}' [{data.get('label', 'no-label')}]")
            
            # Save the graph to different formats
            self.logger.info("Saving knowledge graph to multiple formats...")
            
            # Save as JSON
            graph_json_file = os.path.join(self.output_dir, "knowledge_graph.json")
            self.kg_integration.save_graph_to_file(graph_json_file, format="json")
            self.logger.info(f"  Saved as JSON: {graph_json_file}")
            
            # Save as GraphML
            graph_graphml_file = os.path.join(self.output_dir, "knowledge_graph.graphml")
            self.kg_integration.save_graph_to_file(graph_graphml_file, format="graphml")
            self.logger.info(f"  Saved as GraphML: {graph_graphml_file}")
            
            # Generate and save graph visualizations
            self.logger.info("Generating high-resolution graph visualizations...")
            
            try:
                # Create the visualizer
                visualizer = KnowledgeGraphVisualizer(graph)
                
                # Generate main visualization with adjustments for better readability
                graph_viz_file = os.path.join(self.output_dir, "knowledge_graph.png")
                visualizer.create_advanced_visualization(
                    output_file=graph_viz_file,
                    figsize=(36, 30),  # Much larger canvas
                    layout='fr',        # Fruchterman-Reingold layout
                    edge_curved=0.1     # Slight curve to edges
                )
                self.logger.info(f"✓ Saved high-resolution graph visualization to: {graph_viz_file}")
                
                # Generate NetworkX visualization with small nodes and fonts
                nx_output_file = os.path.join(self.output_dir, "knowledge_graph_networkx.png")
                visualizer.visualize_with_networkx(
                    output_file=nx_output_file,
                    figsize=(36, 30),  # Larger canvas
                    layout='spring',
                    node_size=150,      # Smaller nodes
                    font_size=6         # Smaller fonts
                )
                self.logger.info(f"✓ Saved NetworkX graph visualization to: {nx_output_file}")
                
                # Generate igraph visualization with adjusted parameters
                igraph_output_file = os.path.join(self.output_dir, "knowledge_graph_igraph.png")
                visualizer.visualize_with_igraph(
                    output_file=igraph_output_file,
                    figsize=(2400, 2400)  # Higher resolution
                )
                self.logger.info(f"✓ Saved igraph graph visualization to: {igraph_output_file}")
                
            except Exception as viz_err:
                self.logger.error(f"Error generating graph visualizations: {viz_err}")
                self.logger.error("Falling back to basic visualization method")
                
                # Fallback to simple visualization
                try:
                    basic_viz_file = os.path.join(self.output_dir, "knowledge_graph_basic.png")
                    plt.figure(figsize=(24, 20), dpi=300)
                    pos = nx.spring_layout(graph, k=0.3, seed=42)
                    nx.draw(
                        graph, 
                        pos, 
                        with_labels=True, 
                        node_color='skyblue', 
                        node_size=100,  # Much smaller nodes
                        font_size=6,    # Smaller font
                        edge_color='gray',
                        width=0.5,
                        alpha=0.8
                    )
                    plt.title('Knowledge Graph Basic Visualization')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(basic_viz_file, format='png', dpi=300, bbox_inches='tight')
                    plt.close()
                    self.logger.info(f"✓ Saved basic graph visualization to: {basic_viz_file}")
                except Exception as basic_viz_err:
                    self.logger.error(f"Error generating basic visualization: {basic_viz_err}")

            return graph
            
        except Exception as e:
            self.logger.error(f"ERROR in knowledge graph building: {e}", exc_info=True)
            return nx.DiGraph()  # Return empty graph

    def run_all_tests(self):
        """Run all tests in sequence with detailed logging"""
        self.logger.info("=======================================")
        self.logger.info("STARTING COMPREHENSIVE KG RAG DEBUG TESTS")
        self.logger.info("=======================================")
        
        all_successful = True
        # Initialize test_results as an instance attribute
        self.test_results = {}
        
        try:
            # Run each test in sequence and collect results
            test_functions = [
                ("triplet_generation", self.test_triplet_generation),
                ("text_chunking", self.test_text_chunking),
                ("file_processing", self.test_file_processing),
                ("rag_query", self.test_rag_query),
                ("knowledge_graph_building", self.test_knowledge_graph_building),
                ("graph_query", self.test_graph_query),
                ("hybrid_query", self.test_hybrid_query),
                ("performance_benchmark", self.run_performance_benchmark)
            ]
            
            for test_name, test_func in test_functions:
                try:
                    self.logger.info(f"\nRunning test: {test_name}")
                    result = test_func()
                    test_success = result is not None
                    
                    if isinstance(result, list):
                        test_success = len(result) > 0
                    elif isinstance(result, nx.Graph):
                        test_success = result.number_of_nodes() > 0
                    elif isinstance(result, dict):
                        test_success = len(result) > 0
                        
                    # Store the result in self.test_results instead of a local variable
                    self.test_results[test_name] = test_success
                    
                    if not test_success:
                        all_successful = False
                        self.logger.warning(f"⚠ Test '{test_name}' did not produce expected results")
                except Exception as test_err:
                    self.logger.error(f"Test '{test_name}' failed with error: {test_err}")
                    self.test_results[test_name] = False
                    all_successful = False

            # Create a test summary
            self.logger.info("\n=======================================")
            self.logger.info("TEST EXECUTION SUMMARY")
            self.logger.info("=======================================")
            
            # Use self.test_results instead of test_results
            for test_name, success in self.test_results.items():
                status = "✓ PASSED" if success else "✗ FAILED"
                self.logger.info(f"{status}: {test_name}")
            
            # Log overall success
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            # Generate and save test summary
            summary_file = os.path.join(self.output_dir, "test_summary.txt")
            self.generate_test_summary(summary_file)
            
            self.logger.info("\n=======================================")
            if all_successful:
                self.logger.info(f"ALL TESTS COMPLETED SUCCESSFULLY in {duration}")
            else:
                self.logger.warning(f"TESTS COMPLETED WITH ISSUES in {duration}")
                self.logger.warning("See log for details on failed tests")
            
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("=======================================")
            
            return all_successful
            
        except Exception as e:
            self.logger.critical(f"FATAL ERROR IN TEST EXECUTION: {e}", exc_info=True)
            self.logger.info("\n=======================================")
            self.logger.info("TESTS FAILED - See log for details")
            self.logger.info("=======================================")
            return False

    def _process_graph_query(self, query, max_hops=2):
        """Process a single graph query (for parallel processing)"""
        try:
            start_time = time.time()
            triplets = self.kg_integration.query_graph(query, max_hops=max_hops)
            end_time = time.time()
            
            return {
                "query": query,
                "triplets": triplets,
                "time_taken": end_time - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "query": query,
                "status": "error",
                "error": str(e)
            }

    def test_graph_query(self, batch_queries=None):
        """
        Test querying the knowledge graph with parallel processing
        
        Args:
            batch_queries: List of queries to process in parallel (optional)
        """
        self.logger.info("=======================================")
        self.logger.info("TESTING GRAPH QUERY WITH PARALLEL PROCESSING")
        self.logger.info("=======================================")
        
        try:
            # Define test queries if not provided
            if batch_queries is None:
                batch_queries = DEFAULT_QUERIES["graph_queries"]
            
            self.logger.info(f"Processing {len(batch_queries)} graph queries in parallel...")
            for i, query in enumerate(batch_queries):
                self.logger.info(f"  Query {i+1}: '{query}'")
            
            # Process queries in parallel
            start_time = datetime.now()
            query_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Create a list of future objects with max_hops=2
                futures = {executor.submit(self._process_graph_query, query, 2): query for query in batch_queries}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    query = futures[future]
                    try:
                        result = future.result()
                        query_results.append(result)
                        if result["status"] == "success":
                            triplet_count = len(result["triplets"]) if result["triplets"] else 0
                            self.logger.info(f"✓ Graph query processed successfully: '{query}' - {triplet_count} triplets ({result['time_taken']:.2f}s)")
                        else:
                            self.logger.error(f"✗ Graph query processing failed: '{query}' - {result['error']}")
                    except Exception as exc:
                        self.logger.error(f"✗ Error processing graph query '{query}': {exc}")
            
            duration = datetime.now() - start_time
            
            # Log results summary
            successful_queries = [r for r in query_results if r["status"] == "success"]
            self.logger.info(f"✓ Processed {len(successful_queries)}/{len(batch_queries)} graph queries in {duration.total_seconds():.2f}s")
            
            # Calculate average processing time
            if successful_queries:
                avg_time = sum(r["time_taken"] for r in successful_queries) / len(successful_queries)
                self.logger.info(f"  Average graph query processing time: {avg_time:.2f}s")
                self.logger.info(f"  Single-threaded estimated time: {avg_time * len(batch_queries):.2f}s")
                self.logger.info(f"  Speedup factor: {(avg_time * len(batch_queries)) / duration.total_seconds():.2f}x")
            
            # Save query results
            graph_query_file = os.path.join(self.output_dir, "parallel_graph_query_results.json")
            with open(graph_query_file, "w", encoding="utf-8") as f:
                json.dump(query_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved parallel graph query results to: {graph_query_file}")
            
            return query_results
            
        except Exception as e:
            self.logger.error(f"ERROR in parallel graph query: {e}", exc_info=True)
            return []

    def _create_custom_graph_viz(self, graph, output_file, width=12, height=10):
        """
        Create a custom visualization for small graphs with node sizes based on connectivity.
        
        Args:
            graph: The graph to visualize
            output_file: Path to save the visualization
            width: Width of the figure in inches
            height: Height of the figure in inches
        """
        plt.figure(figsize=(width, height))
        
        # Calculate degree centrality for each node to determine size
        degree_dict = dict(graph.degree())
        
        # Normalize node sizes based on degree (number of connections)
        min_degree = min(degree_dict.values()) if degree_dict else 1
        max_degree = max(degree_dict.values()) if degree_dict else 1
        
        # Size scaling parameters
        min_node_size = 500
        max_node_size = 2500
        min_font_size = 8
        max_font_size = 16
        
        # Calculate node sizes and font sizes
        node_sizes = {}
        font_sizes = {}
        
        for node, degree in degree_dict.items():
            # Normalize degree between 0 and 1
            if max_degree == min_degree:
                norm_degree = 0.5  # All nodes have the same degree
            else:
                norm_degree = (degree - min_degree) / (max_degree - min_degree)
            
            # Set node size based on normalized degree
            node_sizes[node] = min_node_size + norm_degree * (max_node_size - min_node_size)
            
            # Set font size based on normalized degree
            font_sizes[node] = min_font_size + norm_degree * (max_font_size - min_font_size)
        
        # Create a nice layout - spring layout works well for small graphs
        pos = nx.spring_layout(graph, seed=42, k=0.3)
        
        # Draw the graph with varying node sizes
        node_list = list(graph.nodes())
        size_list = [node_sizes[node] for node in node_list]
        
        # Draw nodes with size based on degree
        nx.draw_networkx_nodes(
            graph, 
            pos,
            nodelist=node_list,
            node_size=size_list,
            node_color='lightblue',
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, 
            pos, 
            width=1.0, 
            alpha=0.5,
            arrows=True,
            arrowsize=10
        )
        
        # Draw node labels with varying sizes
        labels = {node: node for node in graph.nodes()}
        for node, (x, y) in pos.items():
            plt.text(
                x, y, labels[node],
                fontsize=font_sizes[node],
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Draw edge labels
        edge_labels = {(u, v): data['label'] for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph, 
            pos, 
            edge_labels=edge_labels, 
            font_size=8
        )
        
        # Turn off the axis
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved custom graph visualization to {output_file}")

    def _process_hybrid_query(self, query, n_chunks=3, max_hops=2):
        """Process a single hybrid query (for parallel processing)"""
        try:
            start_time = time.time()
            results = self.kg_integration.hybrid_query(
                query,
                n_chunks=n_chunks,
                max_hops=max_hops
            )
            end_time = time.time()
            
            return {
                "query": query,
                "results": results,
                "time_taken": end_time - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "query": query,
                "status": "error",
                "error": str(e)
            }

    def test_hybrid_query(self, batch_queries=None):
        """
        Test hybrid querying (RAG + graph) with parallel processing
        
        Args:
            batch_queries: List of queries to process in parallel (optional)
        """
        self.logger.info("=======================================")
        self.logger.info("TESTING HYBRID QUERY WITH PARALLEL PROCESSING")
        self.logger.info("=======================================")
        
        try:
            # Define test queries if not provided
            if batch_queries is None:
                batch_queries = DEFAULT_QUERIES["hybrid_queries"]
            
            self.logger.info(f"Processing {len(batch_queries)} hybrid queries in parallel...")
            for i, query in enumerate(batch_queries):
                self.logger.info(f"  Query {i+1}: '{query}'")
            
            # Process queries in parallel
            start_time = datetime.now()
            query_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Create a list of future objects
                futures = {executor.submit(self._process_hybrid_query, query, 3, 2): query for query in batch_queries}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    query = futures[future]
                    try:
                        result = future.result()
                        query_results.append(result)
                        if result["status"] == "success":
                            self.logger.info(f"✓ Hybrid query processed successfully: '{query}' ({result['time_taken']:.2f}s)")
                        else:
                            self.logger.error(f"✗ Hybrid query processing failed: '{query}' - {result['error']}")
                    except Exception as exc:
                        self.logger.error(f"✗ Error processing hybrid query '{query}': {exc}")
            
            duration = datetime.now() - start_time
            
            # Log results summary
            successful_queries = [r for r in query_results if r["status"] == "success"]
            self.logger.info(f"✓ Processed {len(successful_queries)}/{len(batch_queries)} hybrid queries in {duration.total_seconds():.2f}s")
            
            # Calculate average processing time
            if successful_queries:
                avg_time = sum(r["time_taken"] for r in successful_queries) / len(successful_queries)
                self.logger.info(f"  Average hybrid query processing time: {avg_time:.2f}s")
                self.logger.info(f"  Single-threaded estimated time: {avg_time * len(batch_queries):.2f}s")
                self.logger.info(f"  Speedup factor: {(avg_time * len(batch_queries)) / duration.total_seconds():.2f}x")
            
            # Save detailed results for inspection
            for i, result in enumerate(successful_queries):
                query = result["query"]
                query_id = f"query_{i+1}"
                
                # Save the response for this query
                response_file = os.path.join(self.output_dir, f"hybrid_response_{query_id}.txt")
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(result["results"]["response"])
                self.logger.debug(f"Saved response for query '{query}' to: {response_file}")
            
            # Save complete results JSON
            hybrid_results_file = os.path.join(self.output_dir, "parallel_hybrid_query_results.json")
            
            # Create a serializable version of the results
            serializable_results = []
            for result in query_results:
                if result["status"] == "success":
                    hybrid_results = result["results"]
                    serializable_result = {
                        "query": result["query"],
                        "time_taken": result["time_taken"],
                        "rag_chunks": [
                            {
                                "chunk": chunk["chunk"],
                                "source_id": chunk.get("source_id", "N/A"),
                                "triplet_count": len(chunk["triplets"])
                            }
                            for chunk in hybrid_results["rag_chunks"]
                        ],
                        "graph_triplets_count": len(hybrid_results["graph_triplets"]),
                        "response_preview": hybrid_results["response"][:200] + "..."
                    }
                    serializable_results.append(serializable_result)
                else:
                    serializable_results.append({
                        "query": result["query"],
                        "status": "error",
                        "error": result["error"]
                    })
            
            with open(hybrid_results_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved parallel hybrid query results to: {hybrid_results_file}")
            
            return query_results
            
        except Exception as e:
            self.logger.error(f"ERROR in parallel hybrid query: {e}", exc_info=True)
            return []

    def run_performance_benchmark(self):
        """Run performance benchmarking tests to compare single vs multi-threaded execution"""
        self.logger.info("=======================================")
        self.logger.info("RUNNING MULTITHREADING PERFORMANCE BENCHMARK")
        self.logger.info("=======================================")
        
        try:
            # Create test queries
            test_queries = DEFAULT_QUERIES["performance_benchmark_queries"]
            
            # Test single-threaded performance (use just 1 thread)
            self.logger.info("Testing single-threaded performance...")
            original_threads = self.num_threads
            self.num_threads = 1
            
            # Measure time for single-threaded
            single_start = datetime.now()
            single_results = self.test_rag_query(test_queries)
            single_duration = datetime.now() - single_start
            
            self.logger.info(f"Single-threaded query processing time: {single_duration.total_seconds():.2f}s")
            
            # Restore thread count and test multi-threaded performance
            self.num_threads = original_threads
            self.logger.info(f"Testing multi-threaded performance with {self.num_threads} threads...")
            
            # Measure time for multi-threaded
            multi_start = datetime.now()
            multi_results = self.test_rag_query(test_queries)
            multi_duration = datetime.now() - multi_start
            
            self.logger.info(f"Multi-threaded query processing time: {multi_duration.total_seconds():.2f}s")
            
            # Calculate speedup
            speedup = single_duration.total_seconds() / multi_duration.total_seconds()
            
            # Report results
            self.logger.info("=======================================")
            self.logger.info("PERFORMANCE BENCHMARK RESULTS")
            self.logger.info("=======================================")
            self.logger.info(f"Number of test queries: {len(test_queries)}")
            self.logger.info(f"Number of threads used: {original_threads}")
            self.logger.info(f"Single-threaded execution time: {single_duration.total_seconds():.2f}s")
            self.logger.info(f"Multi-threaded execution time: {multi_duration.total_seconds():.2f}s")
            self.logger.info(f"Speedup factor: {speedup:.2f}x")
            
            # Theoretical vs. actual speedup
            theoretical_speedup = min(original_threads, len(test_queries))
            efficiency = speedup / theoretical_speedup
            self.logger.info(f"Theoretical maximum speedup: {theoretical_speedup:.2f}x")
            self.logger.info(f"Parallel efficiency: {efficiency:.2f} ({efficiency*100:.1f}%)")
            
            # Save benchmark results
            benchmark_file = os.path.join(self.output_dir, "performance_benchmark.json")
            benchmark_results = {
                "test_queries": len(test_queries),
                "thread_count": original_threads,
                "single_threaded_time": single_duration.total_seconds(),
                "multi_threaded_time": multi_duration.total_seconds(),
                "speedup_factor": speedup,
                "theoretical_speedup": theoretical_speedup,
                "parallel_efficiency": efficiency
            }
            
            with open(benchmark_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=2)
            self.logger.info(f"Saved benchmark results to: {benchmark_file}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"ERROR in performance benchmark: {e}", exc_info=True)
            return {}
    def generate_test_summary(self, output_file=None):
        """
        Generate a concise summary of all test results and metrics.
        
        Args:
            output_file: Optional path to save the summary (defaults to summary.txt in output_dir)
        
        Returns:
            A string containing the formatted summary
        """
        # Default output file if none provided
        if output_file is None:
            output_file = os.path.join(self.output_dir, "test_summary.txt")
        
        # Create summary header
        end_time = datetime.now()
        duration = end_time - self.start_time
        summary = []
        summary.append("=" * 80)
        summary.append(f"KG RAG SYSTEM TEST SUMMARY")
        summary.append(f"Run completed at: {end_time}")
        summary.append(f"Total execution time: {duration}")
        summary.append("=" * 80)
        summary.append("")
        
        # Add system configuration
        summary.append("SYSTEM CONFIGURATION:")
        summary.append(f"- Number of threads: {self.num_threads}")
        summary.append(f"- Database path: {self.db_path}")
        summary.append(f"- Input file: {os.path.basename(self.input_file)}")
        summary.append(f"- Output directory: {self.output_dir}")
        summary.append("")
        
        # Add test results
        summary.append("TEST RESULTS:")
        all_passed = True
        for test_name, success in self.test_results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            if not success:
                all_passed = False
            summary.append(f"- {test_name}: {status}")
        summary.append("")
        
        # Add detailed metrics if available
        if hasattr(self, "benchmark_results") and self.benchmark_results:
            summary.append("PERFORMANCE METRICS:")
            br = self.benchmark_results
            summary.append(f"- Single-threaded execution: {br.get('single_threaded_time', 0):.2f}s")
            summary.append(f"- Multi-threaded execution: {br.get('multi_threaded_time', 0):.2f}s")
            summary.append(f"- Speedup factor: {br.get('speedup_factor', 0):.2f}x")
            summary.append(f"- Parallel efficiency: {br.get('parallel_efficiency', 0)*100:.1f}%")
            summary.append("")
        
        # Add knowledge graph stats if available
        if hasattr(self, "kg_integration") and hasattr(self.kg_integration, "graph") and self.kg_integration.graph:
            graph = self.kg_integration.graph
            summary.append("KNOWLEDGE GRAPH STATISTICS:")
            summary.append(f"- Total nodes: {graph.number_of_nodes()}")
            summary.append(f"- Total edges: {graph.number_of_edges()}")
            if graph.number_of_nodes() > 0:
                degrees = [d for _, d in graph.degree()]
                avg_degree = sum(degrees) / len(degrees) if degrees else 0
                max_degree = max(degrees) if degrees else 0
                summary.append(f"- Average node degree: {avg_degree:.2f}")
                summary.append(f"- Maximum node degree: {max_degree}")
                
                # Find most connected nodes
                top_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:5]
                if top_nodes:
                    summary.append("- Top connected entities:")
                    for node, degree in top_nodes:
                        if isinstance(node, str):
                            node_str = node[:40] + "..." if len(node) > 40 else node
                            summary.append(f"  * '{node_str}' ({degree} connections)")
            summary.append("")
        
        # Add triplet extraction stats
        triplets_file = os.path.join(self.output_dir, "all_database_triplets.json")
        if os.path.exists(triplets_file):
            try:
                with open(triplets_file, 'r', encoding='utf-8') as f:
                    triplets_data = json.load(f)
                    summary.append("TRIPLET EXTRACTION STATISTICS:")
                    summary.append(f"- Total unique triplets extracted: {len(triplets_data)}")
                    
                    # Count relationship types
                    predicate_counts = {}
                    for triplet in triplets_data:
                        if isinstance(triplet, list) and len(triplet) >= 3:
                            predicate = triplet[1]
                            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
                    
                    # Show top relationship types
                    if predicate_counts:
                        top_predicates = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        summary.append("- Top relationship types:")
                        for pred, count in top_predicates:
                            summary.append(f"  * '{pred}': {count} occurrences")
                    summary.append("")
            except Exception as e:
                self.logger.error(f"Error parsing triplets file: {e}")
        
        # Add overall assessment
        summary.append("OVERALL ASSESSMENT:")
        if all_passed:
            summary.append("All tests completed successfully. The KG RAG system is functioning as expected.")
        else:
            summary.append("Some tests failed. Review the log file for detailed error information.")
        summary.append("")
        
        # Add timestamp
        summary.append(f"Summary generated at {end_time}")
        summary.append("=" * 80)
        
        # Join all lines
        summary_text = "\n".join(summary)
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        self.logger.info(f"Test summary saved to: {output_file}")
        
        return summary_text

def main():
    """Main function to run the KG RAG debugging script"""
    parser = argparse.ArgumentParser(description="Debug the Knowledge Graph RAG system with multithreading")
    parser.add_argument("--input-file", help="Path to input text file")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--db-path", help="Path to store the vector database")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--threads", type=int, help="Number of threads to use (default: CPU count)")
    args = parser.parse_args()
    
    # Create and run the debugger
    debugger = KGRAGDebugger(
        input_file=args.input_file,
        output_dir=args.output_dir,
        db_path=args.db_path,
        use_temp=not args.keep_temp,
        verbose=args.verbose,
        num_threads=args.threads
    )
    
    try:
        success = debugger.run_all_tests()
        
        # Cleanup if requested
        if not args.keep_temp:
            debugger.cleanup()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDebug session interrupted by user")
        # Cleanup if requested
        if not args.keep_temp:
            debugger.cleanup()
        sys.exit(130)


if __name__ == "__main__":
    main()