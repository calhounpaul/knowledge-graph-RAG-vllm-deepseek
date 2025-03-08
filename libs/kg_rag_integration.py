"""
Knowledge Graph RAG Integration Module

This module provides integration between knowledge graph triplets and RAG systems,
enabling graph-aware retrieval, triplet extraction from RAG chunks, and visualization.
"""

import os
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from libs.text_chunker_rag import KnowledgeGraphRAG
from libs.kg_triplet_generator import KnowledgeGraphTripletGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class KGRAGIntegration:
    """
    A class that integrates knowledge graph triplets with RAG systems
    for enhanced retrieval and reasoning.
    """
    
    def __init__(
        self,
        rag_system: Optional[KnowledgeGraphRAG] = None,
        triplet_generator: Optional[KnowledgeGraphTripletGenerator] = None,
        db_path: str = "kg_rag_db",
        collection_name: str = "kg_chunks"
    ):
        """
        Initialize the KG RAG integration module.
        
        Args:
            rag_system: Optional existing RAG system
            triplet_generator: Optional existing triplet generator
            db_path: Path to the vector database directory
            collection_name: Name of the collection in the vector database
        """
        # Initialize or use provided RAG system
        self.rag_system = rag_system or KnowledgeGraphRAG(
            db_path=db_path,
            collection_name=collection_name
        )
        
        # Initialize or use provided triplet generator
        self.triplet_generator = triplet_generator or KnowledgeGraphTripletGenerator()
        
        # Initialize the graph
        self.graph = nx.DiGraph()
    
    def build_knowledge_graph(self) -> nx.DiGraph:
        """
        Build a NetworkX knowledge graph from all triplets in the RAG database.
        
        Returns:
            A NetworkX directed graph representing the knowledge graph
        """
        # Get all triplets from the RAG database
        triplets = self.rag_system.extract_all_triplets()
        
        # Create a new graph
        self.graph = nx.DiGraph()
        
        # Add nodes and edges from the triplets
        for triplet in triplets:
            if len(triplet) == 3:
                subject, predicate, obj = triplet
                
                # Add the subject and object as nodes if they don't exist
                if not self.graph.has_node(subject):
                    self.graph.add_node(subject)
                if not self.graph.has_node(obj):
                    self.graph.add_node(obj)
                
                # Add the edge with the predicate as the label
                self.graph.add_edge(subject, obj, label=predicate)
        
        logger.info(f"Built knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def visualize_graph(
        self,
        output_file: Optional[str] = None,
        max_nodes: int = 50,
        width: int = 12,
        height: int = 10,
        visualize_all_nodes: bool = False  # Parameter to visualize all nodes
    ):
        """
        Visualize the knowledge graph using matplotlib with node size proportional to connectivity.
        
        Args:
            output_file: Optional path to save the visualization as an image
            max_nodes: Maximum number of nodes to include in the visualization
            width: Width of the figure in inches
            height: Height of the figure in inches
            visualize_all_nodes: If True, visualize all nodes regardless of count
        """
        # Make sure we have a graph
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.build_knowledge_graph()
        
        # Use all nodes if visualize_all_nodes is True, otherwise limit by max_nodes
        if visualize_all_nodes:
            graph_to_viz = self.graph
            logger.info(f"Visualizing complete graph with {self.graph.number_of_nodes()} nodes")
        # If the graph is too large, create a subgraph with the most connected nodes
        elif self.graph.number_of_nodes() > max_nodes:
            # Get the nodes with the highest degree
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
            
            # Create a subgraph with these nodes
            subgraph = self.graph.subgraph(top_nodes)
            logger.info(f"Created subgraph with {subgraph.number_of_nodes()} nodes for visualization")
            graph_to_viz = subgraph
        else:
            graph_to_viz = self.graph
        
        # Calculate node degrees for sizing
        node_degrees = dict(graph_to_viz.degree())
        n_nodes = graph_to_viz.number_of_nodes()
        
        # Base size adjustments
        base_node_size = 500
        base_font_size = 10
        edge_width = 1.0
        show_labels = True
        
        # Scale down base parameters for larger graphs
        if n_nodes > 100:
            base_node_size = 200
            base_font_size = 8
            edge_width = 0.5
        if n_nodes > 500:
            base_node_size = 100
            base_font_size = 6
            edge_width = 0.3
        if n_nodes > 1000:
            base_node_size = 50
            base_font_size = 5
            edge_width = 0.2
            show_labels = False
        if n_nodes > 3000:
            base_node_size = 20
            edge_width = 0.1
            show_labels = False
        
        # Set up the figure with increased size for larger graphs
        fig_width = width
        fig_height = height
        if n_nodes > 500:
            fig_width = width * 1.5
            fig_height = height * 1.5
        if n_nodes > 2000:
            fig_width = width * 2
            fig_height = height * 2
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Use spring layout for node positioning
        # For larger graphs, increase k to space nodes more
        k_value = 0.5 if n_nodes < 500 else 2.0
        pos = nx.spring_layout(graph_to_viz, seed=42, k=k_value)
        
        # Calculate node sizes based on connectivity
        # Use logarithmic scaling to prevent extremely large/small nodes
        min_degree = min(node_degrees.values()) if node_degrees else 1
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        # Create scaled node sizes
        node_sizes = []
        node_font_sizes = {}
        min_size_factor = 0.5  # Minimum node size multiplier
        max_size_factor = 3.0  # Maximum node size multiplier
        
        for node in graph_to_viz.nodes():
            degree = node_degrees[node]
            
            # Scale the size factor logarithmically
            if max_degree == min_degree:
                size_factor = 1.0  # If all nodes have same degree
            else:
                # Normalize the degree to a value between 0 and 1
                norm_degree = (degree - min_degree) / (max_degree - min_degree)
                # Scale between min and max factor using a logarithmic scale
                size_factor = min_size_factor + (max_size_factor - min_size_factor) * norm_degree
            
            # Apply the factor to the base size
            node_size = base_node_size * size_factor
            node_sizes.append(node_size)
            
            # Calculate font size proportionally to node size, with a minimum
            font_factor = size_factor * 0.8 + 0.2  # Slightly less variation in font sizes
            node_font_sizes[node] = max(base_font_size * font_factor, 4) if show_labels else 0
        
        # Draw the nodes with varying sizes
        nx.draw_networkx_nodes(
            graph_to_viz, 
            pos, 
            node_size=node_sizes, 
            node_color="lightblue", 
            alpha=0.8
        )
        
        # Draw the node labels with varying font sizes
        if show_labels:
            # Custom label drawing to allow different font sizes for each node
            for node, (x, y) in pos.items():
                font_size = node_font_sizes[node]
                if font_size > 0:
                    plt.text(
                        x, y, node, 
                        fontsize=font_size,
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
        
        # Draw the edges
        nx.draw_networkx_edges(
            graph_to_viz, 
            pos, 
            width=edge_width, 
            alpha=0.5, 
            arrows=True
        )
        
        # Draw edge labels (predicates) - only if not too many nodes
        if show_labels and n_nodes <= 200:
            edge_labels = {(u, v): data["label"] for u, v, data in graph_to_viz.edges(data=True)}
            nx.draw_networkx_edge_labels(
                graph_to_viz, 
                pos, 
                edge_labels=edge_labels, 
                font_size=base_font_size * 0.8
            )
        
        # Turn off the axis
        plt.axis("off")
        
        # Save the full graph data as JSON alongside the visualization
        if output_file and visualize_all_nodes:
            json_file = os.path.splitext(output_file)[0] + "_full.json"
            graph_data = {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {"source": u, "target": v, "predicate": data["label"]}
                    for u, v, data in self.graph.edges(data=True)
                ]
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved complete graph data to {json_file}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if an output file is specified
        if output_file:
            # For large graphs, increase DPI
            dpi = 300
            if n_nodes > 1000:
                dpi = 500
            
            plt.savefig(output_file, format="png", dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved knowledge graph visualization to {output_file}")
        
        # Show the figure
        plt.show()
    
    def query_graph(self, query: str, max_hops: int = 2) -> List[List[str]]:
        """
        Query the knowledge graph for relationships relevant to the query.
        
        Args:
            query: The query to search for in the graph
            max_hops: Maximum number of hops to traverse in the graph
            
        Returns:
            A list of triplets relevant to the query
        """
        # Make sure we have a graph
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.build_knowledge_graph()
        
        # Extract entities from the query
        # This is a simplistic approach; a better one would use NER
        query_triplets = self.triplet_generator.generate_triplets(query)
        
        # Extract subjects and objects as potential entities
        query_entities = set()
        for triplet in query_triplets:
            if len(triplet) >= 3:
                query_entities.add(triplet[0])  # Subject
                query_entities.add(triplet[2])  # Object
        
        # Find exact matches and similar nodes in the graph
        matched_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                matched_nodes.add(entity)
            else:
                # Look for similar nodes (simple substring matching)
                for node in self.graph.nodes():
                    if (entity.lower() in node.lower() or 
                        node.lower() in entity.lower()):
                        matched_nodes.add(node)
        
        # If no matches found, return empty list
        if not matched_nodes:
            logger.info(f"No matching nodes found for query: {query}")
            return []
        
        # Collect triplets by exploring the graph around matched nodes
        relevant_triplets = []
        visited_edges = set()
        
        # For each matched node, explore up to max_hops steps away
        for start_node in matched_nodes:
            # BFS to explore the graph
            queue = [(start_node, 0)]  # (node, distance)
            visited = {start_node}
            
            while queue:
                node, distance = queue.pop(0)
                
                # Stop if we've gone too far
                if distance >= max_hops:
                    continue
                
                # Explore outgoing edges
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    edge_key = (node, neighbor, data["label"])
                    
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        relevant_triplets.append([node, data["label"], neighbor])
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                
                # Explore incoming edges
                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    edge_key = (neighbor, node, data["label"])
                    
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        relevant_triplets.append([neighbor, data["label"], node])
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        logger.info(f"Found {len(relevant_triplets)} relevant triplets for query: {query}")
        return relevant_triplets
    
    def hybrid_query(
        self,
        query: str,
        n_chunks: int = 3,
        max_hops: int = 2,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a hybrid query using both RAG and knowledge graph.
        
        Args:
            query: The query to search for
            n_chunks: Number of text chunks to retrieve from RAG
            max_hops: Maximum number of hops to traverse in the graph
            model_name: Optional model name to use for generation
            
        Returns:
            A dictionary with both text chunks and graph triplets
        """
        # Get relevant chunks from RAG
        rag_results = self.rag_system.query(query, n_results=n_chunks)
        
        # Query the knowledge graph
        graph_triplets = self.query_graph(query, max_hops=max_hops)
        
        # Combine the results
        combined_results = {
            "rag_chunks": rag_results,
            "graph_triplets": graph_triplets
        }
        
        # Generate a response using both sources
        context_chunks = [result["chunk"] for result in rag_results]
        context = "\n\nBEGIN_CHUNK\n" + "\nEND_CHUNK\n\nBEGIN_CHUNK\n".join(context_chunks) + "\nEND_CHUNK"
        
        # Format the knowledge graph triplets
        kg_context = ""
        if graph_triplets:
            triplet_strs = []
            for t in graph_triplets:
                if len(t) == 3:
                    triplet_strs.append(f"[{t[0]}, {t[1]}, {t[2]}]")
            
            kg_context = "\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n" + "\n".join(triplet_strs)
        
        # Create the prompt
        prompt = (
            "You are a knowledge assistant using both text context and a knowledge graph. "
            "Use the following information to answer the user query.\n\n"
            f"TEXT CONTEXT:{context}\n\n"
            f"{kg_context}\n\n"
            "----------\n\n"
            f"Query: {query}\n"
            "Answer (be concise and accurate):"
        )
        
        # Generate a response
        model = model_name or "deepseek-ai/DeepSeek-V3"
        conversation = self.rag_system.deepseek_client.conversation(
            system_prompt=(
                "You are a helpful assistant with access to a knowledge graph and text context. "
                "Integrate both sources of information to provide accurate and comprehensive answers."
            )
        )
        conversation.add_user_message(prompt)
        response = conversation.get_response()
        
        combined_results["response"] = response
        return combined_results
    
    def save_graph_to_file(self, output_file: str, format: str = "graphml") -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            output_file: Path to the output file
            format: Format to save the graph (graphml, gexf, etc.)
        """
        # Make sure we have a graph
        if not self.graph or self.graph.number_of_nodes() == 0:
            self.build_knowledge_graph()
        
        # Save the graph in the specified format
        if format == "graphml":
            nx.write_graphml(self.graph, output_file)
        elif format == "gexf":
            nx.write_gexf(self.graph, output_file)
        elif format == "gml":
            nx.write_gml(self.graph, output_file)
        elif format == "json":
            # Convert to a JSON-serializable structure
            graph_data = {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {"source": u, "target": v, "predicate": data["label"]}
                    for u, v, data in self.graph.edges(data=True)
                ]
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved knowledge graph to {output_file} in {format} format")


def main(input_file: str, output_dir: str, visualize: bool = True):
    """
    Main function to demonstrate the KG RAG integration.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save outputs
        visualize: Whether to generate and save a visualization
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the RAG system
    rag_system = KnowledgeGraphRAG(
        db_path=os.path.join(output_dir, "rag_db"),
        collection_name="kg_chunks"
    )
    
    # Process the input file
    logger.info(f"Processing input file: {input_file}")
    rag_system.process_file(input_file)
    
    # Save all triplets to a JSON file
    triplets_file = os.path.join(output_dir, "triplets.json")
    rag_system.save_all_triplets(triplets_file)
    logger.info(f"Saved triplets to {triplets_file}")
    
    # Create the integration module
    kg_rag = KGRAGIntegration(rag_system=rag_system)
    
    # Build the knowledge graph
    kg_rag.build_knowledge_graph()
    
    # Save the graph in various formats
    kg_rag.save_graph_to_file(os.path.join(output_dir, "knowledge_graph.graphml"), format="graphml")
    kg_rag.save_graph_to_file(os.path.join(output_dir, "knowledge_graph.json"), format="json")
    
    # Visualize the graph if requested
    if visualize:
        try:
            viz_file = os.path.join(output_dir, "knowledge_graph.png")
            # Modified to visualize all nodes
            kg_rag.visualize_graph(output_file=viz_file, visualize_all_nodes=True)
            logger.info(f"Saved visualization to {viz_file}")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
    
    logger.info("KG RAG integration processing complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG Integration")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("--output-dir", default="./kg_rag_output", help="Directory to save outputs")
    parser.add_argument("--visualize", action="store_true", help="Generate and save a visualization")
    
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.visualize)