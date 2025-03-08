# libs/graph_viz.py

import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    def __init__(self, graph: nx.Graph):
        """Initialize the visualizer with a NetworkX graph."""
        self.graph = graph

    def visualize_with_networkx(self, output_file: str, figsize=(24, 20), layout='spring', node_size=200, font_size=8):
        """
        Visualize the graph using NetworkX with community detection clustering.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in inches
            layout: Layout algorithm ('spring', 'kamada_kawai', or 'circular')
            node_size: Base size of nodes (smaller values reduce overlap)
            font_size: Base size of fonts (smaller values reduce overlap)
        """
        from networkx.algorithms.community import greedy_modularity_communities

        # Detect communities for coloring
        communities = greedy_modularity_communities(self.graph)
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i

        colors = [community_map.get(node, 0) for node in self.graph.nodes()]

        plt.figure(figsize=figsize, dpi=300)  # Higher DPI for better resolution

        # Create layout with appropriate spacing
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=0.3, iterations=100, seed=42)  # Increased k for more spacing
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.circular_layout(self.graph)

        # Scale node sizes based on degree centrality to make important nodes stand out
        degree_dict = dict(self.graph.degree())
        max_degree = max(degree_dict.values()) if degree_dict else 1
        
        # Calculate node sizes: smaller base size, proportional to degree
        node_sizes = [node_size * (0.5 + 0.5 * (degree_dict.get(node, 0) / max_degree)) for node in self.graph.nodes()]

        # Draw the graph with smaller nodes and configurable parameters
        nx.draw(
            self.graph,
            pos,
            node_color=colors,
            with_labels=True,
            cmap=plt.cm.tab20,  # More distinct colors
            node_size=node_sizes,
            edge_color='gray',
            alpha=0.7,
            font_size=font_size,  # Smaller font size
            font_weight='normal',
            font_family='sans-serif',
            width=0.5,  # Thinner edges
            linewidths=0.5,  # Thinner node borders
        )

        plt.title('Knowledge Graph Visualization with Community Clusters')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_with_igraph(self, output_file: str, figsize=(1600, 1600)):
        """
        Visualize the graph using igraph with Louvain clustering.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in pixels
        """
        # Convert the directed graph to an undirected graph for igraph
        undirected_graph = self.graph.to_undirected()
        
        try:
            # Create igraph from the undirected NetworkX graph
            ig_graph = ig.Graph.from_networkx(undirected_graph)

            # Louvain clustering
            communities = ig_graph.community_multilevel()
            
            # Get node degree for sizing
            degrees = ig_graph.degree()
            max_degree = max(degrees) if degrees else 1
            
            # Calculate normalized vertex sizes (smaller overall)
            vertex_sizes = [10 + 15 * (degree / max_degree) for degree in degrees]
            
            # Create labels with appropriate size
            label_sizes = [8 + 4 * (degree / max_degree) for degree in degrees]

            # Configure visualization with smaller nodes and more space
            visual_style = {
                "bbox": figsize,
                "margin": 100,  # Larger margin
                "vertex_size": vertex_sizes,
                "vertex_color": [communities.membership[v.index] for v in ig_graph.vs],
                "vertex_label": ig_graph.vs["_nx_name"],
                "vertex_label_size": label_sizes,  # Smaller labels
                "vertex_label_dist": 1.5,  # Move labels further from nodes
                "edge_arrow_size": 0.5,
                "edge_width": 0.5,  # Thinner edges
                "layout": ig_graph.layout_fruchterman_reingold(weights=None, seed=42, grid="grid")  # More controlled layout
            }

            # Save with high-resolution
            plot = ig.plot(communities, output_file, mark_groups=True, **visual_style)
            plot.save()
            del plot
            
            logger.info(f"Successfully created igraph visualization with {len(ig_graph.vs)} nodes and {len(ig_graph.es)} edges")
            
        except Exception as e:
            logger.error(f"Error in igraph visualization: {e}")
            logger.info("Falling back to NetworkX visualization")
            
            # Fallback to NetworkX visualization with adjusted parameters
            plt.figure(figsize=(figsize[0]/100, figsize[1]/100))  # Convert to inches
            pos = nx.spring_layout(undirected_graph, k=0.3, seed=42)
            
            # Scale node sizes based on degree
            degree_dict = dict(undirected_graph.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            node_sizes = [100 * (0.5 + 0.5 * (degree_dict.get(node, 0) / max_degree)) for node in undirected_graph.nodes()]
            
            nx.draw(
                undirected_graph, 
                pos, 
                with_labels=True, 
                node_color='skyblue', 
                node_size=node_sizes,
                font_size=8,
                edge_color='gray',
                width=0.5,
                alpha=0.8
            )
            plt.title('NetworkX Visualization (igraph fallback)')
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_advanced_visualization(self, output_file: str, figsize=(24, 20), layout='fr', edge_curved=0.2):
        """
        Create a more advanced visualization with better error handling.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in inches
            layout: Layout algorithm ('fr' for Fruchterman-Reingold, 'kk' for Kamada-Kawai)
            edge_curved: Curvature of edges (0 for straight, higher values for more curves)
        """
        try:
            # Convert to igraph for better layout algorithms if possible
            try:
                import igraph as ig
                undirected_graph = self.graph.to_undirected()
                ig_graph = ig.Graph.from_networkx(undirected_graph)
                
                # Get node count
                node_count = len(ig_graph.vs)
                logger.info(f"Creating visualization for graph with {node_count} nodes")
            except Exception as e:
                logger.error(f"Error creating igraph from NetworkX: {e}")
                # Fall back to direct NetworkX visualization
                self._create_networkx_only_visualization(output_file, figsize)
                return
            
            # Auto-adjust parameters based on graph size
            if node_count > 200:
                vertex_size = 3
                label_size = 6
                edge_width = 0.3
            elif node_count > 100:
                vertex_size = 5
                label_size = 7
                edge_width = 0.4
            else:
                vertex_size = 8
                label_size = 8
                edge_width = 0.5
            
            # Create an appropriate layout with fixed seed
            try:
                if layout == 'fr':
                    # Use fixed seed for reproducible layouts
                    layout_coords = ig_graph.layout_fruchterman_reingold(seed=42)
                elif layout == 'kk':
                    layout_coords = ig_graph.layout_kamada_kawai()
                else:
                    layout_coords = ig_graph.layout_auto()
                    
                # Convert positions to dictionary for NetworkX
                positions = layout_coords.coords
                pos = {node: positions[i] for i, node in enumerate(self.graph.nodes())}
            except Exception as e:
                logger.error(f"Error creating layout: {e}")
                # Fall back to NetworkX's built-in layout
                pos = nx.spring_layout(self.graph, seed=42)
            
            # Set up the plot
            plt.figure(figsize=figsize, dpi=300)
            
            # Draw nodes 
            nx.draw_networkx_nodes(
                self.graph, 
                pos, 
                node_size=300,
                node_color='skyblue',
                alpha=0.8,
                linewidths=0.5,
            )
            
            # Draw edges with slight curvature to reduce overlap
            nx.draw_networkx_edges(
                self.graph,
                pos,
                width=edge_width,
                alpha=0.5,
                edge_color='gray',
                connectionstyle=f'arc3, rad={edge_curved}'
            )
            
            # Draw labels with appropriate size
            nx.draw_networkx_labels(
                self.graph,
                pos,
                font_size=label_size,
                font_family='sans-serif',
                font_weight='normal',
                alpha=0.8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1)
            )
            
            # Configure the plot
            plt.title('Knowledge Graph Visualization')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created advanced visualization at {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating advanced visualization: {e}")
            # Fall back to a simpler visualization method
            self._create_networkx_only_visualization(output_file, figsize)
        
    def _create_networkx_only_visualization(self, output_file: str, figsize=(24, 20)):
        """
        Create a visualization using only NetworkX functionality as a fallback.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in inches
        """
        try:
            logger.info("Using NetworkX-only visualization as fallback")
            plt.figure(figsize=figsize, dpi=300)
            
            # Get graph properties
            node_count = self.graph.number_of_nodes()
            
            # Adjust parameters based on graph size
            if node_count > 1000:
                node_size = 20
                font_size = 6
            elif node_count > 500:
                node_size = 50
                font_size = 7
            elif node_count > 100:
                node_size = 100
                font_size = 8
            else:
                node_size = 200
                font_size = 10
            
            # Create a spring layout with fixed seed
            pos = nx.spring_layout(self.graph, seed=42, k=0.3)
            
            # Draw the graph
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color='skyblue',
                node_size=node_size,
                font_size=font_size,
                font_weight='normal',
                edge_color='gray',
                width=0.5,
                alpha=0.8,
                connectionstyle='arc3, rad=0.1'  # Slight curve to reduce overlap
            )
            
            # Set up the plot
            plt.title('Knowledge Graph Visualization (NetworkX fallback)')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created fallback visualization at {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating fallback visualization: {e}")
            # Create an ultra-minimal visualization as last resort
            self._create_minimal_visualization(output_file)

    def _create_minimal_visualization(self, output_file: str):
        """
        Create a minimal visualization with maximum compatibility as last resort.
        
        Args:
            output_file: Path to save the visualization
        """
        try:
            logger.info("Using minimal visualization as last resort")
            plt.figure(figsize=(12, 10), dpi=100)
            
            # Create a small subgraph if the graph is too large
            if self.graph.number_of_nodes() > 50:
                # Get the most connected nodes
                degree_dict = dict(self.graph.degree())
                top_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)[:50]
                graph_to_viz = self.graph.subgraph(top_nodes)
                logger.info(f"Using top 50 nodes for minimal visualization")
            else:
                graph_to_viz = self.graph
            
            # Use the simplest possible layout
            try:
                pos = nx.circular_layout(graph_to_viz)
            except:
                # Create a grid layout as absolute fallback
                nodes = list(graph_to_viz.nodes())
                grid_size = int(np.ceil(np.sqrt(len(nodes))))
                pos = {}
                for i, node in enumerate(nodes):
                    row = i // grid_size
                    col = i % grid_size
                    pos[node] = np.array([col, -row])
            
            # Draw with minimal settings
            nx.draw(
                graph_to_viz,
                pos,
                with_labels=False,  # No labels for maximum compatibility
                node_color='blue',
                node_size=50,
                edge_color='gray',
                width=0.5,
                alpha=0.6
            )
            
            plt.title('Knowledge Graph (Minimal Visualization)')
            plt.axis('off')
            plt.savefig(output_file, format='png', dpi=100)
            plt.close()
            
            logger.info(f"Created minimal visualization at {output_file}")
        except Exception as e:
            logger.error(f"Failed to create even minimal visualization: {e}")
            # If we can't create any visualization, create a text file explaining the error
            with open(output_file.replace('.png', '.txt'), 'w') as f:
                f.write(f"Failed to generate graph visualization: {e}\n")
                f.write(f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def visualize_with_igraph(self, output_file: str, figsize=(1600, 1600)):
        """
        Visualize the graph using igraph with proper error handling around the seed parameter.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in pixels
        """
        # Convert the graph to undirected for igraph
        undirected_graph = self.graph.to_undirected()
        
        try:
            # Create igraph from NetworkX graph
            ig_graph = ig.Graph.from_networkx(undirected_graph)
            
            # Get node degree for sizing
            degrees = ig_graph.degree()
            max_degree = max(degrees) if degrees else 1
            
            # Calculate normalized vertex sizes (smaller overall)
            vertex_sizes = [10 + 15 * (degree / max_degree) for degree in degrees]
            
            # Create labels with appropriate size
            label_sizes = [8 + 4 * (degree / max_degree) for degree in degrees]
            
            # Fix for seed parameter: Use a different layout algorithm that doesn't require a matrix seed
            # or pass seed differently depending on igraph version
            try:
                # Try to get igraph version
                import igraph
                igraph_version = igraph.__version__
                logger.info(f"Using igraph version: {igraph_version}")
                
                # For newer versions, we need to handle seed differently
                if hasattr(igraph, "__version__") and igraph.__version__ >= "0.10.0":
                    # Use a layout without seed parameter
                    layout = ig_graph.layout_fruchterman_reingold()
                else:
                    # For older versions, use integer seed
                    layout = ig_graph.layout_fruchterman_reingold(seed=42)
            except:
                # If we can't determine version or there's an error, try without seed
                layout = ig_graph.layout_fruchterman_reingold()
            
            # Configure visualization with smaller nodes and more space
            visual_style = {
                "layout": layout,
                "bbox": figsize,
                "margin": 100,  # Larger margin
                "vertex_size": vertex_sizes,
                "vertex_color": [i % 9 for i in range(len(ig_graph.vs))],  # Simple color scheme
                "vertex_label": ig_graph.vs["_nx_name"],
                "vertex_label_size": label_sizes,
                "vertex_label_dist": 1.5,  # Move labels further from nodes
                "edge_arrow_size": 0.5,
                "edge_width": 0.5  # Thinner edges
            }
            
            # Try to use communities for coloring if available
            try:
                communities = ig_graph.community_multilevel()
                # Add community colors to visual style
                visual_style["vertex_color"] = [communities.membership[v.index] for v in ig_graph.vs]
                
                # Save with high-resolution
                plot = ig.plot(communities, output_file, mark_groups=True, **visual_style)
                plot.save()
                del plot
                
                logger.info(f"Successfully created igraph visualization with communities")
            except Exception as comm_error:
                logger.warning(f"Could not use community detection: {comm_error}")
                
                # Try without communities
                try:
                    plot = ig.plot(ig_graph, output_file, **visual_style)
                    plot.save()
                    del plot
                    logger.info(f"Successfully created igraph visualization without communities")
                except Exception as plot_error:
                    logger.error(f"Error in basic igraph plotting: {plot_error}")
                    raise
            
        except Exception as e:
            logger.error(f"Error in igraph visualization: {e}")
            logger.info("Falling back to NetworkX visualization")
            
            # Fallback to NetworkX visualization with adjusted parameters
            plt.figure(figsize=(figsize[0]/100, figsize[1]/100))  # Convert to inches
            pos = nx.spring_layout(undirected_graph, k=0.3, seed=42)
            
            # Scale node sizes based on degree
            degree_dict = dict(undirected_graph.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            node_sizes = [100 * (0.5 + 0.5 * (degree_dict.get(node, 0) / max_degree)) for node in undirected_graph.nodes()]
            
            nx.draw(
                undirected_graph, 
                pos, 
                with_labels=True, 
                node_color='skyblue', 
                node_size=node_sizes,
                font_size=8,
                edge_color='gray',
                width=0.5,
                alpha=0.8
            )
            plt.title('NetworkX Visualization (igraph fallback)')
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_advanced_visualization(self, output_file: str, figsize=(24, 20), layout='fr', edge_curved=0.2):
        """
        Create a more advanced visualization with better error handling for the seed parameter.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure (width, height) in inches
            layout: Layout algorithm ('fr' for Fruchterman-Reingold, 'kk' for Kamada-Kawai)
            edge_curved: Curvature of edges (0 for straight, higher values for more curves)
        """
        try:
            # Convert to igraph for better layout algorithms if possible
            try:
                import igraph as ig
                import numpy as np
                undirected_graph = self.graph.to_undirected()
                ig_graph = ig.Graph.from_networkx(undirected_graph)
                
                # Get node count
                node_count = len(ig_graph.vs)
                logger.info(f"Creating visualization for graph with {node_count} nodes")
            except Exception as e:
                logger.error(f"Error creating igraph from NetworkX: {e}")
                # Fall back to direct NetworkX visualization
                self._create_networkx_only_visualization(output_file, figsize)
                return
            
            # Auto-adjust parameters based on graph size
            if node_count > 200:
                vertex_size = 3
                label_size = 6
                edge_width = 0.3
            elif node_count > 100:
                vertex_size = 5
                label_size = 7
                edge_width = 0.4
            else:
                vertex_size = 8
                label_size = 8
                edge_width = 0.5
            
            # Create an appropriate layout with fixed seed
            try:
                # Handle the seed parameter based on igraph version
                if layout == 'fr':
                    try:
                        # Try without seed parameter first
                        layout_coords = ig_graph.layout_fruchterman_reingold()
                    except Exception as e1:
                        logger.warning(f"Error with default FR layout: {e1}")
                        try:
                            # Try with None seed
                            layout_coords = ig_graph.layout_fruchterman_reingold(seed=None)
                        except Exception as e2:
                            logger.warning(f"Error with None seed FR layout: {e2}")
                            # Try with integer seed as last resort
                            layout_coords = ig_graph.layout_fruchterman_reingold(seed=42)
                elif layout == 'kk':
                    layout_coords = ig_graph.layout_kamada_kawai()
                else:
                    layout_coords = ig_graph.layout_circle()
                    
                # Convert positions to dictionary for NetworkX
                positions = layout_coords.coords
                pos = {node: positions[i] for i, node in enumerate(self.graph.nodes())}
            except Exception as e:
                logger.error(f"Error creating layout: {e}")
                # Fall back to NetworkX's built-in layout
                pos = nx.spring_layout(self.graph, seed=42)
            
            # Set up the plot
            plt.figure(figsize=figsize, dpi=300)
            
            # Try to use community detection for coloring
            try:
                # Try to detect communities
                communities = ig_graph.community_multilevel()
                community_map = {}
                for i, comm in enumerate(communities):
                    for node_idx in comm:
                        node_name = ig_graph.vs[node_idx]["_nx_name"]
                        community_map[node_name] = i
                
                # Create color map
                import matplotlib.cm as cm
                num_communities = max(community_map.values()) + 1
                color_map = cm.rainbow(np.linspace(0, 1, num_communities))
                node_colors = [color_map[community_map.get(node, 0)] for node in self.graph.nodes()]
            except Exception as e:
                logger.warning(f"Community detection failed: {e}, using default colors")
                node_colors = 'skyblue'
            
            # Calculate node sizes based on degree
            degree_dict = dict(self.graph.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            node_sizes = []
            for node in self.graph.nodes():
                degree = degree_dict.get(node, 0)
                # Scale size based on degree
                size_factor = 0.5 + 0.5 * (degree / max_degree)
                node_sizes.append(300 * size_factor)
            
            # Draw nodes 
            nx.draw_networkx_nodes(
                self.graph, 
                pos, 
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                linewidths=0.5,
            )
            
            # Draw edges with slight curvature to reduce overlap
            nx.draw_networkx_edges(
                self.graph,
                pos,
                width=edge_width,
                alpha=0.5,
                edge_color='gray',
                connectionstyle=f'arc3, rad={edge_curved}'
            )
            
            # Draw labels with appropriate size and handling for large graphs
            if node_count <= 500:  # Only show labels for smaller graphs
                # Calculate font sizes based on node degree
                font_sizes = {}
                min_font = label_size - 2
                max_font = label_size + 2
                for node in self.graph.nodes():
                    degree = degree_dict.get(node, 0)
                    # Scale font size based on degree
                    font_factor = 0.5 + 0.5 * (degree / max_degree)
                    font_sizes[node] = min_font + font_factor * (max_font - min_font)
                
                # Draw labels with custom sizes
                for node, (x, y) in pos.items():
                    plt.text(
                        x, y, node,
                        fontsize=font_sizes.get(node, label_size),
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1)
                    )
            
            # Configure the plot
            plt.title('Knowledge Graph Visualization')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created advanced visualization at {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating advanced visualization: {e}")
            # Fall back to a simpler visualization method
            self._create_networkx_only_visualization(output_file, figsize)