import numpy as np
import random

# Define the graph structure
class Graph:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.edges = edges # edges is a dict of {(i, j): [cost1, cost2, ...]} format
    
    def get_neighbors(self, node):
        return [j for (i, j) in self.edges.keys() if i == node]

    def get_cost(self, i, j):
        return self.edges[(i, j)] if (i, j) in self.edges else None
    

def generate_random_graph(num_nodes, num_objectives, density=0.3, min_cost=1, max_cost=100, seed=3):
    """
    Generate a random graph for multi-objective ACO testing.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes in the graph
    num_objectives : int
        Number of objective functions (costs) per edge
    density : float, optional (default=0.3)
        Probability of edge creation between any two nodes (0 to 1)
    min_cost : int, optional (default=1)
        Minimum cost value for each objective
    max_cost : int, optional (default=100)
        Maximum cost value for each objective
    seed : int, optional (default=None)
        Random seed for reproducibility
    
    Returns:
    --------
    Graph
        A Graph instance with random edges and costs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize empty edges dictionary
    edges = {}
    
    # Generate random edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Skip self-loops
            if i == j:
                continue
            
            # Randomly decide if this edge should exist
            if random.random() < density:
                # Generate random costs for each objective
                costs = [random.uniform(min_cost, max_cost) for _ in range(num_objectives)]
                edges[(i, j)] = costs
    
    # Ensure the graph is connected
    # First, create a list of all nodes
    nodes = list(range(num_nodes))
    connected_nodes = {0}  # Start with node 0
    remaining_nodes = set(nodes[1:])
    
    # Connect all nodes
    while remaining_nodes:
        # Pick a random node from connected nodes
        from_node = random.choice(list(connected_nodes))
        # Pick a random node from remaining nodes
        to_node = random.choice(list(remaining_nodes))
        
        # Add edges in both directions to ensure connectivity
        if (from_node, to_node) not in edges:
            edges[(from_node, to_node)] = [random.uniform(min_cost, max_cost) for _ in range(num_objectives)]
        if (to_node, from_node) not in edges:
            edges[(to_node, from_node)] = [random.uniform(min_cost, max_cost) for _ in range(num_objectives)]
        
        # Mark the new node as connected
        connected_nodes.add(to_node)
        remaining_nodes.remove(to_node)
    
    return Graph(num_nodes, edges)
