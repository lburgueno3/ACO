import numpy as np
import random

# Parameters for the ACO
num_ants = 100          # Number of ants
num_iterations = 100   # Number of iterations
alpha = 1.0            # Influence of pheromone
beta = 2.0             # Influence of heuristic information
evaporation_rate = 0.5 # Pheromone evaporation rate
pheromone_init = 0.1   # Initial pheromone level

# File information for printing output
log_file = open('output.txt', 'w')

def log_print(*args, **kwargs):
    """
    Prints to both console and log file
    """
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)
    log_file.flush() 

# Pheromone matrix for each objective
def initialize_pheromone(graph):
    pheromone = {}
    for edge in graph.edges:
        pheromone[edge] = pheromone_init
    return pheromone

# Calculate heuristic (inverse of the distance for each objective to minimize)
def heuristic(graph, i, j):
    costs = graph.get_cost(i, j)
    return 1.0 / (sum(costs) / len(costs))

# Ant Class: Builds a path from start to end node
class Ant:
    def __init__(self, graph, pheromone):
        self.graph = graph
        self.pheromone = pheromone
        self.path = []
        self.total_cost = []
        
    def select_next_node(self, current_node, visited):
        neighbors = self.graph.get_neighbors(current_node)
        neighbors = [n for n in neighbors if n not in visited]  # Avoid cycles
        
        if not neighbors:
            return None  # No available moves
        
        weights = []
      

        for neighbor in neighbors:
            pheromone_level = self.pheromone[(current_node, neighbor)]
            heuristic_val = heuristic(self.graph, current_node, neighbor)  # Heuristic value for the current node to get to the  current neighbor

            
            prob = (pheromone_level ** alpha) * (heuristic_val ** beta)
            weights.append(prob)

        
        # Normalize probabilities
        weights = np.array(weights)
        if weights.sum() == 0:
            return random.choice(neighbors)
        probabilities = weights / weights.sum()
        
        # Choose next node based on probabilities
        return np.random.choice(neighbors, p=probabilities)

    def build_path(self, start_node, end_node):
        current_node = start_node
        visited = set()
        self.path = [start_node]
        self.total_cost = [0] * len(self.graph.get_cost(start_node, next(iter(self.graph.get_neighbors(start_node)))))
        
        while current_node != end_node:
            visited.add(current_node)
            next_node = self.select_next_node(current_node, visited)
            
            if next_node is None:
                return False  # No path found
            
            self.path.append(next_node)
            edge_cost = self.graph.get_cost(current_node, next_node)
            self.total_cost = [self.total_cost[i] + edge_cost[i] for i in range(len(edge_cost))]
            current_node = next_node
            
        return True

# Pareto dominance check
def is_dominated(cost_a, cost_b):
    return all(a >= b for a, b in zip(cost_a, cost_b)) and any(a > b for a, b in zip(cost_a, cost_b))

# Update Pheromone levels
def update_pheromones(pheromone, ants, evaporation_rate):
    for edge in pheromone:
        pheromone[edge] *= (1 - evaporation_rate)
    
    for ant in ants:
        for i in range(len(ant.path) - 1):
            edge = (ant.path[i], ant.path[i+1])
            pheromone[edge] += 1 / np.sum(ant.total_cost)  # Reward shorter paths

# Main ACO Loop with output filtering by start and end nodes
def multi_objective_aco(graph, start_node, end_node):
    pheromone = initialize_pheromone(graph)
    pareto_archive = []  # Store non-dominated solutions

    for iteration in range(num_iterations):
        ants = [Ant(graph, pheromone) for _ in range(num_ants)]

        for ant in ants:
            # Build path from start to end node
            if ant.build_path(start_node, end_node):
                # Convert path nodes to integers for clearer output because they were displaying as np.int64
                path = list(map(int, ant.path))  

                # Check if ant's solution is non-dominated and meets start-end node requirement
                is_non_dominated = all(not is_dominated(ant.total_cost, archive_cost) for _, archive_cost in pareto_archive)
                if is_non_dominated:
                    # Remove dominated solutions
                    pareto_archive = [(p, c) for p, c in pareto_archive if not is_dominated(ant.total_cost, c)]
                    # Add the new non-dominated path and its cost if it's unique
                    if (path, ant.total_cost) not in pareto_archive:
                        pareto_archive.append((path, ant.total_cost))
        
        # Update pheromones based on ant paths
        update_pheromones(pheromone, ants, evaporation_rate)

        # Re-check the archive to ensure no internal domination
        pareto_archive = [
            (p, c) for i, (p, c) in enumerate(pareto_archive)
            if all(not is_dominated(c, c_other) for j, (_, c_other) in enumerate(pareto_archive) if i != j)
        ]

        # Output intermediate Pareto front approximation (every few iterations)
        if iteration % 10 == 0:
            log_print(f"Iteration {iteration}: Current Pareto front with {len(pareto_archive)} solutions")

        
    
    # Final output: Only paths from start_node to end_node
    log_print("\nPareto-optimal paths from start to end node:")
    for path, cost in pareto_archive:
        if path[0] == start_node and path[-1] == end_node:
            print(f"Path: {path}, Cost: {cost}")

    # Return only paths that start at start_node and end at end_node
    return [(path, cost) for path, cost in pareto_archive]