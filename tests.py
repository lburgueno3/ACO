from ACO import multi_objective_aco, log_print
from graphs import generate_random_graph
import time
import matplotlib.pyplot as plt




# Generate a random graph 
graph = generate_random_graph(
    num_nodes=100,
    num_objectives=3,
    density=0.3,
    min_cost=1,
    max_cost=100,
    seed = 3
)

 # Print graph information
log_print(f"Number of nodes: {graph.num_nodes}")
log_print(f"Number of edges: {len(graph.edges)}")

log_print("\nEdges and their costs:")
for (i, j), costs in graph.edges.items():
    log_print(f"Edge ({i}, {j}): {[round(c, 2) for c in costs]}")

# Test connectivity by checking if each node has at least one neighbor
for node in range(graph.num_nodes):
    neighbors = graph.get_neighbors(node)
    log_print(f"\nNode {node} neighbors: {neighbors}")

start_time = time.time()
pareto_front = multi_objective_aco(graph, start_node=0, end_node=99)
end_time = time.time()

log_print(f"Algorithm runtime: {end_time-start_time}")

# 3D Plot of Pareto Solutions
if pareto_front:
    # Extract costs for plotting
    costs = [solution[1] for solution in pareto_front]
    
    # Create 3D scatter plot
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # Scatter plot of Pareto solutions
    xs = [cost[0] for cost in costs]
    ys = [cost[1] for cost in costs]
    zs = [cost[2] for cost in costs]
    
    ax.scatter(xs, ys, zs, c='purple', marker='o')
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    ax.set_title('Pareto-Optimal Solutions')
    
    # Save the plot
    plot_file = ('pareto_front_3d.png')
    plt.savefig(plot_file)
    log_print(f"\nPareto front 3D plot saved to {plot_file}")
