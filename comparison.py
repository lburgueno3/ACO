import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from ACO import multi_objective_aco
from graphs import generate_random_graph


def random_search_multiobjective(graph, start_node, end_node, num_iterations=100):
    """
    Perform random search for multi-objective path finding
    """
    pareto_archive = []
    
    for _ in range(num_iterations):
        # Generate a random path
        path = [start_node]
        current = start_node
        visited = set([current])
        
        while current != end_node:
            neighbors = [n for n in graph.get_neighbors(current) if n not in visited]
            if not neighbors:
                break
            
            next_node = np.random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # Check if path is valid
        if path[-1] == end_node:
            # Calculate total costs
            total_cost = [0] * len(graph.get_cost(path[0], path[1]))
            for i in range(len(path)-1):
                edge_costs = graph.get_cost(path[i], path[i+1])
                total_cost = [total_cost[j] + edge_costs[j] for j in range(len(edge_costs))]
            
            # Check if solution is non-dominated
            is_non_dominated = all(
                not all(a <= b for a, b in zip(total_cost, archive_cost)) 
                for _, archive_cost in pareto_archive
            )
            
            if is_non_dominated:
                pareto_archive = [
                    (p, c) for p, c in pareto_archive 
                    if not all(a <= b for a, b in zip(c, total_cost))
                ]
                pareto_archive.append((path, total_cost))
    
    return [(path, cost) for path, cost in pareto_archive]

def run_comparison(num_runs=30, seed=3):
    # Create output directory
    output_dir = 'multi_obj_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate graph
    np.random.seed(seed)
    graph = generate_random_graph(
        num_nodes=100,
        num_objectives=3,
        density=0.3,
        min_cost=1,
        max_cost=100,
        seed=seed
    )
    
    # Tracking results for each method
    results = {
        'ACO': {'solutions': [], 'times': []},
        'Random Search': {'solutions': [], 'times': []}
    }
    
    # Logging files
    log_file_path = os.path.join(output_dir, 'comparison_log.txt')
    solutions_file_path = os.path.join(output_dir, 'solutions.txt')
    with open(log_file_path, 'w') as log_file, open(solutions_file_path, "w") as solutions_file:
        # Perform runs for each method
        for run in range(num_runs):
            log_file.write(f"\n--- Run {run+1} ---\n")
            solutions_file.write(f"\n--- Run {run+1} ---\n")
            
            # ACO
            start_time = time.time()
            aco_solutions = multi_objective_aco(graph, start_node=0, end_node=99)
            aco_time = time.time() - start_time
            aco_objectives = [solution[1] for solution in aco_solutions]

            # Write ACO solutions to solutions file
            solutions_file.write("ACO Solutions:\n")
            for sol in aco_solutions:
                solutions_file.write(f"{sol}\n")
            solutions_file.write(f"Pareto Front Size: {len(aco_solutions)}\n")

            
            
            # Store ACO results
            results['ACO']['solutions'].append(len(aco_solutions))
            results['ACO']['times'].append(aco_time)
            
            log_file.write(f"ACO Time: {aco_time}\n")
            log_file.write(f"ACO Solutions: {len(aco_solutions)}\n")
            
            # Random Search
            start_time = time.time()
            random_solutions = random_search_multiobjective(graph, start_node=0, end_node=99)
            random_time = time.time() - start_time
            random_objectives = [solution[1] for solution in random_solutions]

             # Write Random Search solutions to solutions file
            solutions_file.write("\nRandom Search Solutions:\n")
            for sol in random_solutions:
                solutions_file.write(f"{sol}\n")
            solutions_file.write(f"Pareto Front Size: {len(random_solutions)}\n")
            
             # Plot Pareto front for this run
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            

            # Scatter plot for ACO solutions
            if len(aco_objectives) > 0:
                ax.scatter(
                    [cost[0] for cost in aco_objectives], [cost[1] for cost in aco_objectives], [cost[2] for cost in aco_objectives],
                    c='purple', label='ACO', alpha=0.7
                )
            
            # Scatter plot for Random Search solutions
            if len(random_objectives) > 0:
                ax.scatter(
                    [cost[0] for cost in random_objectives], [cost[1] for cost in random_objectives], [cost[2] for cost in random_objectives],
                    c='green', label='Random Search', alpha=0.7
                )
            
            # Plot settings
            ax.set_title(f'Pareto Front Comparison (Run {run + 1})')
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.legend()
            
            # Save the figure for this run
            run_output_path = os.path.join(output_dir, f'pareto_front_run_{run + 1}.png')
            plt.tight_layout()
            plt.savefig(run_output_path)
            plt.close()
                
            # Store Random Search results
            results['Random Search']['solutions'].append(len(random_solutions))
            results['Random Search']['times'].append(random_time)
            
            log_file.write(f"Random Search Time: {random_time}\n")
            log_file.write(f"Random Search Solutions: {len(random_solutions)}\n")
        
        #Wilcoxon test
        log_file.write("\n--- Wilcoxon Rank-Sum Test Results ---\n")
        
        # Perform Wilcoxon test on number of solutions
        solutions_statistic, solutions_p_value = stats.wilcoxon(
            results['ACO']['solutions'], 
            results['Random Search']['solutions']
        )
        log_file.write("Number of Solutions Comparison:\n")
        log_file.write(f"  Statistic: {solutions_statistic}\n")
        log_file.write(f"  p-value: {solutions_p_value}\n")
        
        # Perform Wilcoxon test on computation times
        time_statistic, time_p_value = stats.wilcoxon(
            results['ACO']['times'], 
            results['Random Search']['times']
        )
        log_file.write("\nComputation Time Comparison:\n")
        log_file.write(f"  Statistic: {time_statistic}\n")
        log_file.write(f"  p-value: {time_p_value}\n")
        
        # Create descriptive statistics DataFrame
        descriptive_stats = pd.DataFrame({
            'ACO Solutions': results['ACO']['solutions'],
            'Random Search Solutions': results['Random Search']['solutions'],
            'ACO Times': results['ACO']['times'],
            'Random Search Times': results['Random Search']['times']
        })
        
        # Save descriptive statistics
        log_file.write("\n--- Descriptive Statistics ---\n")
        log_file.write(str(descriptive_stats.describe()))
        
        # Save results to CSV
        descriptive_stats.to_csv(os.path.join(output_dir, 'method_comparison.csv'), index=False)
        
        # Create visualization of solutions and times
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot for number of solutions
        ax1.boxplot([
            results['ACO']['solutions'], 
            results['Random Search']['solutions']
        ], labels=['ACO', 'Random Search'])
        ax1.set_title('Number of Solutions')
        ax1.set_ylabel('Solution Count')
        
        # Box plot for computation times
        ax2.boxplot([
            results['ACO']['times'], 
            results['Random Search']['times']
        ], labels=['ACO', 'Random Search'])
        ax2.set_title('Computation Times')
        ax2.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'solutions_times_comparison.png'))
        plt.close()
