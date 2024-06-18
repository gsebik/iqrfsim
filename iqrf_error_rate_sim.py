import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import argparse
import sys


# Function to calculate the stability factor of a graph
# The stability factor calculated by the first node wich indegree is smaller then VRN accoding to the research

def calculate_stability_factor(G):
    stability_factors = []
    for node in G.nodes():
        in_degree = G.in_degree(node)
        vrn_number = node
        if in_degree < vrn_number:
            stability_factors.append(in_degree)
    return min(stability_factors) if stability_factors else max(G.in_degree(node) for node in G.nodes())


# Function to create a random network with a given stability factor
def create_random_network(num_nodes, desired_stability_factor):
    attempts = 0
    while attempts < 1000:  # Limiting the attempts to prevent infinite loop
        G = nx.gnp_random_graph(num_nodes, 0.5, directed=True)
        if calculate_stability_factor(G) == desired_stability_factor:
            return G
        attempts += 1
    raise ValueError("Failed to create a network with the desired stability factor within the attempt limit.")


def simulate_broadcast_with_failures_inallbroadcast(G, num_failures=2):
    num_nodes = len(G.nodes())

    # Initialize a fully connected network
    num_timeslots = num_nodes - 1
    
    # Record of nodes that have received the message, initialized with just the source node (0)
    received_message = [False] * num_nodes
    received_message[0] = True  # Assuming node 0 is the source
    
    # Select the failing links once before the timeslot simulation
    all_edges = list(G.edges())
    if num_failures >= len(all_edges):
        raise ValueError("Number of failures exceeds the number of available edges.")
    
    
    
    # Simulate for each timeslot
    for _ in range(num_timeslots):
        
        failed_edge_indices = np.random.choice(range(len(all_edges)), num_failures, replace=False)
        failed_edges = {all_edges[i] for i in failed_edge_indices}
        # Determine which nodes can receive the message in this timeslot
        new_receivers = []
        for i, received in enumerate(received_message):
            if received:
                # Attempt to broadcast the message to neighbors
                for neighbor in list(G.neighbors(i)):
                    if (i, neighbor) not in failed_edges and not received_message[neighbor]:
                        new_receivers.append(neighbor)

                    if (neighbor, i) not in failed_edges and not received_message[neighbor]:
                        new_receivers.append(neighbor)
        
        # Update the receivers list
        for receiver in new_receivers:
            received_message[receiver] = True
    
    return received_message

def print_two_arrays_as_rows(array1, array2):
    # Print headers if necessary
    print("Row 1\t", end="")
    for item in array1:
        print(item, end="\t")
    print()  # Ends the current line to start the second row

    print("Row 2\t", end="")
    for item in array2:
        print(item, end="\t")
    print()  # Ends the current line for formatting

import random

def simulate_broadcast_with_failures(G, num_failures, allow_reception_from_all=True, random_failures_each_timeslot=True):
    num_nodes = len(G.nodes())

    # Initialize a fully connected network
    num_timeslots = num_nodes - 1
    
    # Records for message reception status
    received_message = [False] * num_nodes
    message_status = ['none'] * num_nodes

    # Initialize the source node
    received_message[0] = True
    message_status[0] = 'normal'  # Source node is always normal since it originates the message

    link_errorlist = []
    log_message = ""


    all_edges = list(G.edges())
    np.random.shuffle(all_edges)

    # Set initial failed edges if not randomizing each timeslot
    if not random_failures_each_timeslot:
        failed_edges = set(all_edges[:num_failures])
        link_errorlist.append(failed_edges)

    for timeslot in range(num_timeslots):
        if random_failures_each_timeslot:
            failed_edges = set(all_edges[:num_failures])
            link_errorlist.append(failed_edges)

        i = timeslot
        skip = True
        if received_message[i]:
            skip = False
            for neighbor in list(G.neighbors(i)) + list(G.predecessors(i)):
                if ((i, neighbor) not in failed_edges and (neighbor, i) not in failed_edges):
                    if allow_reception_from_all:
                        received_message[neighbor] = True
                        if message_status[neighbor] == 'none':
                            message_status[neighbor] = 'normal' if timeslot + 1 <= neighbor else 'missed'
                    else:
                        if message_status[i] == 'normal':
                            received_message[neighbor] = True
                            if message_status[neighbor] == 'none':
                                message_status[neighbor] = 'normal' if timeslot + 1 <= neighbor else 'missed'

        # Logging setup
        timeslot_info = f"Timeslot: {timeslot}"
        timeslot_skip = f"Skipped: {skip}"
        neighbors_info = f"Neighbors of Node {i}: {list(G.neighbors(i))}, Predecessors: {list(G.predecessors(i))}"
        received_message_info = f"Received Messages: {received_message}"
        failed_edges_info = f"Failed Edges: {failed_edges}"

        log_message += f"{timeslot_info}\n{timeslot_skip}\n{neighbors_info}\n{received_message_info}\n{failed_edges_info}\n"

    # if num_failures <= 3 and any(not item for item in received_message):
    #     print(f"Node error: {num_failures}")
    #     print_two_arrays_as_rows(received_message, message_status)
    #     print(link_errorlist)
    #     print(log_message)  # Or use logging.info(log_message) if using the logging library

    return received_message


def print_error_rates_per_node(num_simulations, error_rate_results, error_range):
    # Header for the table
    headers = ["Node #"] + [f"{failures} Failures" for failures in error_range]
    print(" | ".join(f"{header:>12}" for header in headers))
    print("-" * (14 * len(headers)))  # Print a separator line

    # Number of nodes is inferred from the length of the first list in error_rate_results
    num_nodes = len(error_rate_results[0])

    # Print error rates for each node
    for node_index in range(num_nodes):
        row = [f"Node {node_index}"]
        for failure_index in range(len(error_range)):
            error_rate = error_rate_results[failure_index][node_index] * 100  # Convert to percentage
            row.append(f"{error_rate:.5f}%")
        print(" | ".join(f"{item:>12}" for item in row))

def simulate_and_print_error_rates(G, num_nodes, num_simulations, error_range):
    error_rate_results = []
    for num_failures in error_range:
        successful_receptions = [0] * num_nodes
        for _ in range(num_simulations):
            received_message = simulate_broadcast_with_failures(G, num_failures)
            # Update successful receptions count
            for i, received in enumerate(received_message):
                if received:
                    successful_receptions[i] += 1

        # Calculate error rate for each node (proportion of failures to receive the message)
        error_rates = [(1 - (successes / num_simulations)) for successes in successful_receptions]
        error_rate_results.append(error_rates)

    # Print error rates per node with headers for different failure scenarios
    print_error_rates_per_node(num_simulations, error_rate_results, error_range)

    return error_rate_results


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Select a DAG by ID and set basic parameters.")
    parser.add_argument('netid', type=str, help='The ID of the DAG to use')
    parser.add_argument('runs', type=str, help='Number of simulations')
    return parser


if __name__ == "__main__":

    parser = setup_arg_parser()
    args = parser.parse_args()
    dag_id = args.netid

    # Define DAGs based on ID
    dags = {
        # network with stability factor 1
        'SF1': np.array([
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ]),
        # network with stability factor 2
        'SF2': np.array([
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ]),
        # network with stability factor 2 with nodes with higher redundancy
        'SF2_higher': np.array([
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ]),
        # network with stability factor 3
        'SF3': np.array([
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ]),
        # network with stability factor 4
        'SF4': np.array([
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]
        ])
    }

    dag = dags.get(dag_id, None)
    
    if dag is None:
        print("Invalid DAG ID provided. Please select a valid ID.")
        sys.exit(1)
    
    num_nodes = len(dag)
    print(f"Number of nodes: {num_nodes}")

    num_simulations = int(args.runs)
        
    print(f"Number of simulation runs: {num_simulations}")
    if num_simulations <= 1000:
        print("CAUTION! Number of simulation runs won't be representational! Recommended number of minimal simulation runs are 1000")

    stability_factor = 2
    errorstart = 1 # start simulation from 1 error per timeslot
    errorend = num_nodes # simulate link failures up to number of nodes 

    print("Selected DAG Matrix:")
    print(dag)
    
    # create agraph object
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)\

    # Run the simulation multiple times    
    error_rate_results = []
    error_range = range(errorstart, errorend)  # From 1 to 4 link failures
    
    error_rate_results = simulate_and_print_error_rates(G, num_nodes, num_simulations, error_range)

    # print the calculated stability factor
    print(f"Stability Factor: {calculate_stability_factor(G)}")

    # Prepare data for 3D plotting
    X, Y = np.meshgrid(error_range, range(num_nodes))
    Z = np.array(error_rate_results).T  # Transpose to match dimensions

    # Prepare the figure for both the network graph and the 3D plot
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Network graph visualization
    ax1 = fig.add_subplot(121)
    nx.draw_circular(G, with_labels=True, node_color='skyblue', edge_color='k')
    ax1.set_title("Network Graph")

    # Placeholder data for Subplot 2 due to reset: 3D plot of error rates
    # error_range = range(1, 5)  # From 1 to 4 link failures
    # X, Y = np.meshgrid(error_range, range(num_nodes))
    # Z = np.random.rand(num_nodes, len(error_range)) * 0.1  # Random small error rates for demonstration

    ax2 = fig.add_subplot(122, projection='3d')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.zaxis.set_major_locator(MaxNLocator(integer=True))

    surf = ax2.plot_surface(X, Y, Z, cmap='viridis')
    ax2.set_xlabel('Number of Link Failures')
    ax2.set_ylabel('Node ID')
    ax2.set_zlabel('Error Rate')
    ax2.set_title('Error Rates Across Nodes With Varying Number of Link Failures')

    # Show the combined figure
    plt.tight_layout()
    plt.show()

