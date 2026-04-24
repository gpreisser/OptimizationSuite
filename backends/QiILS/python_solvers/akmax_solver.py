import dimod
from pyakmaxsat import AKMaxSATSolver
import networkx as nx
import os
import json

########################################################
# Load graph from file
########################################################
def load_graph(filepath, weighted=False):
    G = nx.Graph()
    with open(filepath, 'r') as file:
        for line in file:
            weight_str, node1_str, node2_str = line.strip().split(',')
            node1, node2 = int(node1_str), int(node2_str)
            weight = float(weight_str) if weighted else 1.0
            G.add_edge(node1, node2, weight=weight)
    return G


########################################################
# Run AKMaxSAT solver and save optimal solution
########################################################
def run_solver(N, k, seed, weighted=True):
    # Where your graph files live
    graph_path = f"/Users/guillermo.preisser/Projects/QiILS/graphs/{N}/{k}/"

    # Where we save results INSIDE THE QiILS PACKAGE
    output_dir = f"/Users/guillermo.preisser/Projects/QiILS/solutions/random_regular/{N}/{k}/"
    os.makedirs(output_dir, exist_ok=True)

    seedb = seed  # (Your original code used seedb = seed)
    filename = f"graph_N{N}_k{k}_seed{seed}_seedb{seedb}.txt"
    filepath = os.path.join(graph_path, filename)

    # Load the graph
    G = load_graph(filepath, weighted=weighted)

    # Total weight sum W = sum_{edges} w_ij
    W = sum(d['weight'] for _, _, d in G.edges(data=True))

    # Save W for debugging (optional)
    sumweights_file = os.path.join(
        output_dir, f"sumweights_N{N}_k{k}_seed{seed}_seedb{seedb}.txt"
    )
    with open(sumweights_file, "w") as f:
        f.write(str(W) + "\n")
    print(f"Saved total edge weight W={W} to {sumweights_file}")

    # Add quadratic weights for BQM
    for edge in G.edges:
        G.edges[edge]['quadratic'] = G.edges[edge]['weight']

    # Build the BQM
    bqm = dimod.BinaryQuadraticModel.from_networkx_graph(
        G, vartype='SPIN', edge_attribute_name='quadratic'
    )

    # Run AKMaxSAT
    solver = AKMaxSATSolver()
    sampleset = solver.sample(bqm)

    # Extract samples & energies
    energy_data = list(sampleset.data(fields=['sample', 'energy']))
    print(f"Number of samples returned: {len(energy_data)}")

    # Print all solutions (debug)
    for sample, E in energy_data:
        cut_value = (W - E) / 2
        S0 = [node for node, val in sample.items() if val == -1]
        S1 = [node for node, val in sample.items() if val == 1]

        print(f"S0: {S0}")
        print(f"S1: {S1}")
        print(f"Ising Energy: {E}")
        print(f"Computed MaxCut value: {cut_value}")
        print("-" * 40)

    ########################################################
    # Save FIRST sample (AKMaxSAT returns best first)
    ########################################################
    if energy_data:
        first_energy = energy_data[0][1]
        first_cut = (W - first_energy) / 2

        weight_status = "weighted" if weighted else "unweighted"
        out_file = os.path.join(
            output_dir,
            f"akmaxdata_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.json"
        )

        savedict = {
            "N": N,
            "k": k,
            "seed": seed,
            "weighted": weighted,
            "W": W,
            "ising_energy": first_energy,
            "maxcut_value": first_cut,
            "method": "AKMaxSAT"
        }

        print(f"Saving results to {out_file}")
        with open(out_file, "w") as json_file:
            json.dump(savedict, json_file, indent=4)
        print("File saved successfully!")

    else:
        print("No energy data found. Nothing to save.")


########################################################
# Batch running
########################################################
if __name__ == "__main__":
    seeds = range(1 , 2)   # change as needed
    N = 50
    k = 3
    weighted = False

    for seed in seeds:
        print(f"\nRunning AKMaxSAT for N={N}, k={k}, seed={seed}, weighted={weighted}")
        run_solver(N, k, seed, weighted=weighted)

