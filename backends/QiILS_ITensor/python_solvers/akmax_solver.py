import dimod
from pyakmaxsat import AKMaxSATSolver
import networkx as nx
import os
import json

ROOT = "/Users/guillermo.preisser/Projects/QiILS_ITensor"

########################################################
# Load graph from file
########################################################
def load_graph(filepath, weighted=False):
    G = nx.Graph()
    with open(filepath, "r") as file:
        for line in file:
            weight_str, node1_str, node2_str = line.strip().split(",")
            node1, node2 = int(node1_str), int(node2_str)
            weight = float(weight_str) if weighted else 1.0
            G.add_edge(node1, node2, weight=weight)
    return G

########################################################
# Run AKMaxSAT solver and save optimal solution
########################################################
def run_solver(N, k, seed, weighted=True):
    graph_path = f"{ROOT}/graphs/{N}/{k}/"
    output_dir = f"{ROOT}/solutions/random_regular/{N}/{k}/"
    os.makedirs(output_dir, exist_ok=True)

    seedb = seed
    weight_status = "weighted" if weighted else "unweighted"

    # Graph filename MUST match Julia's new naming
    graph_filename = f"graph_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.txt"
    filepath = os.path.join(graph_path, graph_filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    # Load graph
    G = load_graph(filepath, weighted=weighted)

    # Total weight sum W = sum_{edges} w_ij (in unweighted mode this is m)
    W = sum(d["weight"] for _, _, d in G.edges(data=True))

    # Save W for debugging (optional)
    sumweights_file = os.path.join(
        output_dir, f"sumweights_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.txt"
    )
    with open(sumweights_file, "w") as f:
        f.write(str(W) + "\n")
    print(f"Saved total edge weight W={W} to {sumweights_file}")

    # Add quadratic weights for BQM
    for edge in G.edges:
        G.edges[edge]["quadratic"] = G.edges[edge]["weight"]

    # Build the BQM (note: dimod deprecation warning is OK for now)
    bqm = dimod.BinaryQuadraticModel.from_networkx_graph(
        G, vartype="SPIN", edge_attribute_name="quadratic"
    )

    # Run AKMaxSAT
    solver = AKMaxSATSolver()
    sampleset = solver.sample(bqm)

    energy_data = list(sampleset.data(fields=["sample", "energy"]))
    print(f"Number of samples returned: {len(energy_data)}")

    # Save FIRST sample (AKMaxSAT returns best first)
    if energy_data:
        first_energy = energy_data[0][1]
        first_cut = (W - first_energy) / 2

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
            "method": "AKMaxSAT",
            "graph_file": filepath,
        }

        print(f"Saving results to {out_file}")
        with open(out_file, "w") as json_file:
            json.dump(savedict, json_file, indent=4)
        print("File saved successfully!")

        return out_file, first_cut

    else:
        print("No energy data found. Nothing to save.")
        return None, None

########################################################
# Batch running
########################################################
if __name__ == "__main__":
    N = 70
    k = 3
    weighted = True    
    seeds = [1]

    for seed in seeds:
        print(f"\nRunning AKMaxSAT for N={N}, k={k}, seed={seed}, weighted={weighted}")
        run_solver(N, k, seed, weighted=weighted)