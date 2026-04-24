# solver_weighted_maxcut_ilp.py
import os
import json
import networkx as nx
import pulp

ROOT = "/Users/guillermo.preisser/Projects/QiILS_ITensor"

def load_graph(filepath: str, weighted: bool = False, expected_N: int | None = None) -> nx.Graph:
    """
    Loads a graph from a text file.

    Supported line formats (comma-separated):
      1) w,u,v   (weight first)  e.g. 0.0240943,1,2
      2) u,v     (unweighted)    e.g. 1,2

    If weighted=False:
      - w,u,v lines are accepted but weights are ignored (set to 1.0)
      - u,v lines are accepted (weight=1.0)

    If weighted=True:
      - w,u,v lines are required (otherwise error)

    expected_N (optional):
      - sanity-check node labels are either {1..N} or {0..N-1}
    """
    G = nx.Graph()

    with open(filepath, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if len(parts) == 3:
                w_str, u_str, v_str = parts
                u, v = int(u_str), int(v_str)
                w = float(w_str) if weighted else 1.0
            elif len(parts) == 2:
                if weighted:
                    raise ValueError(
                        f"File appears unweighted (u,v) but weighted=True. Line: '{s}'"
                    )
                u_str, v_str = parts
                u, v = int(u_str), int(v_str)
                w = 1.0
            else:
                raise ValueError(f"Bad line format (expected 2 or 3 fields): '{s}'")

            G.add_edge(u, v, weight=w)

    if expected_N is not None:
        nodes = sorted(G.nodes())
        if not nodes:
            raise ValueError(f"Graph has no nodes: {filepath}")

        min_node, max_node = nodes[0], nodes[-1]
        ok_1_based = (min_node == 1 and max_node == expected_N)
        ok_0_based = (min_node == 0 and max_node == expected_N - 1)

        if not (ok_1_based or ok_0_based):
            raise ValueError(
                f"Unexpected node labels in {filepath}: min={min_node}, max={max_node}, expected_N={expected_N}. "
                f"Expected either 1..N or 0..N-1."
            )

    return G

def solve_maxcut_ilp(G: nx.Graph()):
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))

    # Decision: x_i ∈ {0,1} which side of cut
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in nodes}

    # For each edge (i,j), y_ij = 1 if cut (x_i != x_j)
    y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for (i, j, _) in edges}

    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)

    # Objective: maximize sum w_ij * y_ij
    prob += pulp.lpSum(d["weight"] * y[(i, j)] for (i, j, d) in edges)

    # Linearization of XOR:
    # y >= x_i - x_j
    # y >= x_j - x_i
    # y <= x_i + x_j
    # y <= 2 - (x_i + x_j)
    for (i, j, _) in edges:
        yij = y[(i, j)]
        prob += yij >= x[i] - x[j]
        prob += yij >= x[j] - x[i]
        prob += yij <= x[i] + x[j]
        prob += yij <= 2 - (x[i] + x[j])

    # Solve (CBC bundled with PuLP in most installs)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"ILP did not reach optimality. Status={pulp.LpStatus[prob.status]}")

    maxcut = float(pulp.value(prob.objective))

    # Total weight W
    W = float(sum(d["weight"] for (_, _, d) in edges))

    # Map to an "Ising energy" consistent with cut = (W - Eising)/2
    ising_energy = W - 2.0 * maxcut

    return maxcut, ising_energy, W

def run_solver(N: int, k: int, seed: int, weighted: bool = True):
    graph_dir = os.path.join(ROOT, "graphs", str(N), str(k))
    output_dir = os.path.join(ROOT, "solutions", "random_regular", str(N), str(k))
    os.makedirs(output_dir, exist_ok=True)

    seedb = seed
    weight_status = "weighted" if weighted else "unweighted"
    graph_filename = f"graph_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.txt"
    filepath = os.path.join(graph_dir, graph_filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    G = load_graph(filepath, weighted=weighted, expected_N=N)

    maxcut_value, ising_energy, W = solve_maxcut_ilp(G)

    out_file = os.path.join(
        output_dir,
        f"akmaxdata_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.json"
    )

    savedict = {
        "N": N,
        "k": k,
        "seed": seed,
        "seedb": seedb,
        "weighted": weighted,
        "W": W,
        "ising_energy": ising_energy,
        "maxcut_value": maxcut_value,
        "method": "PuLP_CBC_ILP",
        "graph_file": filepath,
    }

    with open(out_file, "w") as jf:
        json.dump(savedict, jf, indent=4)

    print(f"Saved optimal solution to {out_file}")
    return out_file, maxcut_value

if __name__ == "__main__":
    N = 50
    k = 3
    weighted = True  # set True for weighted graphs

    for seed in range(1, 101):  # 1 to 500 inclusive
        print(f"Running seed {seed} ({'weighted' if weighted else 'unweighted'})")
        run_solver(N, k, seed, weighted=weighted)