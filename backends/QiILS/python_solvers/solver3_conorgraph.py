# solver_pm1weighted_maxcut_ilp.py
import os
import json
import networkx as nx
import pulp

ROOT = "/Users/guillermo.preisser/Projects/QiILS_ITensor"

def load_graph(filepath: str, expected_N: int | None = None) -> nx.Graph:
    """
    Loads a ±1-weighted graph from a text file.

    Expected line format:
      w,u,v

    where w should be ±1 (stored as float/int), and nodes are either
    1..N or 0..N-1.
    """
    G = nx.Graph()

    with open(filepath, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if len(parts) != 3:
                raise ValueError(f"Bad line format (expected w,u,v): '{s}'")

            w_str, u_str, v_str = parts
            u, v = int(u_str), int(v_str)
            w = float(w_str)

            if w not in (-1.0, 1.0):
                raise ValueError(f"Expected ±1 weight, got {w} in line: '{s}'")

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


def solve_maxcut_ilp(G: nx.Graph):
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in nodes}
    y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for (i, j, _) in edges}

    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    prob += pulp.lpSum(d["weight"] * y[(i, j)] for (i, j, d) in edges)

    for (i, j, _) in edges:
        yij = y[(i, j)]
        prob += yij >= x[i] - x[j]
        prob += yij >= x[j] - x[i]
        prob += yij <= x[i] + x[j]
        prob += yij <= 2 - (x[i] + x[j])

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"ILP did not reach optimality. Status={pulp.LpStatus[prob.status]}")

    maxcut = float(pulp.value(prob.objective))
    W = float(sum(d["weight"] for (_, _, d) in edges))
    ising_energy = W - 2.0 * maxcut

    return maxcut, ising_energy, W


def run_solver(N: int, k: int, seed: int):
    graph_dir = os.path.join(ROOT, "graphs", str(N), str(k))
    output_dir = os.path.join(ROOT, "solutions", "random_regular_pm1weighted", str(N), str(k))
    os.makedirs(output_dir, exist_ok=True)

    seedb = seed
    tag = "pm1weighted"

    graph_filename = f"graph_N{N}_k{k}_seed{seed}_seedb{seedb}_{tag}.txt"
    filepath = os.path.join(graph_dir, graph_filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    G = load_graph(filepath, expected_N=N)
    maxcut_value, ising_energy, W = solve_maxcut_ilp(G)

    out_file = os.path.join(
        output_dir,
        f"akmaxdata_N{N}_k{k}_seed{seed}_seedb{seedb}_{tag}.json"
    )

    savedict = {
        "N": N,
        "k": k,
        "seed": seed,
        "seedb": seedb,
        "graph_tag": tag,
        "weighted": True,
        "weight_type": "pm1",
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
    N = 12
    k = 3

    for seed in range(2, 10):
        print(f"Running seed {seed} (pm1weighted)")
        run_solver(N, k, seed)