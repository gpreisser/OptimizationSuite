import os
import json
from ortools.sat.python import cp_model


def load_graph_edges(filepath, weighted=False):
    """
    Reads your graph file lines:
      weight,node1,node2
    Nodes are assumed to be integers (as written in your files).
    Returns:
      edges: list of (u, v, w)
      nodes: sorted list of node ids
    """
    edges = []
    nodeset = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w_str, u_str, v_str = line.split(",")
            u, v = int(u_str), int(v_str)
            w = float(w_str) if weighted else 1.0
            edges.append((u, v, w))
            nodeset.add(u)
            nodeset.add(v)
    nodes = sorted(nodeset)
    return edges, nodes


def solve_maxcut_cp_sat(edges, nodes, time_limit_s=60.0, num_workers=8):
    """
    Exact MaxCut via CP-SAT.
    Variables:
      x_i ∈ {0,1} indicating partition side
      y_e ∈ {0,1} indicating edge is cut (x_u != x_v)
    Maximize sum w_e * y_e
    """
    model = cp_model.CpModel()

    # binary partition vars
    x = {i: model.NewBoolVar(f"x_{i}") for i in nodes}

    # cut indicator vars
    y = []
    weights = []
    for (u, v, w) in edges:
        y_e = model.NewBoolVar(f"y_{u}_{v}")
        # y_e == XOR(x_u, x_v)
        model.Add(x[u] + x[v] == 1).OnlyEnforceIf(y_e)
        model.Add(x[u] == x[v]).OnlyEnforceIf(y_e.Not())
        y.append(y_e)
        weights.append(int(round(w)))  # CP-SAT objective uses integers robustly

    # Objective: maximize sum w * y
    model.Maximize(sum(weights[i] * y[i] for i in range(len(y))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status={status}")

    best_cut = solver.ObjectiveValue()

    return best_cut, status


def run_batch(N, k, seeds, weighted=False, time_limit_s=60.0, num_workers=8):
    base_dir = "/Users/guillermo.preisser/Projects/QiILS"
    graph_dir = os.path.join(base_dir, "graphs", str(N), str(k))
    out_dir = os.path.join(base_dir, "solutions", "random_regular", str(N), str(k))
    os.makedirs(out_dir, exist_ok=True)

    weight_status = "weighted" if weighted else "unweighted"

    for seed in seeds:
        seedb = seed
        filename = f"graph_N{N}_k{k}_seed{seed}_seedb{seedb}.txt"
        filepath = os.path.join(graph_dir, filename)
        if not os.path.isfile(filepath):
            print(f"[skip] missing graph file: {filepath}")
            continue

        edges, nodes = load_graph_edges(filepath, weighted=weighted)
        W = sum(w for (_, _, w) in edges)

        best_cut, status = solve_maxcut_cp_sat(
            edges, nodes, time_limit_s=time_limit_s, num_workers=num_workers
        )

        # Save JSON in your expected folder
        out_file = os.path.join(
            out_dir, f"ortools_maxcut_N{N}_k{k}_seed{seed}_seedb{seedb}_{weight_status}.json"
        )
        savedict = {
            "N": N,
            "k": k,
            "seed": seed,
            "seedb": seedb,
            "weighted": weighted,
            "W": W,
            "maxcut_value": best_cut,
            "method": "OR-Tools CP-SAT",
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
            "time_limit_s": time_limit_s,
        }

        with open(out_file, "w") as f:
            json.dump(savedict, f, indent=2)

        print(f"[ok] seed={seed}  maxcut={best_cut}  saved -> {out_file}")


if __name__ == "__main__":
    # EDIT THESE:
    N = 50
    k = 3
    weighted = False
    seeds = range(1, 101)   # 1..100

    # CP-SAT knobs:
    time_limit_s = 30.0
    num_workers = 8

    run_batch(N, k, seeds, weighted=weighted, time_limit_s=time_limit_s, num_workers=num_workers)