# scripts/run_tabu_gset.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activates QiILS root

using Revise
using QiILS
using JSON
using Printf
using Graphs
using Downloads

println("====================================================")
println("               Tabu Optimizer Runner                ")
println("====================================================")

# ---------------------------------------------------------
# User parameters (Gset)
# ---------------------------------------------------------
gset = 12
seed = 2                   # Tabu RNG seed (not the graph)
weighted = true            # just a tag for output folder

sweeps = 5_000            # 1 sweep = N tabu steps (N flips)
tenure = 80                # tabu tenure in steps
candidate_size = 128         # 0 = evaluate all vertices each step
verbose = false             # print when new best is found

# ---------------------------------------------------------
# Download + load Gset graph
# ---------------------------------------------------------
gset_dir  = joinpath(@__DIR__, "..", "graphs", "gset")
mkpath(gset_dir)

gset_path = joinpath(gset_dir, "G$(gset)")
if !isfile(gset_path)
    println("▶ Downloading G$(gset) …")
    Downloads.download("https://web.stanford.edu/~yyye/yyye/Gset/G$(gset)", gset_path)
end

println("▶ Loading Gset graph G$(gset) from: $gset_path")
wg = load_graph(path=gset_path)
graphfile = gset_path
println("✔ Loaded graph with N = $(nv(wg)), M = $(ne(wg)) edges")

# ---------------------------------------------------------
# Load known optimal cut
# ---------------------------------------------------------
optimal_cut = get_optimal_cut(gset)
if optimal_cut === nothing
    println("⚠ No known optimal cut stored for G$(gset). Will skip ratio.")
else
    println("✔ Known optimal MaxCut value for G$(gset) = $optimal_cut")
end

# ---------------------------------------------------------
# Run Tabu
# ---------------------------------------------------------
println("\n▶ Running Tabu solver…")

best_cut, best_spins, best_hist, sweeps_done = tabu_maxcut(
    wg;
    sweeps=sweeps,
    tenure=tenure,
    candidate_size=candidate_size,
    seed=seed,
    verbose=verbose,
)

println("\n====================================================")
println("                Tabu Solver Results                 ")
println("====================================================")
println("  ✓ Best MaxCut found = $best_cut")
println("  ✓ Sweeps done = $sweeps_done")

ratio = nothing
if optimal_cut !== nothing
    ratio = best_cut / optimal_cut
    println("  ✓ Approximation ratio = $(round(ratio, digits=5))")
end
println("====================================================\n")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
weight_tag = weighted ? "weighted" : "unweighted"
save_dir = "results/Tabu_Gset$(gset)_seed$(seed)_$(weight_tag)/"
mkpath(save_dir)

save_data = Dict(
    "solver" => "Tabu",
    "gset" => gset,
    "graphfile" => graphfile,
    "seed" => seed,
    "weighted_flag" => weighted,
    "sweeps_requested" => sweeps,
    "sweeps_done" => sweeps_done,
    "tenure" => tenure,
    "candidate_size" => candidate_size,
    "best_cut_history" => best_hist,
    "best_cut" => best_cut,
    "optimal_cut" => optimal_cut === nothing ? "none" : optimal_cut,
    "approx_ratio" => ratio === nothing ? "none" : ratio,
)

save_path = joinpath(save_dir, "tabu_results.json")
open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path\n")
println("Done.")