using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activates QiILS root

using Revise
using QiILS
using JSON
using Printf
using Graphs
using Downloads

println("====================================================")
println("                 SA Optimizer Runner                ")
println("====================================================")

# ---------------------------------------------------------
# User parameters (Gset)
# ---------------------------------------------------------
gset = 12
seed = 2                   # SA RNG seed (not the graph)
weighted = true            # just a tag for the output folder

β0 = 0.01
βf = 5.0
sweeps = 20_000
schedule = :linear         # :linear or :geometric

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
# Run SA
# ---------------------------------------------------------
println("\n▶ Running SA solver…")

best_cut, best_spins, best_hist, sweeps_done = sa_maxcut(
    wg;
    β0=β0,
    βf=βf,
    sweeps=sweeps,
    seed=seed,
    schedule=schedule,
)

println("\n====================================================")
println("                  SA Solver Results                 ")
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
save_dir = "results/SA_Gset$(gset)_seed$(seed)_$(weight_tag)/"
mkpath(save_dir)

save_data = Dict(
    "solver" => "SA",
    "gset" => gset,
    "graphfile" => graphfile,
    "seed" => seed,
    "weighted_flag" => weighted,
    "beta0" => β0,
    "betaf" => βf,
    "schedule" => String(schedule),
    "sweeps_requested" => sweeps,
    "sweeps_done" => sweeps_done,
    "best_cut_history" => best_hist,
    "best_cut" => best_cut,
    "optimal_cut" => optimal_cut === nothing ? "none" : optimal_cut,
    "approx_ratio" => ratio === nothing ? "none" : ratio,
)

save_path = joinpath(save_dir, "sa_results.json")
open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path\n")
println("Done.")