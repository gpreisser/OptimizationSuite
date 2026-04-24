using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activates QiILS root

using Revise
using QiILS
using JSON
using Printf
using Graphs
using Downloads

println("====================================================")
println("                QiILS Optimizer Runner              ")
println("====================================================")

# ---------------------------------------------------------
# User parameters (Gset)
# ---------------------------------------------------------
gset = 12
seed = 2                   # used for QiILS randomness (not the graph)
weighted = true            # ignored for Gset unless the file has weights

λ_sweep = 0.28
attempts = 20
sweeps_per_attempt = 80
percentage = 0.3
angle_conv = 0.1           # (not used by current qiils_solve methods you have loaded)

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
# Load known optimal cut (from QiILS/src/graphs/gset_solutions.jl)
# ---------------------------------------------------------
optimal_cut = get_optimal_cut(gset)
if optimal_cut === nothing
    println("⚠ No known optimal cut stored for G$(gset). Will skip ratio.")
else
    println("✔ Known optimal MaxCut value for G$(gset) = $optimal_cut")
end

# ---------------------------------------------------------
# Prepare gvec (currently required by qiils_solve)
# ---------------------------------------------------------
gvec = zeros(Float64, nv(wg))

# ---------------------------------------------------------
# Run solver
# ---------------------------------------------------------
println("\n▶ Running QiILS solver…")

# Your currently-defined methods are positional and start with:
# qiils_solve(wg, λ_sweep, gvec, attempts, ...)
best_history, best_angles, sweeps_cumsum = qiils_solve(
    wg,
    λ_sweep,
    gvec,
    attempts,
    sweeps_per_attempt,
    percentage,
    seed,
    nothing,     # θ0
    angle_conv,  # <-- your 0.1 will be used
    true,        # use_scaled_convergence
)

best_cut = best_history[end]

println("\n====================================================")
println("                 QiILS Solver Results               ")
println("====================================================")
println("  ✓ Best MaxCut found = $best_cut")

ratio = nothing
if optimal_cut !== nothing
    ratio = best_cut / optimal_cut
    println("  ✓ Approximation ratio = $(round(ratio, digits=5))")
    println("  ✓ total_sweeps_done = $(sweeps_cumsum[end])")
end

println("====================================================\n")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
weight_tag = weighted ? "weighted" : "unweighted"
save_dir = "results/Gset$(gset)_seed$(seed)_$(weight_tag)/"
mkpath(save_dir)

save_data = Dict(
    "gset" => gset,
    "graphfile" => graphfile,
    "seed" => seed,
    "weighted_flag" => weighted,
    "λ_sweep" => λ_sweep,
    "attempts" => attempts,
    "sweeps_per_attempt" => sweeps_per_attempt,
    "percentage" => percentage,
    "angle_conv" => angle_conv,  # recorded even if not used
    "best_cut_history" => best_history,
    "best_cut" => best_cut,
    "best_angles" => best_angles,
    "optimal_cut" => optimal_cut === nothing ? "none" : optimal_cut,
    "approx_ratio" => ratio === nothing ? "none" : ratio,
)

save_path = joinpath(save_dir, "qiils_results.json")
open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path\n")
println("Done.")