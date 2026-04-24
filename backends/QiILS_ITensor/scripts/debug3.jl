using Pkg
Pkg.activate("/Users/guillermo.preisser/Projects/QiILS_ITensor")

using QiILS_ITensor
using Graphs
using SimpleWeightedGraphs  # This is what you need for weighted graphs

# Create a tiny test weighted graph
wg = SimpleWeightedGraph(3)
add_edge!(wg, 1, 2, 1.0)
add_edge!(wg, 2, 3, 1.0)
add_edge!(wg, 1, 3, 1.0)

println("Testing solver with 2 attempts...")
best_hist, cut_hist, best_spins = qiils_itensor_solver(
    wg;
    lambda_sweep = 1.0,
    attempts = 2,
    sweeps_per_attempt = 5,
    maxdim = 2,
    percentage = 0.2,
    weighted = false
)

println("\nResults:")
println("best_history: ", best_hist)
println("cut_history:  ", cut_hist)

if length(cut_hist) == 2
    println("\n✓ cut_history is working! Length = $(length(cut_hist))")
else
    println("\n✗ Problem: cut_history length = $(length(cut_hist)), expected 2")
end