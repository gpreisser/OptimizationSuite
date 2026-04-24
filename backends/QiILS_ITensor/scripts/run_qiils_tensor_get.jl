######################## RUN QiILS_ITensor on Gset ########################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using JSON
using Printf
using Graphs
using Downloads
using QiILS_ITensor

########################## Helpers for saving #############################

function instance_dir_gset(gset)
    return joinpath("results", "gset", "G$(gset)")
end

function solver_filename(; attempts, sweeps_per_attempt, percentage,
                         maxdim, sample_mode, angle_conv)

    return "run_att$(attempts)_sw$(sweeps_per_attempt)_pct$(percentage)" *
           "_maxd$(maxdim)_conv$(angle_conv)_mode$(sample_mode).json"
end

############################ Header #######################################

println("====================================================")
println("             QiILS_ITensor Gset Runner              ")
println("====================================================")

########################### User parameters ###############################

gset = 12
weighted = true

λ_sweep = 0.28
attempts = 1
sweeps_per_attempt = 80
percentage = 0.3
angle_conv = 1e-8
maxdim = 2
sample_mode = :local

########################### Load graph ####################################

gset_dir = joinpath(@__DIR__, "..", "graphs", "gset")
mkpath(gset_dir)

gset_path = joinpath(gset_dir, "G$(gset)")
if !isfile(gset_path)
    println("▶ Downloading G$(gset) …")
    Downloads.download("https://web.stanford.edu/~yyye/yyye/Gset/G$(gset)", gset_path)
else
    println("▶ Found local Gset graph: $gset_path")
end

println("▶ Loading Gset graph G$(gset) from: $gset_path")
wg = load_graph(path=gset_path)
graphfile = gset_path
println("✔ Loaded graph with N = $(nv(wg)), M = $(ne(wg)) edges")

########################### Load optimal solution ##########################

optimal_cut = get_gset_optimal_cut(gset)
if optimal_cut === nothing
    println("⚠ No known optimal cut stored for G$(gset).")
else
    println("✔ Known optimal MaxCut value for G$(gset) = $optimal_cut")
end

########################### Run solver ####################################

println("\n▶ Running QiILS_ITensor solver…")

best_history, cut_history, best_spins, energy_history = qiils_itensor_solver(
    wg;
    lambda_sweep       = λ_sweep,
    attempts           = attempts,
    sweeps_per_attempt = sweeps_per_attempt,
    maxdim             = maxdim,
    percentage         = percentage,
    sample_mode        = sample_mode,
    weighted           = weighted,
)

best_cut = best_history[end]

ratio = nothing
if optimal_cut !== nothing
    ratio = best_cut / optimal_cut
end

println("\n====================================================")
println("                 Solver Results                     ")
println("====================================================")
println("  ✓ Best MaxCut = $best_cut")
if ratio !== nothing
    println("  ✓ Approx ratio = $(round(ratio, digits=5))")
end
println("====================================================\n")

########################### Save results ##################################

save_base = instance_dir_gset(gset)
mkpath(save_base)

fname = solver_filename(
    attempts=attempts,
    sweeps_per_attempt=sweeps_per_attempt,
    percentage=percentage,
    maxdim=maxdim,
    sample_mode=sample_mode,
    angle_conv=angle_conv,
)

save_path = joinpath(save_base, fname)

save_data = Dict(
    "instance_type" => "gset",
    "gset" => gset,
    "graphfile" => graphfile,
    "weighted" => weighted,
    "λ_sweep" => λ_sweep,
    "attempts" => attempts,
    "sweeps_per_attempt" => sweeps_per_attempt,
    "percentage" => percentage,
    "angle_conv" => angle_conv,
    "maxdim" => maxdim,
    "sample_mode" => String(sample_mode),
    "best_history" => best_history,
    "cut_history" => cut_history,
    "energy_history" => energy_history,
    "best_cut" => best_cut,
    "best_spins" => best_spins,
    "optimal_cut" => optimal_cut,
    "approx_ratio" => ratio,
)

open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path")
println("Done.")