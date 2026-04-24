############################ RUN QiILS_ITensor ############################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using JSON
using Printf
using Graphs
using QiILS_ITensor   # correct package

########################## Helpers for saving #############################

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath("results", "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

function solver_filename(; attempts, sweeps_per_attempt, percentage,
                         maxdim, sample_mode, angle_conv)

    return "run_att$(attempts)_sw$(sweeps_per_attempt)_pct$(percentage)" *
           "_maxd$(maxdim)_conv$(angle_conv)_mode$(sample_mode).json"
end


############################ Header ########################################

println("====================================================")
println("                QiILS_ITensor Runner                ")
println("====================================================")

########################### User parameters ################################

N        = 10
k        = 3
seed     = 2
weighted = true

λ_sweep = 0.5
attempts = 100
sweeps_per_attempt = 80
percentage = 0.3
angle_conv = 1e-8
maxdim = 2
sample_mode = :local   # important for filename

########################### Load graph #####################################

println("▶ Loading graph N=$N, k=$k, seed=$seed (weighted=$weighted)")
wg, graphfile = create_and_save_graph_QiILS(N, k, seed)
println("Graph saved at: $graphfile")

########################### Load optimal solution ###########################

solution_path = solution_file_path(N, k, seed; weighted=weighted)
optimal_cut   = load_optimal_cut(solution_path)

if optimal_cut === nothing
    println("⚠ No Python optimal solution available.")
else
    println("✔ Optimal MaxCut = $optimal_cut")
end

########################### Run solver #####################################

println("\n▶ Running QiILS_ITensor solver…")

best_history, best_spins = qiils_itensor_solver(
    wg;
    lambda_sweep       = λ_sweep,
    attempts           = attempts,
    sweeps_per_attempt = sweeps_per_attempt,
    maxdim             = maxdim,
    percentage         = percentage,
    sample_mode        = sample_mode,
)

best_cut = best_history[end]

println("\n====================================================")
println("                 Solver Results                     ")
println("====================================================")
println("  ✓ Best MaxCut = $best_cut")

if optimal_cut !== nothing
    ratio = best_cut / optimal_cut
    println("  ✓ Approx ratio = $(round(ratio, digits=5))")
end

println("====================================================\n")

########################### Save results ###################################

save_base = instance_dir(N, k, seed, weighted)
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
    "N" => N,
    "k" => k,
    "seed" => seed,
    "weighted" => weighted,
    "λ_sweep" => λ_sweep,
    "attempts" => attempts,
    "sweeps_per_attempt" => sweeps_per_attempt,
    "percentage" => percentage,
    "angle_conv" => angle_conv,
    "maxdim" => maxdim,
    "sample_mode" => String(sample_mode),
    "best_history" => best_history,
    "best_cut" => best_cut,
    "best_spins" => best_spins,
    "optimal_cut" => optimal_cut,
)

open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path")
println("Done.")