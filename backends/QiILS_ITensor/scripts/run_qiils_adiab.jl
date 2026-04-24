############################ RUN QiILS_ITensor (λ sweep) ############################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using JSON
using Printf
using Graphs
using QiILS_ITensor   # your package

########################## Helpers for saving #############################

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath("results", "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

function sweep_filename(; L, sweeps_per_attempt, percentage, maxdim, sample_mode)
    return "sweep_L$(L)_sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode).json"
end

########################## λ-sweep runner (FUNCTION to avoid scope bugs) #############################

function run_lambda_sweep_continuation(wg, optimal_cut;
    L::Int,
    sweeps_per_attempt::Int,
    maxdim::Int,
    percentage::Float64,
    sample_mode::Symbol,
)
    N = nv(wg)
    hilbert = siteinds("Qubit", N)  # ONE set of indices for the whole sweep

    λs   = collect(range(0.0, 1.0; length=L))
    cuts = Vector{Float64}(undef, L)

    psi = nothing
    best_spins_at_λ1 = nothing
    best_cut_so_far = -Inf

    for (i, λ) in enumerate(λs)
        hist, spins, psi, hilbert = qiils_itensor_solver(
            wg;
            lambda_sweep       = λ,
            attempts           = 1,   # ONE attempt per λ
            sweeps_per_attempt = sweeps_per_attempt,
            maxdim             = maxdim,
            percentage         = percentage,
            sample_mode        = sample_mode,
            init_psi           = psi,
            return_psi         = true,
            hilbert            = hilbert
        )

        current_cut = hist[end]
        best_cut_so_far = max(best_cut_so_far, current_cut)

        # store "best known up to current λ"
        cuts[i] = best_cut_so_far

        # print approximation ratios (current and best-so-far)
        r_current = current_cut / optimal_cut
        r_best    = best_cut_so_far / optimal_cut

        @printf(
            "λ = %.4f (%d/%d) | r(current) = %.6f | r(best so far) = %.6f\n",
            λ, i, L, r_current, r_best
        )

        if i == L
            best_spins_at_λ1 = spins
        end
    end

    ratios = cuts ./ optimal_cut
    one_minus_r = 1 .- ratios

    @printf("\n✓ Best MaxCut up to λ=1 = %.6f\n", best_cut_so_far)
    @printf("✓ Approx ratio (best)   = %.6f\n", best_cut_so_far / optimal_cut)
    @printf("✓ 1 - r (best)          = %.6f\n\n", 1 - best_cut_so_far / optimal_cut)

    return λs, cuts, ratios, one_minus_r, best_spins_at_λ1
end

############################ Header ########################################

println("====================================================")
println("           QiILS_ITensor Runner (λ sweep)           ")
println("====================================================")

########################### User parameters ################################

N        = 30
k        = 3
seed     = 1
weighted = false  # set to false for unweighted (weights forced to 1 below)

# λ sweep settings
L = 21  # number of λ points (and total attempts in your scheme)

# solver settings (fixed)
sweeps_per_attempt = 80
percentage = 0.3
maxdim = 1
sample_mode = :local

########################### Load graph #####################################

println("▶ Loading graph N=$N, k=$k, seed=$seed (weighted=$weighted)")
wg, graphfile = create_and_save_graph_QiILS(
    N, k, seed;
    weighted = weighted,
    base_path = "/Users/guillermo.preisser/Projects/QiILS_ITensor/graphs"
)
println("Graph saved at: $graphfile")


# If unweighted experiment requested, FORCE all weights to 1.0 in-memory.
# (The file written by create_and_save_graph_QiILS may still contain random weights;
#  but solver + evaluation will now be unweighted consistently.)


########################### Load optimal solution ###########################

solution_path = solution_file_path(N, k, seed; weighted=weighted)

println("---- OPTIMAL SOLUTION CHECK ----")
println("Expecting solution JSON at:")
println(solution_path)
println("Exists? ", isfile(solution_path))
println("--------------------------------")

optimal_cut = load_optimal_cut(solution_path)
println("Loaded optimal_cut = ", optimal_cut)

if optimal_cut === nothing
    error("Need optimal_cut to compute 1-r(λ). (Run the Python solver to create the JSON.)")
end

########################### Run λ-sweep continuation ########################

λs, cuts, ratios, one_minus_r, best_spins_at_λ1 = run_lambda_sweep_continuation(
    wg, optimal_cut;
    L=L,
    sweeps_per_attempt=sweeps_per_attempt,
    maxdim=maxdim,
    percentage=percentage,
    sample_mode=sample_mode
)

println("\n====================================================")
println("                 Sweep Results                      ")
println("====================================================")
println("  ✓ Best MaxCut up to λ=1 = $(cuts[end])")
println("  ✓ Approx ratio (best)  = $(round(ratios[end], digits=6))")
println("  ✓ 1 - r (best)         = $(round(one_minus_r[end], digits=6))")
println("====================================================\n")

########################### Save results ###################################

save_base = instance_dir(N, k, seed, weighted)
mkpath(save_base)

fname = sweep_filename(
    L=L,
    sweeps_per_attempt=sweeps_per_attempt,
    percentage=percentage,
    maxdim=maxdim,
    sample_mode=sample_mode,
)

save_path = joinpath(save_base, fname)

save_data = Dict(
    "experiment" => "lambda_sweep_continuation_one_attempt_per_lambda_best_so_far",
    "N" => N,
    "k" => k,
    "seed" => seed,
    "weighted" => weighted,
    "graphfile" => graphfile,

    "L" => L,
    "lambdas" => λs,

    "sweeps_per_attempt" => sweeps_per_attempt,
    "percentage" => percentage,
    "maxdim" => maxdim,
    "sample_mode" => String(sample_mode),

    "optimal_cut" => optimal_cut,
    "cuts_best_so_far" => cuts,
    "ratios_best_so_far" => ratios,
    "one_minus_r_best_so_far" => one_minus_r,

    "best_cut_up_to_lambda1" => cuts[end],
    "best_spins_lambda1" => best_spins_at_λ1,
)

open(save_path, "w") do io
    JSON.print(io, save_data)
end

println("✔ Results saved to: $save_path")
println("Done.")