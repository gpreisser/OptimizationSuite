# scripts/run_tabu_standard_gset_tenures.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using QiILS
using JSON
using Printf
using Graphs
using Downloads
using SimpleWeightedGraphs

println("============================================================")
println(" Standard Tabu (full scan) Runner over Tenures (subsampled) ")
println("============================================================")

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
const ROOT        = normpath(joinpath(@__DIR__, ".."))
const GRAPHS_DIR  = joinpath(ROOT, "graphs", "gset")
const RESULTS_DIR = joinpath(ROOT, "results")

mkpath(GRAPHS_DIR)
mkpath(RESULTS_DIR)

# ---------------------------------------------------------
# User parameters
# ---------------------------------------------------------
gset = 12
weighted = true   # just a tag for output folder

ntrials = 100
base_seed = 1000

# Convention here:
# 1 sweep = 1 move after evaluating all N sites
sweeps = 10_000_000

verbose = false

tenures = [20, 50, 100, 150, 200, 250]

save_checkpoint_every_trial = true

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
function safe_sample_std(sumx::AbstractVector{T}, sumx2::AbstractVector{T}, n::Int) where {T<:Real}
    if n <= 1
        return fill(NaN, length(sumx))
    end
    μ = sumx ./ n
    var = (sumx2 .- n .* (μ .^ 2)) ./ (n - 1)
    var .= max.(var, 0.0)
    return sqrt.(var)
end

function safe_sample_std(sumx::Real, sumx2::Real, n::Int)
    if n <= 1
        return NaN
    end
    μ = sumx / n
    var = (sumx2 - n * μ^2) / (n - 1)
    var = max(var, 0.0)
    return sqrt(var)
end

function make_log_save_indices(n::Int)
    idx = Int[]

    # 1,2,...,9
    for k in 1:min(9, n)
        push!(idx, k)
    end

    # 10,20,...,90 ; 100,200,...,900 ; ...
    base = 10
    while base <= n
        for m in 1:9
            k = m * base
            k <= n && push!(idx, k)
        end
        base *= 10
    end

    push!(idx, n)

    sort!(idx)
    unique!(idx)
    return idx
end

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, JSON.json(data; allownan=true))
    end
    mv(tmp, path; force=true)
    return nothing
end

# ---------------------------------------------------------
# Download + load Gset graph
# ---------------------------------------------------------
gset_path = joinpath(GRAPHS_DIR, "G$(gset)")
if !isfile(gset_path)
    println("▶ Downloading G$(gset) …")
    Downloads.download("https://web.stanford.edu/~yyye/yyye/Gset/G$(gset)", gset_path)
end

println("▶ Loading Gset graph G$(gset) from: $gset_path")
W, N = load_graph(gset_path)
wg = SimpleWeightedGraph(W)
graphfile = gset_path
println("✔ Loaded graph with N = $(nv(wg)), M = $(ne(wg)) edges")

# ---------------------------------------------------------
# Load known optimal cut
# ---------------------------------------------------------
optimal_cut = get_optimal_cut(gset)
if optimal_cut === nothing
    error("No known optimal cut stored for G$(gset). Cannot compute ratio/error history.")
else
    println("✔ Known optimal MaxCut value for G$(gset) = $optimal_cut")
end

# ---------------------------------------------------------
# History subsampling (for log-x plotting)
# ---------------------------------------------------------
save_idx = make_log_save_indices(sweeps)
nsaved = length(save_idx)

println("✔ Saving only $(nsaved) history points instead of $(sweeps).")
println("  First saved indices: ", save_idx[1:min(end, 20)])
println("  Last saved index: ", save_idx[end])

# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
weight_tag = weighted ? "weighted" : "unweighted"
grid_dir = joinpath(
    RESULTS_DIR,
    "StandardTabu_Gset$(gset)_tenures_$(weight_tag)_ntrials$(ntrials)_sweeps$(sweeps)"
)
mkpath(grid_dir)

master_summary = Vector{Dict{String,Any}}()

println("\n▶ Starting tenure sweep...")
println("   tenures = $(tenures)")
println("   ntrials = $(ntrials)")
println("   sweeps  = $(sweeps)")
println()

# ---------------------------------------------------------
# Tenure loop
# ---------------------------------------------------------
for tenure in tenures
    println("============================================================")
    println("Running combination: tenure = $(tenure)")
    println("============================================================")

    combo_dir = joinpath(grid_dir, "tenure$(tenure)")
    mkpath(combo_dir)

    checkpoint_path = joinpath(
        combo_dir,
        "standard_tabu_checkpoint_G$(gset)_tenure$(tenure)_sweeps$(sweeps)_ntrials$(ntrials).json"
    )

    final_save_path = joinpath(
        combo_dir,
        "standard_tabu_results_G$(gset)_tenure$(tenure)_sweeps$(sweeps)_ntrials$(ntrials).json"
    )

    # -----------------------------------------------------
    # Running accumulators: subsampled histories only
    # -----------------------------------------------------
    sum_best_cut_history   = zeros(Float64, nsaved)
    sumsq_best_cut_history = zeros(Float64, nsaved)

    sum_ratio_history      = zeros(Float64, nsaved)
    sumsq_ratio_history    = zeros(Float64, nsaved)

    sum_error_history      = zeros(Float64, nsaved)
    sumsq_error_history    = zeros(Float64, nsaved)

    sum_sweep_time_history   = zeros(Float64, nsaved)
    sumsq_sweep_time_history = zeros(Float64, nsaved)

    # -----------------------------------------------------
    # Running accumulators: final scalars
    # -----------------------------------------------------
    sum_final_best_cut   = 0.0
    sumsq_final_best_cut = 0.0

    sum_final_ratio      = 0.0
    sumsq_final_ratio    = 0.0

    sum_final_error      = 0.0
    sumsq_final_error    = 0.0

    sum_sweeps_done      = 0.0
    sumsq_sweeps_done    = 0.0

    sum_avg_time_per_sweep_sec      = 0.0
    sumsq_avg_time_per_sweep_sec    = 0.0

    sum_median_time_per_sweep_sec   = 0.0
    sumsq_median_time_per_sweep_sec = 0.0

    sum_total_runtime_sec      = 0.0
    sumsq_total_runtime_sec    = 0.0

    completed_trials = 0
    trial_seeds = Int[]

    # -----------------------------------------------------
    # Trial loop
    # -----------------------------------------------------
    for trial in 1:ntrials
        seed = base_seed + (trial - 1)

        println("------------------------------------------------------------")
        println("Trial $(trial) / $(ntrials), seed = $(seed)")
        println("tenure = $(tenure)")
        println("------------------------------------------------------------")

        best_cut, best_spins, best_hist, sweeps_done,
        sweep_times_sec, avg_time_per_sweep_sec,
        median_time_per_sweep_sec, total_runtime_sec = tabu_maxcut_fullscan(
            wg;
            sweeps=sweeps,
            tenure=tenure,
            seed=seed,
            verbose=verbose,
        )

        if length(best_hist) != sweeps
            error("Expected length(best_hist) = $(sweeps), got $(length(best_hist))")
        end
        if length(sweep_times_sec) != sweeps
            error("Expected length(sweep_times_sec) = $(sweeps), got $(length(sweep_times_sec))")
        end

        best_hist_saved   = Float64.(best_hist[save_idx])
        ratio_hist_saved  = best_hist_saved ./ optimal_cut
        error_hist_saved  = 1.0 .- ratio_hist_saved
        sweep_time_saved  = Float64.(sweep_times_sec[save_idx])

        final_ratio = float(best_cut) / optimal_cut
        final_error = 1.0 - final_ratio

        # -------------------------------------------------
        # Update histories
        # -------------------------------------------------
        sum_best_cut_history   .+= best_hist_saved
        sumsq_best_cut_history .+= best_hist_saved .^ 2

        sum_ratio_history      .+= ratio_hist_saved
        sumsq_ratio_history    .+= ratio_hist_saved .^ 2

        sum_error_history      .+= error_hist_saved
        sumsq_error_history    .+= error_hist_saved .^ 2

        sum_sweep_time_history   .+= sweep_time_saved
        sumsq_sweep_time_history .+= sweep_time_saved .^ 2

        # -------------------------------------------------
        # Update final scalars
        # -------------------------------------------------
        sum_final_best_cut   += float(best_cut)
        sumsq_final_best_cut += float(best_cut)^2

        sum_final_ratio      += final_ratio
        sumsq_final_ratio    += final_ratio^2

        sum_final_error      += final_error
        sumsq_final_error    += final_error^2

        sum_sweeps_done      += float(sweeps_done)
        sumsq_sweeps_done    += float(sweeps_done)^2

        sum_avg_time_per_sweep_sec   += avg_time_per_sweep_sec
        sumsq_avg_time_per_sweep_sec += avg_time_per_sweep_sec^2

        sum_median_time_per_sweep_sec   += median_time_per_sweep_sec
        sumsq_median_time_per_sweep_sec += median_time_per_sweep_sec^2

        sum_total_runtime_sec   += total_runtime_sec
        sumsq_total_runtime_sec += total_runtime_sec^2

        completed_trials += 1
        push!(trial_seeds, seed)

        # -------------------------------------------------
        # Running means: histories
        # -------------------------------------------------
        avg_best_cut_history       = sum_best_cut_history ./ completed_trials
        avg_ratio_history          = sum_ratio_history    ./ completed_trials
        avg_error_history          = sum_error_history    ./ completed_trials
        avg_sweep_time_history_sec = sum_sweep_time_history ./ completed_trials

        # -------------------------------------------------
        # Running std/sem: histories
        # -------------------------------------------------
        std_best_cut_history       = safe_sample_std(sum_best_cut_history, sumsq_best_cut_history, completed_trials)
        std_ratio_history          = safe_sample_std(sum_ratio_history,    sumsq_ratio_history,    completed_trials)
        std_error_history          = safe_sample_std(sum_error_history,    sumsq_error_history,    completed_trials)
        std_sweep_time_history_sec = safe_sample_std(sum_sweep_time_history, sumsq_sweep_time_history, completed_trials)

        if completed_trials > 1
            sem_best_cut_history       = std_best_cut_history ./ sqrt(completed_trials)
            sem_ratio_history          = std_ratio_history    ./ sqrt(completed_trials)
            sem_error_history          = std_error_history    ./ sqrt(completed_trials)
            sem_sweep_time_history_sec = std_sweep_time_history_sec ./ sqrt(completed_trials)
        else
            sem_best_cut_history       = fill(NaN, nsaved)
            sem_ratio_history          = fill(NaN, nsaved)
            sem_error_history          = fill(NaN, nsaved)
            sem_sweep_time_history_sec = fill(NaN, nsaved)
        end

        # -------------------------------------------------
        # Running means: final scalars
        # -------------------------------------------------
        avg_final_best_cut           = sum_final_best_cut / completed_trials
        avg_final_ratio              = sum_final_ratio    / completed_trials
        avg_final_error              = sum_final_error    / completed_trials
        avg_sweeps_done              = sum_sweeps_done    / completed_trials
        avg_avg_time_per_sweep_sec   = sum_avg_time_per_sweep_sec / completed_trials
        avg_median_time_per_sweep_sec = sum_median_time_per_sweep_sec / completed_trials
        avg_total_runtime_sec        = sum_total_runtime_sec / completed_trials

        # -------------------------------------------------
        # Running std/sem: final scalars
        # -------------------------------------------------
        std_final_best_cut = safe_sample_std(sum_final_best_cut, sumsq_final_best_cut, completed_trials)
        std_final_ratio    = safe_sample_std(sum_final_ratio,    sumsq_final_ratio,    completed_trials)
        std_final_error    = safe_sample_std(sum_final_error,    sumsq_final_error,    completed_trials)
        std_sweeps_done    = safe_sample_std(sum_sweeps_done,    sumsq_sweeps_done,    completed_trials)

        std_avg_time_per_sweep_sec = safe_sample_std(
            sum_avg_time_per_sweep_sec,
            sumsq_avg_time_per_sweep_sec,
            completed_trials
        )
        std_median_time_per_sweep_sec = safe_sample_std(
            sum_median_time_per_sweep_sec,
            sumsq_median_time_per_sweep_sec,
            completed_trials
        )
        std_total_runtime_sec = safe_sample_std(
            sum_total_runtime_sec,
            sumsq_total_runtime_sec,
            completed_trials
        )

        if completed_trials > 1
            sem_final_best_cut = std_final_best_cut / sqrt(completed_trials)
            sem_final_ratio    = std_final_ratio    / sqrt(completed_trials)
            sem_final_error    = std_final_error    / sqrt(completed_trials)
            sem_sweeps_done    = std_sweeps_done    / sqrt(completed_trials)

            sem_avg_time_per_sweep_sec = std_avg_time_per_sweep_sec / sqrt(completed_trials)
            sem_median_time_per_sweep_sec = std_median_time_per_sweep_sec / sqrt(completed_trials)
            sem_total_runtime_sec      = std_total_runtime_sec / sqrt(completed_trials)
        else
            sem_final_best_cut = NaN
            sem_final_ratio    = NaN
            sem_final_error    = NaN
            sem_sweeps_done    = NaN

            sem_avg_time_per_sweep_sec = NaN
            sem_median_time_per_sweep_sec = NaN
            sem_total_runtime_sec      = NaN
        end

        println("  ✓ Trial best cut             = $best_cut")
        println("  ✓ Trial final ratio          = $(round(final_ratio, digits=6))")
        println("  ✓ Trial final error          = $(round(final_error, digits=6))")
        println("  ✓ Trial avg time / sweep     = $(round(avg_time_per_sweep_sec, digits=6)) s")
        println("  ✓ Trial median time / sweep  = $(round(median_time_per_sweep_sec, digits=6)) s")
        println("  ✓ Trial total runtime        = $(round(total_runtime_sec, digits=6)) s")
        println("  ✓ Running avg final ratio    = $(round(avg_final_ratio, digits=6))")
        println("  ✓ Running avg final error    = $(round(avg_final_error, digits=6))")
        println("  ✓ Running avg time / sweep   = $(round(avg_avg_time_per_sweep_sec, digits=6)) s")
        println("  ✓ Running avg median / sweep = $(round(avg_median_time_per_sweep_sec, digits=6)) s")
        println("  ✓ Running avg total runtime  = $(round(avg_total_runtime_sec, digits=6)) s")

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------
        if save_checkpoint_every_trial
            checkpoint_data = Dict(
                "solver" => "StandardTabuFullScan",
                "mode" => "running_average_checkpoint",

                "gset" => gset,
                "graphfile" => graphfile,
                "weighted_flag" => weighted,
                "N" => N,

                "tenure" => tenure,

                "ntrials_requested" => ntrials,
                "ntrials_completed" => completed_trials,
                "trial_seeds_completed" => trial_seeds,
                "base_seed" => base_seed,

                "sweeps_requested" => sweeps,
                "optimal_cut" => optimal_cut,

                "sweep_convention" => "1 sweep = 1 move after evaluating all sites",
                "history_sampling" => "saved only at logarithmically spaced sweeps",
                "saved_sweep_indices" => save_idx,

                "avg_best_cut_history" => avg_best_cut_history,
                "std_best_cut_history" => std_best_cut_history,
                "sem_best_cut_history" => sem_best_cut_history,

                "avg_ratio_history" => avg_ratio_history,
                "std_ratio_history" => std_ratio_history,
                "sem_ratio_history" => sem_ratio_history,

                "avg_error_history" => avg_error_history,
                "std_error_history" => std_error_history,
                "sem_error_history" => sem_error_history,

                "avg_sweep_time_history_sec" => avg_sweep_time_history_sec,
                "std_sweep_time_history_sec" => std_sweep_time_history_sec,
                "sem_sweep_time_history_sec" => sem_sweep_time_history_sec,

                "avg_final_best_cut" => avg_final_best_cut,
                "std_final_best_cut" => std_final_best_cut,
                "sem_final_best_cut" => sem_final_best_cut,

                "avg_final_ratio" => avg_final_ratio,
                "std_final_ratio" => std_final_ratio,
                "sem_final_ratio" => sem_final_ratio,

                "avg_final_error" => avg_final_error,
                "std_final_error" => std_final_error,
                "sem_final_error" => sem_final_error,

                "avg_sweeps_done" => avg_sweeps_done,
                "std_sweeps_done" => std_sweeps_done,
                "sem_sweeps_done" => sem_sweeps_done,

                "avg_avg_time_per_sweep_sec" => avg_avg_time_per_sweep_sec,
                "std_avg_time_per_sweep_sec" => std_avg_time_per_sweep_sec,
                "sem_avg_time_per_sweep_sec" => sem_avg_time_per_sweep_sec,

                "avg_median_time_per_sweep_sec" => avg_median_time_per_sweep_sec,
                "std_median_time_per_sweep_sec" => std_median_time_per_sweep_sec,
                "sem_median_time_per_sweep_sec" => sem_median_time_per_sweep_sec,

                "avg_total_runtime_sec" => avg_total_runtime_sec,
                "std_total_runtime_sec" => std_total_runtime_sec,
                "sem_total_runtime_sec" => sem_total_runtime_sec,
            )

            atomic_json_write(checkpoint_path, checkpoint_data)
            println("  ✓ Checkpoint saved to: $checkpoint_path")
        end
    end

    # -----------------------------------------------------
    # Final stats for this tenure
    # -----------------------------------------------------
    avg_best_cut_history       = sum_best_cut_history ./ completed_trials
    avg_ratio_history          = sum_ratio_history    ./ completed_trials
    avg_error_history          = sum_error_history    ./ completed_trials
    avg_sweep_time_history_sec = sum_sweep_time_history ./ completed_trials

    std_best_cut_history       = safe_sample_std(sum_best_cut_history, sumsq_best_cut_history, completed_trials)
    std_ratio_history          = safe_sample_std(sum_ratio_history,    sumsq_ratio_history,    completed_trials)
    std_error_history          = safe_sample_std(sum_error_history,    sumsq_error_history,    completed_trials)
    std_sweep_time_history_sec = safe_sample_std(sum_sweep_time_history, sumsq_sweep_time_history, completed_trials)

    if completed_trials > 1
        sem_best_cut_history       = std_best_cut_history ./ sqrt(completed_trials)
        sem_ratio_history          = std_ratio_history    ./ sqrt(completed_trials)
        sem_error_history          = std_error_history    ./ sqrt(completed_trials)
        sem_sweep_time_history_sec = std_sweep_time_history_sec ./ sqrt(completed_trials)
    else
        sem_best_cut_history       = fill(NaN, nsaved)
        sem_ratio_history          = fill(NaN, nsaved)
        sem_error_history          = fill(NaN, nsaved)
        sem_sweep_time_history_sec = fill(NaN, nsaved)
    end

    avg_final_best_cut            = sum_final_best_cut / completed_trials
    avg_final_ratio               = sum_final_ratio    / completed_trials
    avg_final_error               = sum_final_error    / completed_trials
    avg_sweeps_done               = sum_sweeps_done    / completed_trials
    avg_avg_time_per_sweep_sec    = sum_avg_time_per_sweep_sec / completed_trials
    avg_median_time_per_sweep_sec = sum_median_time_per_sweep_sec / completed_trials
    avg_total_runtime_sec         = sum_total_runtime_sec / completed_trials

    std_final_best_cut = safe_sample_std(sum_final_best_cut, sumsq_final_best_cut, completed_trials)
    std_final_ratio    = safe_sample_std(sum_final_ratio,    sumsq_final_ratio,    completed_trials)
    std_final_error    = safe_sample_std(sum_final_error,    sumsq_final_error,    completed_trials)
    std_sweeps_done    = safe_sample_std(sum_sweeps_done,    sumsq_sweeps_done,    completed_trials)

    std_avg_time_per_sweep_sec = safe_sample_std(
        sum_avg_time_per_sweep_sec,
        sumsq_avg_time_per_sweep_sec,
        completed_trials
    )
    std_median_time_per_sweep_sec = safe_sample_std(
        sum_median_time_per_sweep_sec,
        sumsq_median_time_per_sweep_sec,
        completed_trials
    )
    std_total_runtime_sec = safe_sample_std(
        sum_total_runtime_sec,
        sumsq_total_runtime_sec,
        completed_trials
    )

    if completed_trials > 1
        sem_final_best_cut = std_final_best_cut / sqrt(completed_trials)
        sem_final_ratio    = std_final_ratio    / sqrt(completed_trials)
        sem_final_error    = std_final_error    / sqrt(completed_trials)
        sem_sweeps_done    = std_sweeps_done    / sqrt(completed_trials)

        sem_avg_time_per_sweep_sec = std_avg_time_per_sweep_sec / sqrt(completed_trials)
        sem_median_time_per_sweep_sec = std_median_time_per_sweep_sec / sqrt(completed_trials)
        sem_total_runtime_sec      = std_total_runtime_sec / sqrt(completed_trials)
    else
        sem_final_best_cut = NaN
        sem_final_ratio    = NaN
        sem_final_error    = NaN
        sem_sweeps_done    = NaN

        sem_avg_time_per_sweep_sec = NaN
        sem_median_time_per_sweep_sec = NaN
        sem_total_runtime_sec      = NaN
    end

    combo_data = Dict(
        "solver" => "StandardTabuFullScan",
        "mode" => "final_average",

        "gset" => gset,
        "graphfile" => graphfile,
        "weighted_flag" => weighted,
        "N" => N,

        "tenure" => tenure,

        "ntrials_requested" => ntrials,
        "ntrials_completed" => completed_trials,
        "trial_seeds_completed" => trial_seeds,
        "base_seed" => base_seed,

        "sweeps_requested" => sweeps,
        "optimal_cut" => optimal_cut,

        "sweep_convention" => "1 sweep = 1 move after evaluating all sites",
        "history_sampling" => "saved only at logarithmically spaced sweeps",
        "saved_sweep_indices" => save_idx,

        "avg_best_cut_history" => avg_best_cut_history,
        "std_best_cut_history" => std_best_cut_history,
        "sem_best_cut_history" => sem_best_cut_history,

        "avg_ratio_history" => avg_ratio_history,
        "std_ratio_history" => std_ratio_history,
        "sem_ratio_history" => sem_ratio_history,

        "avg_error_history" => avg_error_history,
        "std_error_history" => std_error_history,
        "sem_error_history" => sem_error_history,

        "avg_sweep_time_history_sec" => avg_sweep_time_history_sec,
        "std_sweep_time_history_sec" => std_sweep_time_history_sec,
        "sem_sweep_time_history_sec" => sem_sweep_time_history_sec,

        "avg_final_best_cut" => avg_final_best_cut,
        "std_final_best_cut" => std_final_best_cut,
        "sem_final_best_cut" => sem_final_best_cut,

        "avg_final_ratio" => avg_final_ratio,
        "std_final_ratio" => std_final_ratio,
        "sem_final_ratio" => sem_final_ratio,

        "avg_final_error" => avg_final_error,
        "std_final_error" => std_final_error,
        "sem_final_error" => sem_final_error,

        "avg_sweeps_done" => avg_sweeps_done,
        "std_sweeps_done" => std_sweeps_done,
        "sem_sweeps_done" => sem_sweeps_done,

        "avg_avg_time_per_sweep_sec" => avg_avg_time_per_sweep_sec,
        "std_avg_time_per_sweep_sec" => std_avg_time_per_sweep_sec,
        "sem_avg_time_per_sweep_sec" => sem_avg_time_per_sweep_sec,

        "avg_median_time_per_sweep_sec" => avg_median_time_per_sweep_sec,
        "std_median_time_per_sweep_sec" => std_median_time_per_sweep_sec,
        "sem_median_time_per_sweep_sec" => sem_median_time_per_sweep_sec,

        "avg_total_runtime_sec" => avg_total_runtime_sec,
        "std_total_runtime_sec" => std_total_runtime_sec,
        "sem_total_runtime_sec" => sem_total_runtime_sec,
    )

    atomic_json_write(final_save_path, combo_data)

    println()
    println("✔ Finished combination: tenure = $(tenure)")
    println("  Avg final ratio           = $(round(avg_final_ratio, digits=6))")
    println("  Avg final error           = $(round(avg_final_error, digits=6))")
    println("  Avg time per sweep        = $(round(avg_avg_time_per_sweep_sec, digits=6)) s")
    println("  Avg median time / sweep   = $(round(avg_median_time_per_sweep_sec, digits=6)) s")
    println("  Avg total runtime         = $(round(avg_total_runtime_sec, digits=6)) s")
    println("  Saved to                  = $final_save_path")
    println()

    push!(master_summary, Dict(
        "tenure" => tenure,
        "ntrials_completed" => completed_trials,
        "avg_final_ratio" => avg_final_ratio,
        "std_final_ratio" => std_final_ratio,
        "sem_final_ratio" => sem_final_ratio,
        "avg_final_error" => avg_final_error,
        "std_final_error" => std_final_error,
        "sem_final_error" => sem_final_error,
        "avg_avg_time_per_sweep_sec" => avg_avg_time_per_sweep_sec,
        "std_avg_time_per_sweep_sec" => std_avg_time_per_sweep_sec,
        "sem_avg_time_per_sweep_sec" => sem_avg_time_per_sweep_sec,
        "avg_median_time_per_sweep_sec" => avg_median_time_per_sweep_sec,
        "std_median_time_per_sweep_sec" => std_median_time_per_sweep_sec,
        "sem_median_time_per_sweep_sec" => sem_median_time_per_sweep_sec,
        "avg_total_runtime_sec" => avg_total_runtime_sec,
        "std_total_runtime_sec" => std_total_runtime_sec,
        "sem_total_runtime_sec" => sem_total_runtime_sec,
    ))
end

# ---------------------------------------------------------
# Save master summary
# ---------------------------------------------------------
master_summary_path = joinpath(
    grid_dir,
    "standard_tabu_master_summary_G$(gset)_sweeps$(sweeps)_ntrials$(ntrials).json"
)

atomic_json_write(master_summary_path, master_summary)

println("============================================================")
println("                  Tenure Sweep Complete                     ")
println("============================================================")
println("✔ Master summary saved to: $master_summary_path")
println("Done.")

