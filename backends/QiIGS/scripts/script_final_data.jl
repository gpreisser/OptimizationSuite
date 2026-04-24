# scripts/qiigs_lambda_sweep_unique_minima.jl
#
# Adds:
#   - unique_angle_count_raw:         distinct optimized angle minima BEFORE rounding
#                                     with no global-flip reduction
#   - unique_angle_count_reduced:     distinct optimized angle minima BEFORE rounding
#                                     reduced by θ ~ θ + π/2
#   - unique_angle_opt_count_raw:     distinct optimized angle minima that ROUND to the
#                                     optimal solution, no global-flip reduction
#   - unique_angle_opt_count_reduced: distinct optimized angle minima that ROUND to the
#                                     optimal solution, reduced by θ ~ θ + π/2
#   - opt_min_hess_cond_mean:         pooled mean Hessian condition number of minima whose
#                                     rounded spin configuration attains the exact optimal cut value
#   - opt_min_hess_cond_median:       pooled median Hessian condition number of minima whose
#                                     rounded spin configuration attains the exact optimal cut value
#   - n_opt_min_hess_conds:           number of pooled condition-number samples used above
#   - opt_min_probs_raw:              ALL empirical probabilities p_α of distinct optimal minima,
#                                     pooled across graphs; each p_α = hits / n_inits for one graph
#   - opt_min_probs_mean_contrib:     same probabilities divided by n_graphs, so the sum over all
#                                     points at fixed λ equals success_rate_mean in the final JSON
#   - opt_min_probs_by_graph:         per-graph lists of the optimal-minimum probabilities
#   - devZ_abs:                       mean_i |cos(2θ_i)| from accepted θ_best
#   - devTheta_abs:                   mean_i |0.5*acos(|cos(2θ_i)|) - π/4| from accepted θ_best
#   - init_hess_mineig:               Hessian minimum eigenvalue at the INITIAL random state
#   - init_hess_cond:                 Hessian condition number at the INITIAL random state
#
# NOTE:
#   - Requires save_params=true so solver stores theta_converged / theta_best in metadata.
#   - Angles are NOT saved to JSON; only counts/diagnostics are recorded.
#
# HESSIAN MODE:
#   - If compute_hessian=false:
#       counts all converged angle configurations (old behavior)
#   - If compute_hessian=true:
#       counts only those converged angle configurations classified as minima
#
# IMPORTANT:
#   - angle_conv is included in the FINAL output filename
#   - Hessian mode/tolerance are also included in the filename
#   - angle_conv is stored at top level in save_data
#
# NEW FOR N*:
#   - N* is taken to be the number of DISTINCT ANGLE MINIMA whose rounded spin
#     configuration attains the exact optimal cut value.
#   - We report both raw and reduced angle-space counts; the reduced version is the
#     natural plotted choice for N*.
#
# NEW FOR PANEL (b) INSET:
#   - Saves summary statistics of condition numbers of optimal minima:
#       results_per_lambda[i]["opt_min_hess_cond_mean"]
#       results_per_lambda[i]["opt_min_hess_cond_median"]
#       results_per_lambda[i]["n_opt_min_hess_conds"]
#
# NEW FOR PANEL (c)-type basin-probability plot:
#   - For each λ and graph, count how many random initializations land in each DISTINCT
#     optimal minimum (in reduced angle-key space).
#   - Convert counts to probabilities p_α = hits / n_inits.
#   - Save:
#       * opt_min_probs_raw         : pooled raw p_α values across graphs
#       * opt_min_probs_mean_contrib: pooled p_α / n_graphs values
#       * opt_min_probs_by_graph    : per-graph probability lists
#
# Then:
#   - For a fixed graph:   sum_α p_α(λ) = success rate for that graph at λ
#   - In the final JSON:   sum over opt_min_probs_mean_contrib at fixed λ equals success_rate_mean

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using Dates
using Statistics
using SparseArrays

# ---------------------------
# Helpers
# ---------------------------

@inline function mean_and_stderr(sumx, sumx2, n::Int)
    μ = sumx / n
    if n ≤ 1
        return μ, 0.0
    end
    var = (sumx2 - n * μ^2) / (n - 1)
    var = max(var, 0.0)
    σ = sqrt(var)
    return μ, σ / sqrt(n)
end

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

@inline function is_exact_optimum(cut_hat, opt)
    if opt === nothing
        return false
    end
    if cut_hat isa Integer && opt isa Integer
        return cut_hat == opt
    end
    return isapprox(Float64(cut_hat), Float64(opt); atol=1e-12, rtol=0.0)
end

# ---------------------------
# Angle-key hashing (pre-rounding uniqueness)
# ---------------------------

function angle_key(θ::AbstractVector{<:AbstractFloat}; δ::Float64=1e-3, flip_equiv::Bool=true)
    function qvec_for(shift::Float64)
        q = Vector{Int32}(undef, length(θ))
        @inbounds for i in eachindex(θ)
            x = mod(Float64(θ[i]) + shift, pi)
            q[i] = Int32(floor(x / δ + 0.5))
        end
        return Tuple(q)
    end

    q0 = qvec_for(0.0)
    if !flip_equiv
        return q0
    end

    q1 = qvec_for(pi / 2)
    return min(q0, q1)
end

# ---------------------------
# Main
# ---------------------------

function main()
    println("============================================================================")
    println("  QiIGS λ-sweep: uniques + N* + optimal-minimum κ summaries + probabilities ")
    println("  Inputs: graphs/solutions from QiILS_ITensor                                 ")
    println("  Outputs: QiIGS/results/... (with rolling checkpoint)                        ")
    println("============================================================================")

    # -----------------------
    # Experiment parameters
    # -----------------------
    N = 50
    k = 3
    weighted = false

    graph_seeds = collect(1:100)
    λs = collect(0.0:0.025:1.0)
    nλ = length(λs)

    n_inits = 10000

    iterations = 1
    inner_iterations = 5000
    tao = 0.025
    angle_conv = 1e-8
    init_mode = :uniform

    save_params = true

    success_thr = 0.999

    angle_bin = 1e-2
    angle_flip_equiv = true

    # Hessian options
    compute_hessian = true
    hessian_tol = 1e-8

    ckpt_namespace = "v13"

    hmode_tag = compute_hessian ? "hess_on" : "hess_off"
    htol_tag = replace(@sprintf("%.3g", hessian_tol), "." => "p")

    init_sig = "ns=$(ckpt_namespace)_init=$(String(init_mode))_tao=$(tao)_conv=$(angle_conv)_save=$(save_params)_abin=$(angle_bin)_aflip=$(angle_flip_equiv)_$(hmode_tag)_htol$(htol_tag)"
    seed_salt = 0

    keyT = typeof(angle_key(zeros(Float64, N); δ=angle_bin, flip_equiv=false))

    # -----------------------
    # Paths
    # -----------------------
    ROOT = normpath(joinpath(@__DIR__, ".."))
    RESULTS_DIR = joinpath(ROOT, "results")
    mkpath(RESULTS_DIR)

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    wtag = weighted ? "weighted" : "unweighted"

    save_dir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(wtag)"
    )
    mkpath(save_dir)

    λmin = first(λs)
    λmax = last(λs)
    dλ = (length(λs) > 1 ? (λs[2] - λs[1]) : 0.0)
    tag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"

    ckpt_path = joinpath(
        save_dir,
        "checkpoint_unique_ratio_meanbest_succ_grad_$(tag)_$(init_sig)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    @info "QiIGS ROOT" ROOT
    @info "Saving under" save_dir
    @info "Reusing graphs from" GRAPHS_ROOT
    @info "Reusing solutions from" SOLUTIONS_ROOT
    @info "Checkpoint" ckpt_path
    @info "compute_hessian" compute_hessian
    @info "hessian_tol" hessian_tol

    # -----------------------
    # Accumulators
    # -----------------------
    sum_unique  = zeros(Float64, nλ)
    sum_unique2 = zeros(Float64, nλ)

    sum_unique_angle_raw  = zeros(Float64, nλ)
    sum_unique_angle_raw2 = zeros(Float64, nλ)

    sum_unique_angle_reduced  = zeros(Float64, nλ)
    sum_unique_angle_reduced2 = zeros(Float64, nλ)

    # N*
    sum_unique_angle_opt_raw  = zeros(Float64, nλ)
    sum_unique_angle_opt_raw2 = zeros(Float64, nλ)

    sum_unique_angle_opt_reduced  = zeros(Float64, nλ)
    sum_unique_angle_opt_reduced2 = zeros(Float64, nλ)

    sum_best_ratio  = zeros(Float64, nλ)
    sum_best_ratio2 = zeros(Float64, nλ)

    sum_mean_ratio  = zeros(Float64, nλ)
    sum_mean_ratio2 = zeros(Float64, nλ)

    n_ratio = zeros(Int, nλ)

    sum_succ  = zeros(Float64, nλ)
    sum_succ2 = zeros(Float64, nλ)
    n_succ    = zeros(Int, nλ)

    sum_gn_final_mean  = zeros(Float64, nλ)
    sum_gn_final_mean2 = zeros(Float64, nλ)
    sum_gn_final_min   = zeros(Float64, nλ)
    sum_gn_final_min2  = zeros(Float64, nλ)

    sum_gn_init_mean  = zeros(Float64, nλ)
    sum_gn_init_mean2 = zeros(Float64, nλ)

    sum_gn_maxinner_mean  = zeros(Float64, nλ)
    sum_gn_maxinner_mean2 = zeros(Float64, nλ)

    sum_gn_meaninner_mean  = zeros(Float64, nλ)
    sum_gn_meaninner_mean2 = zeros(Float64, nλ)

    sum_devZ_abs_mean  = zeros(Float64, nλ)
    sum_devZ_abs_mean2 = zeros(Float64, nλ)

    sum_devTheta_abs_mean  = zeros(Float64, nλ)
    sum_devTheta_abs_mean2 = zeros(Float64, nλ)

    sum_inner_sweeps_mean  = zeros(Float64, nλ)
    sum_inner_sweeps_mean2 = zeros(Float64, nλ)

    sum_rt_mean  = zeros(Float64, nλ)
    sum_rt_mean2 = zeros(Float64, nλ)

    # Final-state Hessian-derived accumulators
    sum_hess_frac_min  = zeros(Float64, nλ)
    sum_hess_frac_min2 = zeros(Float64, nλ)

    sum_hess_frac_saddle  = zeros(Float64, nλ)
    sum_hess_frac_saddle2 = zeros(Float64, nλ)

    sum_hess_frac_max  = zeros(Float64, nλ)
    sum_hess_frac_max2 = zeros(Float64, nλ)

    sum_hess_frac_degenerate  = zeros(Float64, nλ)
    sum_hess_frac_degenerate2 = zeros(Float64, nλ)

    sum_hess_mineig_mean  = zeros(Float64, nλ)
    sum_hess_mineig_mean2 = zeros(Float64, nλ)

    sum_hess_cond_mean  = zeros(Float64, nλ)
    sum_hess_cond_mean2 = zeros(Float64, nλ)

    # Initial-state Hessian-derived accumulators
    sum_init_hess_mineig_mean  = zeros(Float64, nλ)
    sum_init_hess_mineig_mean2 = zeros(Float64, nλ)
    n_init_hess_mineig_graphs  = zeros(Int, nλ)

    sum_init_hess_cond_mean  = zeros(Float64, nλ)
    sum_init_hess_cond_mean2 = zeros(Float64, nλ)
    n_init_hess_cond_graphs  = zeros(Int, nλ)

    n_graphs = 0
    start_index = 1

    # -----------------------
    # Resume
    # -----------------------
    if isfile(ckpt_path)
        ck = JSON.parsefile(ckpt_path)

        if ck["N"] == N && ck["k"] == k && ck["weighted_flag"] == weighted &&
           length(ck["λs"]) == length(λs) &&
           ck["n_inits"] == n_inits &&
           ck["iterations"] == iterations &&
           ck["inner_iterations"] == inner_iterations &&
           ck["success_thr"] == success_thr &&
           get(ck, "init_sig", "") == init_sig &&
           get(ck, "compute_hessian", false) == compute_hessian &&
           isapprox(Float64(get(ck, "hessian_tol", hessian_tol)), hessian_tol; atol=0.0, rtol=0.0)

            sum_unique  .= Float64.(ck["sum_unique"])
            sum_unique2 .= Float64.(ck["sum_unique2"])

            if haskey(ck, "sum_unique_angle_raw")
                sum_unique_angle_raw  .= Float64.(ck["sum_unique_angle_raw"])
                sum_unique_angle_raw2 .= Float64.(ck["sum_unique_angle_raw2"])
            end

            if haskey(ck, "sum_unique_angle_reduced")
                sum_unique_angle_reduced  .= Float64.(ck["sum_unique_angle_reduced"])
                sum_unique_angle_reduced2 .= Float64.(ck["sum_unique_angle_reduced2"])
            end

            if haskey(ck, "sum_unique_angle_opt_raw")
                sum_unique_angle_opt_raw  .= Float64.(ck["sum_unique_angle_opt_raw"])
                sum_unique_angle_opt_raw2 .= Float64.(ck["sum_unique_angle_opt_raw2"])
            end

            if haskey(ck, "sum_unique_angle_opt_reduced")
                sum_unique_angle_opt_reduced  .= Float64.(ck["sum_unique_angle_opt_reduced"])
                sum_unique_angle_opt_reduced2 .= Float64.(ck["sum_unique_angle_opt_reduced2"])
            end

            sum_best_ratio  .= Float64.(ck["sum_best_ratio"])
            sum_best_ratio2 .= Float64.(ck["sum_best_ratio2"])
            sum_mean_ratio  .= Float64.(ck["sum_mean_ratio"])
            sum_mean_ratio2 .= Float64.(ck["sum_mean_ratio2"])
            n_ratio .= Int.(ck["n_ratio"])

            sum_succ  .= Float64.(ck["sum_succ"])
            sum_succ2 .= Float64.(ck["sum_succ2"])
            n_succ    .= Int.(ck["n_succ"])

            sum_gn_final_mean  .= Float64.(ck["sum_gn_final_mean"])
            sum_gn_final_mean2 .= Float64.(ck["sum_gn_final_mean2"])
            sum_gn_final_min   .= Float64.(ck["sum_gn_final_min"])
            sum_gn_final_min2  .= Float64.(ck["sum_gn_final_min2"])

            sum_gn_init_mean  .= Float64.(ck["sum_gn_init_mean"])
            sum_gn_init_mean2 .= Float64.(ck["sum_gn_init_mean2"])
            sum_gn_maxinner_mean  .= Float64.(ck["sum_gn_maxinner_mean"])
            sum_gn_maxinner_mean2 .= Float64.(ck["sum_gn_maxinner_mean2"])
            sum_gn_meaninner_mean  .= Float64.(ck["sum_gn_meaninner_mean"])
            sum_gn_meaninner_mean2 .= Float64.(ck["sum_gn_meaninner_mean2"])

            if haskey(ck, "sum_devZ_abs_mean")
                sum_devZ_abs_mean  .= Float64.(ck["sum_devZ_abs_mean"])
                sum_devZ_abs_mean2 .= Float64.(ck["sum_devZ_abs_mean2"])
            end

            if haskey(ck, "sum_devTheta_abs_mean")
                sum_devTheta_abs_mean  .= Float64.(ck["sum_devTheta_abs_mean"])
                sum_devTheta_abs_mean2 .= Float64.(ck["sum_devTheta_abs_mean2"])
            end

            if haskey(ck, "sum_inner_sweeps_mean")
                sum_inner_sweeps_mean  .= Float64.(ck["sum_inner_sweeps_mean"])
                sum_inner_sweeps_mean2 .= Float64.(ck["sum_inner_sweeps_mean2"])
            end

            sum_rt_mean  .= Float64.(ck["sum_rt_mean"])
            sum_rt_mean2 .= Float64.(ck["sum_rt_mean2"])

            if haskey(ck, "sum_hess_frac_min")
                sum_hess_frac_min  .= Float64.(ck["sum_hess_frac_min"])
                sum_hess_frac_min2 .= Float64.(ck["sum_hess_frac_min2"])
                sum_hess_frac_saddle  .= Float64.(ck["sum_hess_frac_saddle"])
                sum_hess_frac_saddle2 .= Float64.(ck["sum_hess_frac_saddle2"])
                sum_hess_frac_max  .= Float64.(ck["sum_hess_frac_max"])
                sum_hess_frac_max2 .= Float64.(ck["sum_hess_frac_max2"])
                sum_hess_frac_degenerate  .= Float64.(ck["sum_hess_frac_degenerate"])
                sum_hess_frac_degenerate2 .= Float64.(ck["sum_hess_frac_degenerate2"])
                sum_hess_mineig_mean  .= Float64.(ck["sum_hess_mineig_mean"])
                sum_hess_mineig_mean2 .= Float64.(ck["sum_hess_mineig_mean2"])
                sum_hess_cond_mean  .= Float64.(ck["sum_hess_cond_mean"])
                sum_hess_cond_mean2 .= Float64.(ck["sum_hess_cond_mean2"])
            end

            if haskey(ck, "sum_init_hess_mineig_mean")
                sum_init_hess_mineig_mean  .= Float64.(ck["sum_init_hess_mineig_mean"])
                sum_init_hess_mineig_mean2 .= Float64.(ck["sum_init_hess_mineig_mean2"])
            end
            if haskey(ck, "n_init_hess_mineig_graphs")
                n_init_hess_mineig_graphs .= Int.(ck["n_init_hess_mineig_graphs"])
            end

            if haskey(ck, "sum_init_hess_cond_mean")
                sum_init_hess_cond_mean  .= Float64.(ck["sum_init_hess_cond_mean"])
                sum_init_hess_cond_mean2 .= Float64.(ck["sum_init_hess_cond_mean2"])
            end
            if haskey(ck, "n_init_hess_cond_graphs")
                n_init_hess_cond_graphs .= Int.(ck["n_init_hess_cond_graphs"])
            end

            n_graphs = Int(ck["n_graphs"])
            start_index = Int(ck["next_seed_index"])
            println("↻ Resuming: n_graphs=$n_graphs, next_seed_index=$start_index")
        else
            println("⚠ Found checkpoint but parameters differ; ignoring it.")
        end
    end

    # These store pooled condition-number samples across all graphs, grouped by λ.
    # They are only used in-memory to compute summary statistics in the final JSON.
    all_opt_min_hess_conds = [Float64[] for _ in 1:nλ]

    # These store per-graph probability lists, one entry per processed graph.
    # each entry is itself a vector of probabilities of distinct optimal minima for that graph.
    all_opt_min_probs_by_graph = [Vector{Vector{Float64}}() for _ in 1:nλ]

    # -----------------------
    # Outer loop
    # -----------------------
    for idx in start_index:length(graph_seeds)
        gs = graph_seeds[idx]
        println("\n► graph seed = $gs  (graph $idx / $(length(graph_seeds)))")

        gpath = QiIGS.graph_path(N, k, gs; weighted=weighted, graphs_root=GRAPHS_ROOT)
        @assert isfile(gpath) "Graph file not found: $gpath"
        W = QiIGS.load_weight_matrix(gpath)

        spath = QiIGS.akmax_solution_path(N, k, gs; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
        opt = QiIGS.load_optimal_cut(spath)
        if opt === nothing
            println("    ⚠ No optimal cut found: $spath")
        else
            println("    ✔ optimal_cut = $opt")
        end

        unique_count = zeros(Float64, nλ)
        unique_angle_count_raw = zeros(Float64, nλ)
        unique_angle_count_reduced = zeros(Float64, nλ)

        unique_angle_opt_count_raw = zeros(Float64, nλ)
        unique_angle_opt_count_reduced = zeros(Float64, nλ)

        best_ratio = Vector{Union{Nothing,Float64}}(undef, nλ)
        fill!(best_ratio, nothing)

        mean_ratio = Vector{Union{Nothing,Float64}}(undef, nλ)
        fill!(mean_ratio, nothing)

        success_rate = zeros(Float64, nλ)

        gn_final_mean = zeros(Float64, nλ)
        gn_final_min  = zeros(Float64, nλ)

        gn_init_mean      = zeros(Float64, nλ)
        gn_maxinner_mean  = zeros(Float64, nλ)
        gn_meaninner_mean = zeros(Float64, nλ)

        devZ_abs_mean = zeros(Float64, nλ)
        devTheta_abs_mean = zeros(Float64, nλ)

        inner_sweeps_used_mean = zeros(Float64, nλ)
        rt_mean = zeros(Float64, nλ)

        hess_frac_min = fill(NaN, nλ)
        hess_frac_saddle = fill(NaN, nλ)
        hess_frac_max = fill(NaN, nλ)
        hess_frac_degenerate = fill(NaN, nλ)
        hess_mineig_mean = fill(NaN, nλ)
        hess_cond_mean = fill(NaN, nλ)

        init_hess_mineig_mean = fill(NaN, nλ)
        init_hess_cond_mean = fill(NaN, nλ)

        # Per-graph probability lists
        graph_opt_min_probs = [Float64[] for _ in 1:nλ]

        for (iλ, λ) in enumerate(λs)
            seen_spins = Set{UInt64}()
            seen_angles_raw = Set{keyT}()
            seen_angles_reduced = Set{keyT}()

            seen_angles_opt_raw = Set{keyT}()
            seen_angles_opt_reduced = Set{keyT}()

            # count hits of each distinct optimal minimum (reduced key)
            counts_opt_min_reduced = Dict{keyT, Int}()

            sum_gn_final = 0.0
            min_gn_final = Inf

            sum_gn_init = 0.0
            sum_gn_maxinner = 0.0
            sum_gn_meaninner = 0.0

            sum_devZ_abs = 0.0
            sum_devTheta_abs = 0.0

            sum_inner_sweeps_used = 0.0
            sum_rt = 0.0

            sum_init_hess_mineig = 0.0
            sum_init_hess_cond = 0.0

            cnt_init_hess_mineig = 0
            cnt_init_hess_cond = 0

            best_r = -Inf
            sum_r = 0.0
            succ = 0

            n_min = 0
            n_saddle = 0
            n_max = 0
            n_deg = 0
            sum_mineig_minima = 0.0
            sum_cond_minima = 0.0

            nκ_opt = 0

            for r in 1:n_inits
                run_seed = gs * 1_000_000 + r * 10_000 + Int(round(λ * 1000)) + seed_salt * 1_000_000_000

                t0 = time()
                sol = QiIGS.solve!(
                    W, N;
                    solver = :grad,
                    seed = run_seed,
                    lambda = λ,
                    iterations = iterations,
                    inner_iterations = inner_iterations,
                    tao = tao,
                    angle_conv = angle_conv,
                    init_mode = init_mode,
                    save_params = save_params,
                    progressbar = false,
                    compute_hessian = compute_hessian,
                    hessian_tol = hessian_tol,
                )
                dt = time() - t0

                push!(seen_spins, QiIGS.spin_config_key(sol.configuration))

                θc = get(sol.metadata, :theta_converged, Float64[])
                isempty(θc) && error("Missing :theta_converged in sol.metadata")

                angle_ok = true
                hess_cond_this = NaN
                hess_mineig_this = NaN

                if compute_hessian
                    is_minimum = get(sol.metadata, :hess_is_minimum, false)
                    is_saddle = get(sol.metadata, :hess_is_saddle, false)
                    is_maximum = get(sol.metadata, :hess_is_maximum, false)
                    is_degenerate = get(sol.metadata, :hess_is_degenerate, false)

                    if is_minimum
                        n_min += 1
                        hess_mineig_this = Float64(get(sol.metadata, :hess_mineig, NaN))
                        hess_cond_this   = Float64(get(sol.metadata, :hess_cond, NaN))
                        sum_mineig_minima += hess_mineig_this
                        sum_cond_minima += hess_cond_this
                    elseif is_saddle
                        n_saddle += 1
                    elseif is_maximum
                        n_max += 1
                    elseif is_degenerate
                        n_deg += 1
                    else
                        n_deg += 1
                    end

                    angle_ok = is_minimum
                end

                if angle_ok
                    push!(seen_angles_raw, angle_key(θc; δ=angle_bin, flip_equiv=false))
                    push!(seen_angles_reduced, angle_key(θc; δ=angle_bin, flip_equiv=angle_flip_equiv))
                end

                sum_gn_final += sol.grad_norm
                min_gn_final = min(min_gn_final, sol.grad_norm)

                gni   = get(sol.metadata, :gn_init, NaN)
                gnm   = get(sol.metadata, :gn_max_inner, NaN)
                gmean = get(sol.metadata, :gn_mean_inner, NaN)

                dz = get(sol.metadata, :devZ_abs, NaN)
                dθ = get(sol.metadata, :devTheta_abs, NaN)

                sum_gn_init += gni
                sum_gn_maxinner += gnm
                sum_gn_meaninner += gmean

                sum_devZ_abs += dz
                sum_devTheta_abs += dθ

                sused = get(sol.metadata, :inner_sweeps_used, -1)
                sum_inner_sweeps_used += sused

                sum_rt += dt

                if compute_hessian
                    ihmig = Float64(get(sol.metadata, :init_hess_mineig, NaN))
                    ihcon = Float64(get(sol.metadata, :init_hess_cond, NaN))

                    if isfinite(ihmig)
                        sum_init_hess_mineig += ihmig
                        cnt_init_hess_mineig += 1
                    end

                    if isfinite(ihcon)
                        sum_init_hess_cond += ihcon
                        cnt_init_hess_cond += 1
                    end
                end

                if opt !== nothing
                    cut_hat, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
                    best_r = max(best_r, ratio)
                    sum_r += ratio
                    if ratio >= success_thr
                        succ += 1
                    end

                    if angle_ok && is_exact_optimum(cut_hat, opt)
                        key_raw = angle_key(θc; δ=angle_bin, flip_equiv=false)
                        key_red = angle_key(θc; δ=angle_bin, flip_equiv=angle_flip_equiv)

                        push!(seen_angles_opt_raw, key_raw)
                        push!(seen_angles_opt_reduced, key_red)
                        counts_opt_min_reduced[key_red] = get(counts_opt_min_reduced, key_red, 0) + 1

                        if isfinite(hess_cond_this)
                            push!(all_opt_min_hess_conds[iλ], hess_cond_this)
                            nκ_opt += 1
                        end
                    end
                end
            end

            unique_count[iλ] = length(seen_spins)
            unique_angle_count_raw[iλ] = length(seen_angles_raw)
            unique_angle_count_reduced[iλ] = length(seen_angles_reduced)

            unique_angle_opt_count_raw[iλ] = length(seen_angles_opt_raw)
            unique_angle_opt_count_reduced[iλ] = length(seen_angles_opt_reduced)

            # convert distinct-optimal-minimum hit counts into empirical probabilities
            if !isempty(counts_opt_min_reduced)
                probs = Float64[]
                sizehint!(probs, length(counts_opt_min_reduced))
                for c in values(counts_opt_min_reduced)
                    push!(probs, c / n_inits)
                end
                graph_opt_min_probs[iλ] = sort(probs; rev=true)
            end

            gn_final_mean[iλ] = sum_gn_final / n_inits
            gn_final_min[iλ] = min_gn_final

            gn_init_mean[iλ] = sum_gn_init / n_inits
            gn_maxinner_mean[iλ] = sum_gn_maxinner / n_inits
            gn_meaninner_mean[iλ] = sum_gn_meaninner / n_inits

            devZ_abs_mean[iλ] = sum_devZ_abs / n_inits
            devTheta_abs_mean[iλ] = sum_devTheta_abs / n_inits

            inner_sweeps_used_mean[iλ] = sum_inner_sweeps_used / n_inits
            rt_mean[iλ] = sum_rt / n_inits

            if compute_hessian
                hess_frac_min[iλ] = n_min / n_inits
                hess_frac_saddle[iλ] = n_saddle / n_inits
                hess_frac_max[iλ] = n_max / n_inits
                hess_frac_degenerate[iλ] = n_deg / n_inits
                hess_mineig_mean[iλ] = (n_min > 0) ? (sum_mineig_minima / n_min) : NaN
                hess_cond_mean[iλ] = (n_min > 0) ? (sum_cond_minima / n_min) : NaN

                init_hess_mineig_mean[iλ] = (cnt_init_hess_mineig > 0) ? (sum_init_hess_mineig / cnt_init_hess_mineig) : NaN
                init_hess_cond_mean[iλ]   = (cnt_init_hess_cond > 0) ? (sum_init_hess_cond / cnt_init_hess_cond) : NaN
            end

            if opt === nothing
                best_ratio[iλ] = nothing
                mean_ratio[iλ] = nothing
                success_rate[iλ] = 0.0
            else
                best_ratio[iλ] = best_r
                mean_ratio[iλ] = sum_r / n_inits
                success_rate[iλ] = succ / n_inits
            end

            if compute_hessian
                @printf("    λ=%.4f  angle_min_red=%d  N*_red=%d  nκ*(pts)=%d  np*(mins)=%d  Psolved(graph)=%.6f\n",
                    λ,
                    Int(unique_angle_count_reduced[iλ]),
                    Int(unique_angle_opt_count_reduced[iλ]),
                    nκ_opt,
                    length(graph_opt_min_probs[iλ]),
                    success_rate[iλ])
            else
                @printf("    λ=%.4f  angles_reduced=%d  N*_red=%d  np*(mins)=%d  Psolved(graph)=%.6f\n",
                    λ,
                    Int(unique_angle_count_reduced[iλ]),
                    Int(unique_angle_opt_count_reduced[iλ]),
                    length(graph_opt_min_probs[iλ]),
                    success_rate[iλ])
            end
        end

        for iλ in 1:nλ
            push!(all_opt_min_probs_by_graph[iλ], graph_opt_min_probs[iλ])
        end

        for iλ in 1:nλ
            u = unique_count[iλ]
            sum_unique[iλ]  += u
            sum_unique2[iλ] += u^2

            uar = unique_angle_count_raw[iλ]
            sum_unique_angle_raw[iλ]  += uar
            sum_unique_angle_raw2[iλ] += uar^2

            uared = unique_angle_count_reduced[iλ]
            sum_unique_angle_reduced[iλ]  += uared
            sum_unique_angle_reduced2[iλ] += uared^2

            uopt_raw = unique_angle_opt_count_raw[iλ]
            sum_unique_angle_opt_raw[iλ]  += uopt_raw
            sum_unique_angle_opt_raw2[iλ] += uopt_raw^2

            uopt_red = unique_angle_opt_count_reduced[iλ]
            sum_unique_angle_opt_reduced[iλ]  += uopt_red
            sum_unique_angle_opt_reduced2[iλ] += uopt_red^2

            gf = gn_final_mean[iλ]
            sum_gn_final_mean[iλ]  += gf
            sum_gn_final_mean2[iλ] += gf^2

            gfm = gn_final_min[iλ]
            sum_gn_final_min[iλ]  += gfm
            sum_gn_final_min2[iλ] += gfm^2

            gni = gn_init_mean[iλ]
            sum_gn_init_mean[iλ]  += gni
            sum_gn_init_mean2[iλ] += gni^2

            gnm = gn_maxinner_mean[iλ]
            sum_gn_maxinner_mean[iλ]  += gnm
            sum_gn_maxinner_mean2[iλ] += gnm^2

            gmean = gn_meaninner_mean[iλ]
            sum_gn_meaninner_mean[iλ]  += gmean
            sum_gn_meaninner_mean2[iλ] += gmean^2

            dz = devZ_abs_mean[iλ]
            sum_devZ_abs_mean[iλ]  += dz
            sum_devZ_abs_mean2[iλ] += dz^2

            dθ = devTheta_abs_mean[iλ]
            sum_devTheta_abs_mean[iλ]  += dθ
            sum_devTheta_abs_mean2[iλ] += dθ^2

            isw = inner_sweeps_used_mean[iλ]
            sum_inner_sweeps_mean[iλ]  += isw
            sum_inner_sweeps_mean2[iλ] += isw^2

            rm = rt_mean[iλ]
            sum_rt_mean[iλ]  += rm
            sum_rt_mean2[iλ] += rm^2

            if compute_hessian
                hfmin = hess_frac_min[iλ]
                hfsad = hess_frac_saddle[iλ]
                hfmax = hess_frac_max[iλ]
                hfdeg = hess_frac_degenerate[iλ]
                hmig = hess_mineig_mean[iλ]
                hcon = hess_cond_mean[iλ]
                ihmig = init_hess_mineig_mean[iλ]
                ihcon = init_hess_cond_mean[iλ]

                if isfinite(hfmin)
                    sum_hess_frac_min[iλ]  += hfmin
                    sum_hess_frac_min2[iλ] += hfmin^2
                end
                if isfinite(hfsad)
                    sum_hess_frac_saddle[iλ]  += hfsad
                    sum_hess_frac_saddle2[iλ] += hfsad^2
                end
                if isfinite(hfmax)
                    sum_hess_frac_max[iλ]  += hfmax
                    sum_hess_frac_max2[iλ] += hfmax^2
                end
                if isfinite(hfdeg)
                    sum_hess_frac_degenerate[iλ]  += hfdeg
                    sum_hess_frac_degenerate2[iλ] += hfdeg^2
                end
                if isfinite(hmig)
                    sum_hess_mineig_mean[iλ]  += hmig
                    sum_hess_mineig_mean2[iλ] += hmig^2
                end
                if isfinite(hcon)
                    sum_hess_cond_mean[iλ]  += hcon
                    sum_hess_cond_mean2[iλ] += hcon^2
                end
                if isfinite(ihmig)
                    sum_init_hess_mineig_mean[iλ]  += ihmig
                    sum_init_hess_mineig_mean2[iλ] += ihmig^2
                    n_init_hess_mineig_graphs[iλ] += 1
                end
                if isfinite(ihcon)
                    sum_init_hess_cond_mean[iλ]  += ihcon
                    sum_init_hess_cond_mean2[iλ] += ihcon^2
                    n_init_hess_cond_graphs[iλ] += 1
                end
            end

            if best_ratio[iλ] !== nothing
                br = Float64(best_ratio[iλ])
                mr = Float64(mean_ratio[iλ])

                sum_best_ratio[iλ]  += br
                sum_best_ratio2[iλ] += br^2

                sum_mean_ratio[iλ]  += mr
                sum_mean_ratio2[iλ] += mr^2

                n_ratio[iλ] += 1

                sr = success_rate[iλ]
                sum_succ[iλ]  += sr
                sum_succ2[iλ] += sr^2
                n_succ[iλ] += 1
            end
        end

        n_graphs += 1

        ckpt = Dict(
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "graph_seeds" => graph_seeds,
            "λs" => λs,
            "n_inits" => n_inits,
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "init_sig" => init_sig,
            "seed_salt" => seed_salt,
            "success_thr" => success_thr,
            "angle_bin" => angle_bin,
            "angle_flip_equiv" => angle_flip_equiv,
            "compute_hessian" => compute_hessian,
            "hessian_tol" => hessian_tol,
            "n_graphs" => n_graphs,
            "next_seed_index" => idx + 1,
            "sum_unique" => sum_unique,
            "sum_unique2" => sum_unique2,
            "sum_unique_angle_raw" => sum_unique_angle_raw,
            "sum_unique_angle_raw2" => sum_unique_angle_raw2,
            "sum_unique_angle_reduced" => sum_unique_angle_reduced,
            "sum_unique_angle_reduced2" => sum_unique_angle_reduced2,
            "sum_unique_angle_opt_raw" => sum_unique_angle_opt_raw,
            "sum_unique_angle_opt_raw2" => sum_unique_angle_opt_raw2,
            "sum_unique_angle_opt_reduced" => sum_unique_angle_opt_reduced,
            "sum_unique_angle_opt_reduced2" => sum_unique_angle_opt_reduced2,
            "sum_best_ratio" => sum_best_ratio,
            "sum_best_ratio2" => sum_best_ratio2,
            "sum_mean_ratio" => sum_mean_ratio,
            "sum_mean_ratio2" => sum_mean_ratio2,
            "n_ratio" => n_ratio,
            "sum_succ" => sum_succ,
            "sum_succ2" => sum_succ2,
            "n_succ" => n_succ,
            "sum_gn_final_mean" => sum_gn_final_mean,
            "sum_gn_final_mean2" => sum_gn_final_mean2,
            "sum_gn_final_min" => sum_gn_final_min,
            "sum_gn_final_min2" => sum_gn_final_min2,
            "sum_gn_init_mean" => sum_gn_init_mean,
            "sum_gn_init_mean2" => sum_gn_init_mean2,
            "sum_gn_maxinner_mean" => sum_gn_maxinner_mean,
            "sum_gn_maxinner_mean2" => sum_gn_maxinner_mean2,
            "sum_gn_meaninner_mean" => sum_gn_meaninner_mean,
            "sum_gn_meaninner_mean2" => sum_gn_meaninner_mean2,
            "sum_devZ_abs_mean" => sum_devZ_abs_mean,
            "sum_devZ_abs_mean2" => sum_devZ_abs_mean2,
            "sum_devTheta_abs_mean" => sum_devTheta_abs_mean,
            "sum_devTheta_abs_mean2" => sum_devTheta_abs_mean2,
            "sum_inner_sweeps_mean" => sum_inner_sweeps_mean,
            "sum_inner_sweeps_mean2" => sum_inner_sweeps_mean2,
            "sum_rt_mean" => sum_rt_mean,
            "sum_rt_mean2" => sum_rt_mean2,
            "sum_hess_frac_min" => sum_hess_frac_min,
            "sum_hess_frac_min2" => sum_hess_frac_min2,
            "sum_hess_frac_saddle" => sum_hess_frac_saddle,
            "sum_hess_frac_saddle2" => sum_hess_frac_saddle2,
            "sum_hess_frac_max" => sum_hess_frac_max,
            "sum_hess_frac_max2" => sum_hess_frac_max2,
            "sum_hess_frac_degenerate" => sum_hess_frac_degenerate,
            "sum_hess_frac_degenerate2" => sum_hess_frac_degenerate2,
            "sum_hess_mineig_mean" => sum_hess_mineig_mean,
            "sum_hess_mineig_mean2" => sum_hess_mineig_mean2,
            "sum_hess_cond_mean" => sum_hess_cond_mean,
            "sum_hess_cond_mean2" => sum_hess_cond_mean2,
            "sum_init_hess_mineig_mean" => sum_init_hess_mineig_mean,
            "sum_init_hess_mineig_mean2" => sum_init_hess_mineig_mean2,
            "n_init_hess_mineig_graphs" => n_init_hess_mineig_graphs,
            "sum_init_hess_cond_mean" => sum_init_hess_cond_mean,
            "sum_init_hess_cond_mean2" => sum_init_hess_cond_mean2,
            "n_init_hess_cond_graphs" => n_init_hess_cond_graphs,
            "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        )
        atomic_json_write(ckpt_path, ckpt)
    end

    # -----------------------
    # Final aggregation
    # -----------------------
    results = Vector{Dict}(undef, nλ)

    for (iλ, λ) in enumerate(λs)
        μu, eu = mean_and_stderr(sum_unique[iλ], sum_unique2[iλ], n_graphs)
        μuar, euar = mean_and_stderr(sum_unique_angle_raw[iλ], sum_unique_angle_raw2[iλ], n_graphs)
        μuared, euared = mean_and_stderr(sum_unique_angle_reduced[iλ], sum_unique_angle_reduced2[iλ], n_graphs)

        μuopt_raw, euopt_raw = mean_and_stderr(sum_unique_angle_opt_raw[iλ], sum_unique_angle_opt_raw2[iλ], n_graphs)
        μuopt_red, euopt_red = mean_and_stderr(sum_unique_angle_opt_reduced[iλ], sum_unique_angle_opt_reduced2[iλ], n_graphs)

        μgf, egf = mean_and_stderr(sum_gn_final_mean[iλ], sum_gn_final_mean2[iλ], n_graphs)
        μgfm, egfm = mean_and_stderr(sum_gn_final_min[iλ], sum_gn_final_min2[iλ], n_graphs)

        μgni, egni = mean_and_stderr(sum_gn_init_mean[iλ], sum_gn_init_mean2[iλ], n_graphs)
        μgnm, egnm = mean_and_stderr(sum_gn_maxinner_mean[iλ], sum_gn_maxinner_mean2[iλ], n_graphs)
        μgmean, egmean = mean_and_stderr(sum_gn_meaninner_mean[iλ], sum_gn_meaninner_mean2[iλ], n_graphs)

        μdz, edz = mean_and_stderr(sum_devZ_abs_mean[iλ], sum_devZ_abs_mean2[iλ], n_graphs)
        μdθ, edθ = mean_and_stderr(sum_devTheta_abs_mean[iλ], sum_devTheta_abs_mean2[iλ], n_graphs)

        μisw, eisw = mean_and_stderr(sum_inner_sweeps_mean[iλ], sum_inner_sweeps_mean2[iλ], n_graphs)
        μrt, ert = mean_and_stderr(sum_rt_mean[iλ], sum_rt_mean2[iλ], n_graphs)

        best_ratio_mean = "none"
        best_ratio_stderr = "none"
        mean_ratio_mean = "none"
        mean_ratio_stderr = "none"
        if n_ratio[iλ] > 0
            μbr, ebr = mean_and_stderr(sum_best_ratio[iλ], sum_best_ratio2[iλ], n_ratio[iλ])
            μmr, emr = mean_and_stderr(sum_mean_ratio[iλ], sum_mean_ratio2[iλ], n_ratio[iλ])
            best_ratio_mean = μbr
            best_ratio_stderr = ebr
            mean_ratio_mean = μmr
            mean_ratio_stderr = emr
        end

        succ_mean = "none"
        succ_stderr = "none"
        if n_succ[iλ] > 0
            μs, es = mean_and_stderr(sum_succ[iλ], sum_succ2[iλ], n_succ[iλ])
            succ_mean = μs
            succ_stderr = es
        end

        hess_frac_min_mean = "none"
        hess_frac_min_stderr = "none"
        hess_frac_saddle_mean = "none"
        hess_frac_saddle_stderr = "none"
        hess_frac_max_mean = "none"
        hess_frac_max_stderr = "none"
        hess_frac_degenerate_mean = "none"
        hess_frac_degenerate_stderr = "none"
        hess_mineig_mean_mean = "none"
        hess_mineig_mean_stderr = "none"
        hess_cond_mean_mean = "none"
        hess_cond_mean_stderr = "none"

        init_hess_mineig_mean_mean = "none"
        init_hess_mineig_mean_stderr = "none"
        init_hess_cond_mean_mean = "none"
        init_hess_cond_mean_stderr = "none"

        if compute_hessian
            μhfmin, ehfmin = mean_and_stderr(sum_hess_frac_min[iλ], sum_hess_frac_min2[iλ], n_graphs)
            μhfsad, ehfsad = mean_and_stderr(sum_hess_frac_saddle[iλ], sum_hess_frac_saddle2[iλ], n_graphs)
            μhfmax, ehfmax = mean_and_stderr(sum_hess_frac_max[iλ], sum_hess_frac_max2[iλ], n_graphs)
            μhfdeg, ehfdeg = mean_and_stderr(sum_hess_frac_degenerate[iλ], sum_hess_frac_degenerate2[iλ], n_graphs)
            μhmig, ehmig = mean_and_stderr(sum_hess_mineig_mean[iλ], sum_hess_mineig_mean2[iλ], n_graphs)
            μhcon, ehcon = mean_and_stderr(sum_hess_cond_mean[iλ], sum_hess_cond_mean2[iλ], n_graphs)

            hess_frac_min_mean = μhfmin
            hess_frac_min_stderr = ehfmin
            hess_frac_saddle_mean = μhfsad
            hess_frac_saddle_stderr = ehfsad
            hess_frac_max_mean = μhfmax
            hess_frac_max_stderr = ehfmax
            hess_frac_degenerate_mean = μhfdeg
            hess_frac_degenerate_stderr = ehfdeg
            hess_mineig_mean_mean = μhmig
            hess_mineig_mean_stderr = ehmig
            hess_cond_mean_mean = μhcon
            hess_cond_mean_stderr = ehcon

            if n_init_hess_mineig_graphs[iλ] > 0
                μihmig, eihmig = mean_and_stderr(
                    sum_init_hess_mineig_mean[iλ],
                    sum_init_hess_mineig_mean2[iλ],
                    n_init_hess_mineig_graphs[iλ]
                )
                init_hess_mineig_mean_mean = μihmig
                init_hess_mineig_mean_stderr = eihmig
            end

            if n_init_hess_cond_graphs[iλ] > 0
                μihcon, eihcon = mean_and_stderr(
                    sum_init_hess_cond_mean[iλ],
                    sum_init_hess_cond_mean2[iλ],
                    n_init_hess_cond_graphs[iλ]
                )
                init_hess_cond_mean_mean = μihcon
                init_hess_cond_mean_stderr = eihcon
            end
        end

        # pooled condition-number summaries across all graphs
        opt_cond_mean = "none"
        opt_cond_median = "none"
        n_opt_cond = 0

        κvals = all_opt_min_hess_conds[iλ]
        if !isempty(κvals)
            opt_cond_mean = mean(κvals)
            opt_cond_median = median(κvals)
            n_opt_cond = length(κvals)
        end

        # pooled raw probabilities across graphs
        opt_min_probs_raw = Float64[]
        for probs in all_opt_min_probs_by_graph[iλ]
            append!(opt_min_probs_raw, probs)
        end

        # same probabilities rescaled so their sum matches success_rate_mean
        opt_min_probs_mean_contrib = n_graphs > 0 ? [p / n_graphs for p in opt_min_probs_raw] : Float64[]

        results[iλ] = Dict(
            "λ" => λ,
            "n_graphs" => n_graphs,

            "unique_count_mean" => μu,
            "unique_count_stderr" => eu,

            "unique_angle_count_raw_mean" => μuar,
            "unique_angle_count_raw_stderr" => euar,

            "unique_angle_count_reduced_mean" => μuared,
            "unique_angle_count_reduced_stderr" => euared,

            # backward-compatible alias:
            "unique_angle_count_mean" => μuared,
            "unique_angle_count_stderr" => euared,

            # N*
            "unique_angle_opt_count_raw_mean" => μuopt_raw,
            "unique_angle_opt_count_raw_stderr" => euopt_raw,

            "unique_angle_opt_count_reduced_mean" => μuopt_red,
            "unique_angle_opt_count_reduced_stderr" => euopt_red,

            # convenient alias for plotted N*
            "Nstar_mean" => μuopt_red,
            "Nstar_stderr" => euopt_red,

            # summary statistics for optimal-minimum condition numbers
            "opt_min_hess_cond_mean" => opt_cond_mean,
            "opt_min_hess_cond_median" => opt_cond_median,
            "n_opt_min_hess_conds" => n_opt_cond,

            # full probability information for distinct optimal minima
            "opt_min_probs_raw" => opt_min_probs_raw,
            "opt_min_probs_mean_contrib" => opt_min_probs_mean_contrib,
            "opt_min_probs_by_graph" => all_opt_min_probs_by_graph[iλ],
            "n_opt_min_probs_raw" => length(opt_min_probs_raw),

            "angle_bin" => angle_bin,
            "angle_flip_equiv" => angle_flip_equiv,

            "best_ratio_mean" => best_ratio_mean,
            "best_ratio_stderr" => best_ratio_stderr,
            "mean_ratio_mean" => mean_ratio_mean,
            "mean_ratio_stderr" => mean_ratio_stderr,

            "n_ratio" => n_ratio[iλ],

            "success_thr" => success_thr,
            "success_rate_mean" => succ_mean,
            "success_rate_stderr" => succ_stderr,
            "n_succ" => n_succ[iλ],

            "grad_norm_final_mean" => μgf,
            "grad_norm_final_stderr" => egf,

            "grad_norm_final_min_mean" => μgfm,
            "grad_norm_final_min_stderr" => egfm,

            "grad_norm_init_mean" => μgni,
            "grad_norm_init_stderr" => egni,

            "grad_norm_maxinner_mean" => μgnm,
            "grad_norm_maxinner_stderr" => egnm,

            "grad_norm_meaninner_mean" => μgmean,
            "grad_norm_meaninner_stderr" => egmean,

            "devZ_abs_mean" => μdz,
            "devZ_abs_stderr" => edz,

            "devTheta_abs_mean" => μdθ,
            "devTheta_abs_stderr" => edθ,

            "inner_sweeps_used_mean" => μisw,
            "inner_sweeps_used_stderr" => eisw,

            "runtime_mean_s" => μrt,
            "runtime_mean_stderr_s" => ert,

            "compute_hessian" => compute_hessian,
            "hessian_tol" => hessian_tol,

            "hess_frac_min_mean" => hess_frac_min_mean,
            "hess_frac_min_stderr" => hess_frac_min_stderr,

            "hess_frac_saddle_mean" => hess_frac_saddle_mean,
            "hess_frac_saddle_stderr" => hess_frac_saddle_stderr,

            "hess_frac_max_mean" => hess_frac_max_mean,
            "hess_frac_max_stderr" => hess_frac_max_stderr,

            "hess_frac_degenerate_mean" => hess_frac_degenerate_mean,
            "hess_frac_degenerate_stderr" => hess_frac_degenerate_stderr,

            "hess_mineig_mean_mean" => hess_mineig_mean_mean,
            "hess_mineig_mean_stderr" => hess_mineig_mean_stderr,

            "hess_cond_mean_mean" => hess_cond_mean_mean,
            "hess_cond_mean_stderr" => hess_cond_mean_stderr,

            "init_hess_mineig_mean_mean" => init_hess_mineig_mean_mean,
            "init_hess_mineig_mean_stderr" => init_hess_mineig_mean_stderr,

            "init_hess_cond_mean_mean" => init_hess_cond_mean_mean,
            "init_hess_cond_mean_stderr" => init_hess_cond_mean_stderr,

            "n_init_hess_mineig_graphs" => n_init_hess_mineig_graphs[iλ],
            "n_init_hess_cond_graphs" => n_init_hess_cond_graphs[iλ],
        )
    end

    save_data = Dict(
        "experiment" => "qiigs_lambda_sweep_unique_minima",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,
        "λs" => λs,
        "n_inits" => n_inits,
        "angle_conv" => angle_conv,
        "compute_hessian" => compute_hessian,
        "hessian_tol" => hessian_tol,

        "solver" => Dict(
            "name" => "grad",
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "save_params" => save_params,
            "init_sig" => init_sig,
            "seed_salt" => seed_salt,
            "compute_hessian" => compute_hessian,
            "hessian_tol" => hessian_tol,
            "outputs" => compute_hessian ?
                ["gn_final", "gn_init", "gn_max_inner", "gn_mean_inner",
                 "devZ_abs", "devTheta_abs", "inner_sweeps_used",
                 "hess_mineig", "hess_maxeig", "hess_cond", "hess_is_minimum", "hess_is_saddle",
                 "hess_is_maximum", "hess_is_degenerate",
                 "init_hess_mineig", "init_hess_maxeig", "init_hess_cond",
                 "init_hess_is_minimum", "init_hess_is_saddle",
                 "init_hess_is_maximum", "init_hess_is_degenerate"] :
                ["gn_final", "gn_init", "gn_max_inner", "gn_mean_inner",
                 "devZ_abs", "devTheta_abs", "inner_sweeps_used"],
        ),

        "angle_uniqueness" => Dict(
            "angle_bin" => angle_bin,
            "flip_equiv" => angle_flip_equiv,
            "wrap" => "mod_pi",
            "source" => "sol.metadata[:theta_converged]",
            "count_policy" => compute_hessian ? "count only angles classified as minima" : "count all converged angles",
            "reported_counts" => ["raw", "reduced"],
        ),

        "Nstar_definition" => Dict(
            "description" => "Distinct angle minima whose rounded spin configuration attains the exact optimal cut value",
            "source" => "sol.metadata[:theta_converged] filtered to Hessian minima, then checked via cut_hat against opt",
            "exact_optimum_rule" => "cut_hat == opt (or isapprox with atol=1e-12 for non-integer values)",
            "reported_counts" => ["raw", "reduced"],
            "recommended_plot_field" => "unique_angle_opt_count_reduced_mean",
        ),

        "optimal_minimum_hessian_summary" => Dict(
            "description" => "Summary statistics of Hessian condition numbers for minima whose rounded spin configuration attains the exact optimum",
            "saved_per_lambda_fields" => ["opt_min_hess_cond_mean", "opt_min_hess_cond_median", "n_opt_min_hess_conds"],
            "definition" => "Statistics are computed over the pooled set of all optimal-minimum condition numbers at fixed λ across all processed graphs",
            "use_case" => "Median or mean condition number vs lambda without storing the full cloud",
        ),

        "optimal_minimum_probability_clouds" => Dict(
            "description" => "Empirical basin probabilities of distinct optimal minima",
            "definition" => "For each graph and λ, each distinct reduced optimal minimum α gets p_α = hits_α / n_inits",
            "saved_per_lambda_fields" => ["opt_min_probs_raw", "opt_min_probs_mean_contrib", "opt_min_probs_by_graph"],
            "sum_rule_per_graph" => "sum(opt_min_probs_by_graph[g]) = success rate of graph g at that λ",
            "sum_rule_aggregated" => "sum(opt_min_probs_mean_contrib) = success_rate_mean at that λ",
        ),

        "angle_space_diagnostics" => Dict(
            "source" => "sol.metadata from accepted θ_best",
            "devZ_abs" => "mean_i |cos(2θ_i)|",
            "devTheta_abs" => "mean_i |0.5*acos(|cos(2θ_i)|) - π/4|",
            "devTheta_abs_range" => [0.0, pi / 4],
        ),

        "init_hessian_diagnostics" => Dict(
            "source" => "sol.metadata from initial random θ_init",
            "mineig" => "minimum eigenvalue of Hessian at initial random state",
            "cond" => "condition number maxeig/mineig if initial state is a strict minimum, else NaN",
        ),

        "success_thr" => success_thr,

        "inputs" => Dict(
            "qiils_root" => ROOT_QIILS,
            "graphs_root" => GRAPHS_ROOT,
            "solutions_root" => SOLUTIONS_ROOT,
            "solution_key" => "maxcut_value",
        ),

        "results_per_lambda" => results,
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    abin_tag = replace(@sprintf("%.3g", angle_bin), "."=>"p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "."=>"p")
    tao_tag  = replace(@sprintf("%.3g", tao), "."=>"p")

    out_path = joinpath(
        save_dir,
        "qiigs_unique_ratio_meanbest_succ_grad_$(tag)_tao$(tao_tag)_conv$(conv_tag)_abin$(abin_tag)_$(hmode_tag)_htol$(htol_tag)_ngraphs$(n_graphs)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Aggregated results saved to: $out_path")
    println("✔ Checkpoint saved to: $ckpt_path")
    println("Done.")
end

main()