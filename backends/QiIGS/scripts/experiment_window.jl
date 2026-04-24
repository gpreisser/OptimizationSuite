# scripts/qiigs_transition_window_pergraph.jl
#
# Purpose:
#   Focused λ-scan around the transition window, saving PER-GRAPH / PER-λ data
#   so we can later make plots such as:
#     - all (graph, λ) points: y = P(success), color = mean_ratio
#     - one best-λ point per graph
#     - best-λ relative to graph-specific transition estimate
#
# What this saves:
#   1) Per-graph, per-λ data:
#        - success_rate
#        - mean_ratio
#        - best_ratio
#        - devTheta_abs_mean
#        - ΔdevTheta_abs_mean
#        - unique_count
#        - unique_angle_count_raw
#        - unique_angle_count_reduced
#        - optcurv diagnostics (if enabled)
#   2) Per-graph summary:
#        - λ maximizing P(success), with deterministic tie-breaking
#        - graph-specific transition λ estimates
#   3) Aggregated λ-level means/stderr across graphs
#
# Tie-breaking for best λ per graph:
#   1) maximize P(success)
#   2) among ties, maximize mean_ratio
#   3) among ties, choose smallest λ
#
# Transition markers saved per graph:
#   1) λ at argmax ΔdevTheta_abs_mean
#   2) first λ where devTheta_abs_mean > 0 (numerically)
#   3) first λ where devTheta_abs_mean exceeds a small threshold
#
# Output:
#   ROOT/results/qiigs_transition_window_N.../*.json

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

function total_edge_weight(W::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds for col in 1:size(W, 2)
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                s += W.nzval[idx]
            end
        end
    end
    return s
end

# If solver energy is E = sum_{i<j} w_ij s_i s_j and opt_cut is the MaxCut value,
# then cut = (Wtot - E)/2  =>  E_opt = Wtot - 2 * opt_cut.
function optimal_cut_to_ising_energy(W::SparseMatrixCSC{T}, opt_cut) where {T<:AbstractFloat}
    return total_edge_weight(W) - 2 * T(opt_cut)
end

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

function as_json_num_or_none(x)
    return isfinite(x) ? x : "none"
end

function argbest_transition_point(λs, ps, meanr)
    best_i = 0
    best_ps = -Inf
    best_mr = -Inf
    best_λ = Inf

    for i in eachindex(λs)
        p = ps[i]
        r = meanr[i]
        λ = λs[i]

        if !isfinite(p)
            continue
        end

        rr = isfinite(r) ? r : -Inf

        better =
            (p > best_ps) ||
            (p == best_ps && rr > best_mr) ||
            (p == best_ps && rr == best_mr && λ < best_λ)

        if better
            best_i = i
            best_ps = p
            best_mr = rr
            best_λ = λ
        end
    end

    return best_i
end

function argmax_finite(v::AbstractVector{<:Real})
    best_i = 0
    best_v = -Inf
    for i in eachindex(v)
        x = v[i]
        if isfinite(x) && x > best_v
            best_v = x
            best_i = i
        end
    end
    return best_i
end

function first_above_threshold(v::AbstractVector{<:Real}, thr::Real)
    for i in eachindex(v)
        x = v[i]
        if isfinite(x) && x > thr
            return i
        end
    end
    return 0
end

# ---------------------------
# Main
# ---------------------------

function main()
    println("====================================================================================")
    println("  QiIGS transition-window scan (per-graph + per-λ storage)                        ")
    println("  Goal: plot graph-specific success / ratio structure near the transition          ")
    println("====================================================================================")

    # -----------------------
    # Experiment parameters
    # -----------------------
    N = 50
    k = 3
    weighted = false

    graph_seeds = collect(1:50)

    λstart = 0.15
    λstep  = 0.0125
    λstop  = 0.5
    λs = collect(λstart:λstep:λstop)
    nλ = length(λs)

    n_inits = 10000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-8
    init_mode = :uniform

    save_params = true
    success_thr = 0.999

    angle_bin = 1e-2
    angle_flip_equiv = true

    # Transition marker threshold for devTheta_abs_mean
    devTheta_transition_threshold = 0.02

    # old generic hessian mode off here; this run is about transition-window behavior
    compute_hessian = false
    hessian_tol = 1e-8

    compute_optimum_curvature = true
    optimal_energy_atol = 1e-9

    ckpt_namespace = "v10_transition_window"

    hmode_tag = compute_hessian ? "hess_on" : "hess_off"
    htol_tag = replace(@sprintf("%.3g", hessian_tol), "." => "p")
    optcurv_tag = compute_optimum_curvature ? "optcurv_on" : "optcurv_off"
    optetol_tag = replace(@sprintf("%.3g", optimal_energy_atol), "." => "p")
    λwindow_tag = "lam$(replace(@sprintf("%.4f", λstart), "."=>"p"))_to_$(replace(@sprintf("%.4f", λstop), "."=>"p"))_d$(replace(@sprintf("%.4f", λstep), "."=>"p"))"

    init_sig = "ns=$(ckpt_namespace)_init=$(String(init_mode))_tao=$(tao)_conv=$(angle_conv)_save=$(save_params)_abin=$(angle_bin)_aflip=$(angle_flip_equiv)_$(hmode_tag)_htol$(htol_tag)_$(optcurv_tag)_oeatol$(optetol_tag)"
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
        "qiigs_transition_window_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(wtag)"
    )
    mkpath(save_dir)

    ckpt_path = joinpath(
        save_dir,
        "checkpoint_transition_window_$(λwindow_tag)_$(init_sig)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    @info "QiIGS ROOT" ROOT
    @info "Saving under" save_dir
    @info "Reusing graphs from" GRAPHS_ROOT
    @info "Reusing solutions from" SOLUTIONS_ROOT
    @info "Checkpoint" ckpt_path

    # -----------------------
    # Accumulators (aggregated over graphs at fixed λ)
    # -----------------------
    sum_unique  = zeros(Float64, nλ)
    sum_unique2 = zeros(Float64, nλ)

    sum_unique_angle_raw  = zeros(Float64, nλ)
    sum_unique_angle_raw2 = zeros(Float64, nλ)

    sum_unique_angle_reduced  = zeros(Float64, nλ)
    sum_unique_angle_reduced2 = zeros(Float64, nλ)

    sum_best_ratio  = zeros(Float64, nλ)
    sum_best_ratio2 = zeros(Float64, nλ)

    sum_mean_ratio  = zeros(Float64, nλ)
    sum_mean_ratio2 = zeros(Float64, nλ)

    n_ratio = zeros(Int, nλ)

    sum_succ  = zeros(Float64, nλ)
    sum_succ2 = zeros(Float64, nλ)
    n_succ    = zeros(Int, nλ)

    sum_devTheta_abs_mean  = zeros(Float64, nλ)
    sum_devTheta_abs_mean2 = zeros(Float64, nλ)

    sum_ΔdevTheta_abs_mean  = zeros(Float64, nλ)
    sum_ΔdevTheta_abs_mean2 = zeros(Float64, nλ)
    n_ΔdevTheta = zeros(Int, nλ)

    sum_optcurv_reached_frac  = zeros(Float64, nλ)
    sum_optcurv_reached_frac2 = zeros(Float64, nλ)

    sum_optcurv_hess_minpos_mean  = zeros(Float64, nλ)
    sum_optcurv_hess_minpos_mean2 = zeros(Float64, nλ)
    n_optcurv_hess_minpos_graphs  = zeros(Int, nλ)

    sum_optcurv_hess_cond_pos_mean  = zeros(Float64, nλ)
    sum_optcurv_hess_cond_pos_mean2 = zeros(Float64, nλ)
    n_optcurv_hess_cond_pos_graphs  = zeros(Int, nλ)

    # Per-graph saved data
    per_graph_results = Vector{Dict}()
    per_graph_best_points = Vector{Dict}()

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
           get(ck, "compute_optimum_curvature", false) == compute_optimum_curvature

            sum_unique  .= Float64.(ck["sum_unique"])
            sum_unique2 .= Float64.(ck["sum_unique2"])

            sum_unique_angle_raw  .= Float64.(ck["sum_unique_angle_raw"])
            sum_unique_angle_raw2 .= Float64.(ck["sum_unique_angle_raw2"])

            sum_unique_angle_reduced  .= Float64.(ck["sum_unique_angle_reduced"])
            sum_unique_angle_reduced2 .= Float64.(ck["sum_unique_angle_reduced2"])

            sum_best_ratio  .= Float64.(ck["sum_best_ratio"])
            sum_best_ratio2 .= Float64.(ck["sum_best_ratio2"])

            sum_mean_ratio  .= Float64.(ck["sum_mean_ratio"])
            sum_mean_ratio2 .= Float64.(ck["sum_mean_ratio2"])

            n_ratio .= Int.(ck["n_ratio"])

            sum_succ  .= Float64.(ck["sum_succ"])
            sum_succ2 .= Float64.(ck["sum_succ2"])
            n_succ    .= Int.(ck["n_succ"])

            sum_devTheta_abs_mean  .= Float64.(ck["sum_devTheta_abs_mean"])
            sum_devTheta_abs_mean2 .= Float64.(ck["sum_devTheta_abs_mean2"])

            sum_ΔdevTheta_abs_mean  .= Float64.(ck["sum_ΔdevTheta_abs_mean"])
            sum_ΔdevTheta_abs_mean2 .= Float64.(ck["sum_ΔdevTheta_abs_mean2"])
            n_ΔdevTheta .= Int.(ck["n_ΔdevTheta"])

            sum_optcurv_reached_frac  .= Float64.(ck["sum_optcurv_reached_frac"])
            sum_optcurv_reached_frac2 .= Float64.(ck["sum_optcurv_reached_frac2"])

            sum_optcurv_hess_minpos_mean  .= Float64.(ck["sum_optcurv_hess_minpos_mean"])
            sum_optcurv_hess_minpos_mean2 .= Float64.(ck["sum_optcurv_hess_minpos_mean2"])
            n_optcurv_hess_minpos_graphs  .= Int.(ck["n_optcurv_hess_minpos_graphs"])

            sum_optcurv_hess_cond_pos_mean  .= Float64.(ck["sum_optcurv_hess_cond_pos_mean"])
            sum_optcurv_hess_cond_pos_mean2 .= Float64.(ck["sum_optcurv_hess_cond_pos_mean2"])
            n_optcurv_hess_cond_pos_graphs  .= Int.(ck["n_optcurv_hess_cond_pos_graphs"])

            if haskey(ck, "per_graph_results")
                per_graph_results = Vector{Dict}(ck["per_graph_results"])
            end
            if haskey(ck, "per_graph_best_points")
                per_graph_best_points = Vector{Dict}(ck["per_graph_best_points"])
            end

            n_graphs = Int(ck["n_graphs"])
            start_index = Int(ck["next_seed_index"])
            println("↻ Resuming: n_graphs=$n_graphs, next_seed_index=$start_index")
        else
            println("⚠ Found checkpoint but parameters differ; ignoring it.")
        end
    end

    # -----------------------
    # Outer loop over graphs
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

        opt_energy = (opt === nothing) ? nothing : optimal_cut_to_ising_energy(W, opt)

        unique_count = zeros(Float64, nλ)
        unique_angle_count_raw = zeros(Float64, nλ)
        unique_angle_count_reduced = zeros(Float64, nλ)

        best_ratio = fill(NaN, nλ)
        mean_ratio = fill(NaN, nλ)
        success_rate = fill(NaN, nλ)

        devTheta_abs_mean = fill(NaN, nλ)
        ΔdevTheta_abs_mean = fill(NaN, nλ)

        optcurv_reached_frac = fill(NaN, nλ)
        optcurv_hess_minpos_mean = fill(NaN, nλ)
        optcurv_hess_cond_pos_mean = fill(NaN, nλ)

        prev_devTheta = NaN

        for (iλ, λ) in enumerate(λs)
            seen_spins = Set{UInt64}()
            seen_angles_raw = Set{keyT}()
            seen_angles_reduced = Set{keyT}()

            sum_devTheta = 0.0
            best_r = -Inf
            sum_r = 0.0
            succ = 0

            n_optcurv_reached = 0
            n_optcurv_minpos = 0
            n_optcurv_condpos = 0
            sum_optcurv_minpos = 0.0
            sum_optcurv_condpos = 0.0

            for r in 1:n_inits
                run_seed = gs * 1_000_000 + r * 10_000 + Int(round(λ * 10_000)) + seed_salt * 1_000_000_000

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
                    compute_optimum_curvature = compute_optimum_curvature,
                    optimal_energy = opt_energy,
                    optimal_energy_atol = optimal_energy_atol,
                )

                push!(seen_spins, QiIGS.spin_config_key(sol.configuration))

                θc = get(sol.metadata, :theta_converged, Float64[])
                isempty(θc) && error("Missing :theta_converged in sol.metadata")

                push!(seen_angles_raw, angle_key(θc; δ=angle_bin, flip_equiv=false))
                push!(seen_angles_reduced, angle_key(θc; δ=angle_bin, flip_equiv=angle_flip_equiv))

                dθ = Float64(get(sol.metadata, :devTheta_abs, NaN))
                sum_devTheta += dθ

                if opt !== nothing
                    _, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
                    best_r = max(best_r, ratio)
                    sum_r += ratio
                    if ratio >= success_thr
                        succ += 1
                    end
                end

                if compute_optimum_curvature
                    reached = get(sol.metadata, :optimal_energy_reached, false)
                    if reached
                        n_optcurv_reached += 1

                        v_minpos = Float64(get(sol.metadata, :optcurv_hess_minpos, NaN))
                        v_condpos = Float64(get(sol.metadata, :optcurv_hess_cond_pos, NaN))

                        if isfinite(v_minpos)
                            sum_optcurv_minpos += v_minpos
                            n_optcurv_minpos += 1
                        end
                        if isfinite(v_condpos)
                            sum_optcurv_condpos += v_condpos
                            n_optcurv_condpos += 1
                        end
                    end
                end
            end

            unique_count[iλ] = length(seen_spins)
            unique_angle_count_raw[iλ] = length(seen_angles_raw)
            unique_angle_count_reduced[iλ] = length(seen_angles_reduced)

            devTheta_abs_mean[iλ] = sum_devTheta / n_inits
            ΔdevTheta_abs_mean[iλ] = (iλ == 1 || !isfinite(prev_devTheta)) ? NaN : (devTheta_abs_mean[iλ] - prev_devTheta)
            prev_devTheta = devTheta_abs_mean[iλ]

            if opt !== nothing
                best_ratio[iλ] = best_r
                mean_ratio[iλ] = sum_r / n_inits
                success_rate[iλ] = succ / n_inits
            end

            if compute_optimum_curvature
                optcurv_reached_frac[iλ] = n_optcurv_reached / n_inits
                optcurv_hess_minpos_mean[iλ] = (n_optcurv_minpos > 0) ? (sum_optcurv_minpos / n_optcurv_minpos) : NaN
                optcurv_hess_cond_pos_mean[iλ] = (n_optcurv_condpos > 0) ? (sum_optcurv_condpos / n_optcurv_condpos) : NaN
            end

            @printf("    λ=%.4f  uniques(spin)=%d  angles_raw=%d  angles_reduced=%d  psucc=%.4f  mean_ratio=%s  devTheta=%.6f  ΔdevTheta=%s\n",
                λ,
                Int(unique_count[iλ]),
                Int(unique_angle_count_raw[iλ]),
                Int(unique_angle_count_reduced[iλ]),
                isfinite(success_rate[iλ]) ? success_rate[iλ] : NaN,
                isfinite(mean_ratio[iλ]) ? @sprintf("%.4f", mean_ratio[iλ]) : "NaN",
                isfinite(devTheta_abs_mean[iλ]) ? devTheta_abs_mean[iλ] : NaN,
                isfinite(ΔdevTheta_abs_mean[iλ]) ? @sprintf("%.6e", ΔdevTheta_abs_mean[iλ]) : "NaN")
        end

        # -----------------------
        # Graph-specific transition markers
        # -----------------------
        itr_argmax_delta = argmax_finite(ΔdevTheta_abs_mean)
        itr_first_nonzero = first_above_threshold(devTheta_abs_mean, 1e-12)
        itr_first_thresh = first_above_threshold(devTheta_abs_mean, devTheta_transition_threshold)

        λ_transition_argmax_delta =
            itr_argmax_delta > 0 ? λs[itr_argmax_delta] : NaN
        delta_devTheta_max =
            itr_argmax_delta > 0 ? ΔdevTheta_abs_mean[itr_argmax_delta] : NaN

        λ_transition_first_nonzero =
            itr_first_nonzero > 0 ? λs[itr_first_nonzero] : NaN

        λ_transition_first_threshold =
            itr_first_thresh > 0 ? λs[itr_first_thresh] : NaN

        # Save per-graph detailed curve
        graph_curve = Vector{Dict}(undef, nλ)
        for iλ in 1:nλ
            graph_curve[iλ] = Dict(
                "λ" => λs[iλ],
                "unique_count" => unique_count[iλ],
                "unique_angle_count_raw" => unique_angle_count_raw[iλ],
                "unique_angle_count_reduced" => unique_angle_count_reduced[iλ],
                "best_ratio" => as_json_num_or_none(best_ratio[iλ]),
                "mean_ratio" => as_json_num_or_none(mean_ratio[iλ]),
                "success_rate" => as_json_num_or_none(success_rate[iλ]),
                "devTheta_abs_mean" => as_json_num_or_none(devTheta_abs_mean[iλ]),
                "ΔdevTheta_abs_mean" => as_json_num_or_none(ΔdevTheta_abs_mean[iλ]),
                "optcurv_reached_frac" => as_json_num_or_none(optcurv_reached_frac[iλ]),
                "optcurv_hess_minpos_mean" => as_json_num_or_none(optcurv_hess_minpos_mean[iλ]),
                "optcurv_hess_cond_pos_mean" => as_json_num_or_none(optcurv_hess_cond_pos_mean[iλ]),
            )
        end

        push!(per_graph_results, Dict(
            "graph_seed" => gs,
            "optimal_cut" => opt === nothing ? "none" : opt,
            "lambda_transition_argmax_delta_devTheta" => as_json_num_or_none(λ_transition_argmax_delta),
            "delta_devTheta_max" => as_json_num_or_none(delta_devTheta_max),
            "lambda_transition_first_nonzero_devTheta" => as_json_num_or_none(λ_transition_first_nonzero),
            "lambda_transition_first_devTheta_threshold" => as_json_num_or_none(λ_transition_first_threshold),
            "devTheta_threshold_used" => devTheta_transition_threshold,
            "results_per_lambda" => graph_curve,
        ))

        # Save best λ point for this graph
        ibest = argbest_transition_point(λs, success_rate, mean_ratio)
        if ibest > 0
            λbest = λs[ibest]

            push!(per_graph_best_points, Dict(
                "graph_seed" => gs,
                "optimal_cut" => opt === nothing ? "none" : opt,

                "lambda_best_psuccess" => λbest,
                "psuccess_best" => as_json_num_or_none(success_rate[ibest]),
                "mean_ratio_at_best_psuccess" => as_json_num_or_none(mean_ratio[ibest]),
                "best_ratio_at_best_psuccess" => as_json_num_or_none(best_ratio[ibest]),
                "devTheta_at_best_psuccess" => as_json_num_or_none(devTheta_abs_mean[ibest]),
                "delta_devTheta_at_best_psuccess" => as_json_num_or_none(ΔdevTheta_abs_mean[ibest]),
                "unique_count_at_best_psuccess" => unique_count[ibest],
                "unique_angle_count_raw_at_best_psuccess" => unique_angle_count_raw[ibest],

                "lambda_transition_argmax_delta_devTheta" => as_json_num_or_none(λ_transition_argmax_delta),
                "delta_devTheta_max" => as_json_num_or_none(delta_devTheta_max),

                "lambda_transition_first_nonzero_devTheta" => as_json_num_or_none(λ_transition_first_nonzero),
                "lambda_transition_first_devTheta_threshold" => as_json_num_or_none(λ_transition_first_threshold),
                "devTheta_threshold_used" => devTheta_transition_threshold,

                "lambda_best_minus_lambda_transition_argmax" =>
                    as_json_num_or_none(isfinite(λ_transition_argmax_delta) ? (λbest - λ_transition_argmax_delta) : NaN),

                "lambda_best_minus_lambda_transition_first_nonzero" =>
                    as_json_num_or_none(isfinite(λ_transition_first_nonzero) ? (λbest - λ_transition_first_nonzero) : NaN),

                "lambda_best_minus_lambda_transition_threshold" =>
                    as_json_num_or_none(isfinite(λ_transition_first_threshold) ? (λbest - λ_transition_first_threshold) : NaN),

                "tie_break_rule" => "maximize psuccess, then mean_ratio, then choose smallest lambda",
            ))
        end

        # Update accumulators
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

            dθ = devTheta_abs_mean[iλ]
            if isfinite(dθ)
                sum_devTheta_abs_mean[iλ]  += dθ
                sum_devTheta_abs_mean2[iλ] += dθ^2
            end

            ddθ = ΔdevTheta_abs_mean[iλ]
            if isfinite(ddθ)
                sum_ΔdevTheta_abs_mean[iλ]  += ddθ
                sum_ΔdevTheta_abs_mean2[iλ] += ddθ^2
                n_ΔdevTheta[iλ] += 1
            end

            if isfinite(best_ratio[iλ]) && isfinite(mean_ratio[iλ])
                br = best_ratio[iλ]
                mr = mean_ratio[iλ]

                sum_best_ratio[iλ]  += br
                sum_best_ratio2[iλ] += br^2

                sum_mean_ratio[iλ]  += mr
                sum_mean_ratio2[iλ] += mr^2

                n_ratio[iλ] += 1
            end

            if isfinite(success_rate[iλ])
                sr = success_rate[iλ]
                sum_succ[iλ]  += sr
                sum_succ2[iλ] += sr^2
                n_succ[iλ] += 1
            end

            if isfinite(optcurv_reached_frac[iλ])
                v = optcurv_reached_frac[iλ]
                sum_optcurv_reached_frac[iλ]  += v
                sum_optcurv_reached_frac2[iλ] += v^2
            end

            if isfinite(optcurv_hess_minpos_mean[iλ])
                v = optcurv_hess_minpos_mean[iλ]
                sum_optcurv_hess_minpos_mean[iλ]  += v
                sum_optcurv_hess_minpos_mean2[iλ] += v^2
                n_optcurv_hess_minpos_graphs[iλ] += 1
            end

            if isfinite(optcurv_hess_cond_pos_mean[iλ])
                v = optcurv_hess_cond_pos_mean[iλ]
                sum_optcurv_hess_cond_pos_mean[iλ]  += v
                sum_optcurv_hess_cond_pos_mean2[iλ] += v^2
                n_optcurv_hess_cond_pos_graphs[iλ] += 1
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
            "compute_optimum_curvature" => compute_optimum_curvature,
            "optimal_energy_atol" => optimal_energy_atol,
            "devTheta_transition_threshold" => devTheta_transition_threshold,
            "n_graphs" => n_graphs,
            "next_seed_index" => idx + 1,

            "sum_unique" => sum_unique,
            "sum_unique2" => sum_unique2,
            "sum_unique_angle_raw" => sum_unique_angle_raw,
            "sum_unique_angle_raw2" => sum_unique_angle_raw2,
            "sum_unique_angle_reduced" => sum_unique_angle_reduced,
            "sum_unique_angle_reduced2" => sum_unique_angle_reduced2,

            "sum_best_ratio" => sum_best_ratio,
            "sum_best_ratio2" => sum_best_ratio2,
            "sum_mean_ratio" => sum_mean_ratio,
            "sum_mean_ratio2" => sum_mean_ratio2,
            "n_ratio" => n_ratio,

            "sum_succ" => sum_succ,
            "sum_succ2" => sum_succ2,
            "n_succ" => n_succ,

            "sum_devTheta_abs_mean" => sum_devTheta_abs_mean,
            "sum_devTheta_abs_mean2" => sum_devTheta_abs_mean2,

            "sum_ΔdevTheta_abs_mean" => sum_ΔdevTheta_abs_mean,
            "sum_ΔdevTheta_abs_mean2" => sum_ΔdevTheta_abs_mean2,
            "n_ΔdevTheta" => n_ΔdevTheta,

            "sum_optcurv_reached_frac" => sum_optcurv_reached_frac,
            "sum_optcurv_reached_frac2" => sum_optcurv_reached_frac2,

            "sum_optcurv_hess_minpos_mean" => sum_optcurv_hess_minpos_mean,
            "sum_optcurv_hess_minpos_mean2" => sum_optcurv_hess_minpos_mean2,
            "n_optcurv_hess_minpos_graphs" => n_optcurv_hess_minpos_graphs,

            "sum_optcurv_hess_cond_pos_mean" => sum_optcurv_hess_cond_pos_mean,
            "sum_optcurv_hess_cond_pos_mean2" => sum_optcurv_hess_cond_pos_mean2,
            "n_optcurv_hess_cond_pos_graphs" => n_optcurv_hess_cond_pos_graphs,

            "per_graph_results" => per_graph_results,
            "per_graph_best_points" => per_graph_best_points,

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

        μdθ, edθ = mean_and_stderr(sum_devTheta_abs_mean[iλ], sum_devTheta_abs_mean2[iλ], n_graphs)

        Δmean = "none"
        Δstderr = "none"
        if n_ΔdevTheta[iλ] > 0
            μΔ, eΔ = mean_and_stderr(sum_ΔdevTheta_abs_mean[iλ], sum_ΔdevTheta_abs_mean2[iλ], n_ΔdevTheta[iλ])
            Δmean = μΔ
            Δstderr = eΔ
        end

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

        optcurv_reached_frac_mean = "none"
        optcurv_reached_frac_stderr = "none"
        optcurv_hess_minpos_mean_mean = "none"
        optcurv_hess_minpos_mean_stderr = "none"
        optcurv_hess_cond_pos_mean_mean = "none"
        optcurv_hess_cond_pos_mean_stderr = "none"

        if compute_optimum_curvature
            μocr, eocr = mean_and_stderr(sum_optcurv_reached_frac[iλ], sum_optcurv_reached_frac2[iλ], n_graphs)
            optcurv_reached_frac_mean = μocr
            optcurv_reached_frac_stderr = eocr

            if n_optcurv_hess_minpos_graphs[iλ] > 0
                μx, ex = mean_and_stderr(
                    sum_optcurv_hess_minpos_mean[iλ],
                    sum_optcurv_hess_minpos_mean2[iλ],
                    n_optcurv_hess_minpos_graphs[iλ],
                )
                optcurv_hess_minpos_mean_mean = μx
                optcurv_hess_minpos_mean_stderr = ex
            end

            if n_optcurv_hess_cond_pos_graphs[iλ] > 0
                μx, ex = mean_and_stderr(
                    sum_optcurv_hess_cond_pos_mean[iλ],
                    sum_optcurv_hess_cond_pos_mean2[iλ],
                    n_optcurv_hess_cond_pos_graphs[iλ],
                )
                optcurv_hess_cond_pos_mean_mean = μx
                optcurv_hess_cond_pos_mean_stderr = ex
            end
        end

        results[iλ] = Dict(
            "λ" => λ,
            "n_graphs" => n_graphs,

            "unique_count_mean" => μu,
            "unique_count_stderr" => eu,

            "unique_angle_count_raw_mean" => μuar,
            "unique_angle_count_raw_stderr" => euar,

            "unique_angle_count_reduced_mean" => μuared,
            "unique_angle_count_reduced_stderr" => euared,

            "best_ratio_mean" => best_ratio_mean,
            "best_ratio_stderr" => best_ratio_stderr,

            "mean_ratio_mean" => mean_ratio_mean,
            "mean_ratio_stderr" => mean_ratio_stderr,

            "success_rate_mean" => succ_mean,
            "success_rate_stderr" => succ_stderr,

            "devTheta_abs_mean" => μdθ,
            "devTheta_abs_stderr" => edθ,

            "ΔdevTheta_abs_mean" => Δmean,
            "ΔdevTheta_abs_stderr" => Δstderr,

            "optcurv_reached_frac_mean" => optcurv_reached_frac_mean,
            "optcurv_reached_frac_stderr" => optcurv_reached_frac_stderr,

            "optcurv_hess_minpos_mean_mean" => optcurv_hess_minpos_mean_mean,
            "optcurv_hess_minpos_mean_stderr" => optcurv_hess_minpos_mean_stderr,

            "optcurv_hess_cond_pos_mean_mean" => optcurv_hess_cond_pos_mean_mean,
            "optcurv_hess_cond_pos_mean_stderr" => optcurv_hess_cond_pos_mean_stderr,
        )
    end

    save_data = Dict(
        "experiment" => "qiigs_transition_window_pergraph",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,
        "λs" => λs,
        "n_inits" => n_inits,
        "success_thr" => success_thr,
        "angle_conv" => angle_conv,
        "compute_hessian" => compute_hessian,
        "hessian_tol" => hessian_tol,
        "compute_optimum_curvature" => compute_optimum_curvature,
        "optimal_energy_atol" => optimal_energy_atol,
        "devTheta_transition_threshold" => devTheta_transition_threshold,

        "transition_window" => Dict(
            "lambda_start" => λstart,
            "lambda_stop" => λstop,
            "lambda_step" => λstep,
            "purpose" => "save graph-specific transition-window data for cloud plots, best-lambda-per-graph summaries, and transition-relative analyses",
        ),

        "tie_break_rule_for_best_lambda" => "maximize psuccess, then mean_ratio, then choose smallest lambda",

        "transition_marker_definitions" => Dict(
            "lambda_transition_argmax_delta_devTheta" => "lambda where ΔdevTheta_abs_mean is maximal",
            "lambda_transition_first_nonzero_devTheta" => "first lambda where devTheta_abs_mean > 1e-12",
            "lambda_transition_first_devTheta_threshold" => "first lambda where devTheta_abs_mean exceeds devTheta_transition_threshold",
        ),

        "solver" => Dict(
            "name" => "grad",
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "save_params" => save_params,
            "compute_optimum_curvature" => compute_optimum_curvature,
        ),

        "per_graph_results" => per_graph_results,
        "per_graph_best_points" => per_graph_best_points,
        "results_per_lambda" => results,

        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    abin_tag = replace(@sprintf("%.3g", angle_bin), "."=>"p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "."=>"p")
    dthr_tag = replace(@sprintf("%.3g", devTheta_transition_threshold), "."=>"p")

    out_path = joinpath(
        save_dir,
        "qiigs_transition_window_pergraph_$(λwindow_tag)_conv$(conv_tag)_abin$(abin_tag)_dthresh$(dthr_tag)_$(hmode_tag)_htol$(htol_tag)_$(optcurv_tag)_oeatol$(optetol_tag)_ngraphs$(n_graphs)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Aggregated + per-graph results saved to: $out_path")
    println("✔ Checkpoint saved to: $ckpt_path")
    println("Done.")
end

main()