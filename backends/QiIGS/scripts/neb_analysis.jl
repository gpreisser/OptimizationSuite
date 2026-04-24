# scripts/find_two_minima_onegraph.jl
#
# Purpose:
#   For ONE graph seed, sweep over λ values and identify distinct continuous
#   angle-space minima found by the gradient solver.
#
# What it does:
#   - runs many random initializations for each λ
#   - keeps only runs classified by the Hessian as minima
#   - clusters minima in angle space using a quantized angle key
#   - stores one representative θ per cluster
#   - prints a compact summary so you can spot the first λ with 2 minima
#
# Output:
#   Saves a JSON file under ROOT/results/minima_scan_onegraph/
#
# Notes:
#   - This is a stripped-down script meant only to FIND candidate (graph, λ)
#     points with two minima.
#   - It does NOT do NEB yet.
#   - It uses theta_converged from solver metadata, so save_params=true is required.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using Dates
using SparseArrays
using Statistics

# ---------------------------
# Helpers
# ---------------------------

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, JSON.json(data; allownan=true))
    end
    mv(tmp, path; force=true)
    return nothing
end

function angle_key(
    θ::AbstractVector{<:AbstractFloat};
    δ::Float64 = 1e-3,
    flip_equiv::Bool = true,
)
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

function theta_distance_modpi(
    θ1::AbstractVector{<:AbstractFloat},
    θ2::AbstractVector{<:AbstractFloat},
)
    @assert length(θ1) == length(θ2)
    s = 0.0
    @inbounds for i in eachindex(θ1)
        d = abs(mod(Float64(θ1[i]) - Float64(θ2[i]), pi))
        d = min(d, pi - d)
        s += d^2
    end
    return sqrt(s)
end

function vec_to_json(v::AbstractVector{<:Real})
    return [Float64(x) for x in v]
end

# ---------------------------
# Main
# ---------------------------

function main()
    println("==============================================================")
    println("  Find candidate λ with two angle-space minima for ONE graph ")
    println("==============================================================")

    # -----------------------
    # Experiment parameters
    # -----------------------
    N = 50
    k = 3
    weighted = false

    graph_seed = 1

    # You can narrow this window later if needed.
    λs = collect(0.0:0.025:1.0)

    n_inits = 2000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-8
    init_mode = :uniform
    save_params = true

    # Keep only Hessian-classified minima
    compute_hessian = true
    hessian_tol = 1e-8

    # Angle clustering
    angle_bin = 1e-2
    angle_flip_equiv = true

    # Optional post-filter on final grad norm
    require_grad_norm_max = true
    grad_norm_max = 1e-6

    seed_salt = 0

    # -----------------------
    # Paths
    # -----------------------
    ROOT = normpath(joinpath(@__DIR__, ".."))
    RESULTS_DIR = joinpath(ROOT, "results")
    OUT_DIR = joinpath(RESULTS_DIR, "minima_scan_onegraph")
    mkpath(OUT_DIR)

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    wtag = weighted ? "weighted" : "unweighted"
    λmin = first(λs)
    λmax = last(λs)
    dλ = length(λs) > 1 ? (λs[2] - λs[1]) : 0.0

    λtag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"
    abin_tag = replace(@sprintf("%.3g", angle_bin), "."=>"p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "."=>"p")
    htol_tag = replace(@sprintf("%.3g", hessian_tol), "."=>"p")
    gn_tag = replace(@sprintf("%.3g", grad_norm_max), "."=>"p")

    out_path = joinpath(
        OUT_DIR,
        "onegraph_minima_scan_N$(N)_k$(k)_seed$(graph_seed)_$(wtag)_$(λtag)_conv$(conv_tag)_abin$(abin_tag)_aflip$(angle_flip_equiv)_hess_on_htol$(htol_tag)_gfilter$(require_grad_norm_max)_gnmax$(gn_tag)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations).json"
    )

    println("Graph path: $gpath")
    println("Output path: $out_path")
    println()

    # -----------------------
    # Run sweep
    # -----------------------
    results_per_lambda = Vector{Dict{String, Any}}()

    first_lambda_with_two_reduced = nothing
    first_lambda_with_two_raw = nothing

    for λ in λs
        println("--------------------------------------------------------------")
        @printf("λ = %.4f\n", λ)

        # key -> stored info
        raw_clusters = Dict{Any, Dict{String, Any}}()
        red_clusters = Dict{Any, Dict{String, Any}}()

        accepted_minima = 0
        rejected_not_minimum = 0
        rejected_grad = 0

        for r in 1:n_inits
            run_seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ * 1000)) + seed_salt * 1_000_000_000

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

            θ = get(sol.metadata, :theta_converged, Float64[])
            isempty(θ) && error("Missing :theta_converged in solver metadata")

            is_minimum = get(sol.metadata, :hess_is_minimum, false)
            if !is_minimum
                rejected_not_minimum += 1
                continue
            end

            gn = Float64(sol.grad_norm)
            if require_grad_norm_max && !(gn <= grad_norm_max)
                rejected_grad += 1
                continue
            end

            accepted_minima += 1

            mineig = Float64(get(sol.metadata, :hess_mineig, NaN))
            maxeig = Float64(get(sol.metadata, :hess_maxeig, NaN))
            cond = Float64(get(sol.metadata, :hess_cond, NaN))
            rounded_energy = Float64(sol.energy)

            kraw = angle_key(θ; δ=angle_bin, flip_equiv=false)
            kred = angle_key(θ; δ=angle_bin, flip_equiv=angle_flip_equiv)

            # raw clusters
            if !haskey(raw_clusters, kraw)
                raw_clusters[kraw] = Dict{String, Any}(
                    "count" => 0,
                    "representative_theta" => vec_to_json(θ),
                    "best_grad_norm" => gn,
                    "best_hess_mineig" => mineig,
                    "best_hess_maxeig" => maxeig,
                    "best_hess_cond" => cond,
                    "rounded_energy" => rounded_energy,
                )
            end
            raw_clusters[kraw]["count"] += 1
            if gn < raw_clusters[kraw]["best_grad_norm"]
                raw_clusters[kraw]["representative_theta"] = vec_to_json(θ)
                raw_clusters[kraw]["best_grad_norm"] = gn
                raw_clusters[kraw]["best_hess_mineig"] = mineig
                raw_clusters[kraw]["best_hess_maxeig"] = maxeig
                raw_clusters[kraw]["best_hess_cond"] = cond
                raw_clusters[kraw]["rounded_energy"] = rounded_energy
            end

            # reduced clusters
            if !haskey(red_clusters, kred)
                red_clusters[kred] = Dict{String, Any}(
                    "count" => 0,
                    "representative_theta" => vec_to_json(θ),
                    "best_grad_norm" => gn,
                    "best_hess_mineig" => mineig,
                    "best_hess_maxeig" => maxeig,
                    "best_hess_cond" => cond,
                    "rounded_energy" => rounded_energy,
                )
            end
            red_clusters[kred]["count"] += 1
            if gn < red_clusters[kred]["best_grad_norm"]
                red_clusters[kred]["representative_theta"] = vec_to_json(θ)
                red_clusters[kred]["best_grad_norm"] = gn
                red_clusters[kred]["best_hess_mineig"] = mineig
                red_clusters[kred]["best_hess_maxeig"] = maxeig
                red_clusters[kred]["best_hess_cond"] = cond
                red_clusters[kred]["rounded_energy"] = rounded_energy
            end
        end

        # sort clusters by frequency, descending
        raw_list = collect(values(raw_clusters))
        sort!(raw_list, by = x -> (-Int(x["count"]), Float64(x["best_grad_norm"])))

        red_list = collect(values(red_clusters))
        sort!(red_list, by = x -> (-Int(x["count"]), Float64(x["best_grad_norm"])))

        nraw = length(raw_list)
        nred = length(red_list)

        if first_lambda_with_two_raw === nothing && nraw == 2
            first_lambda_with_two_raw = λ
        end
        if first_lambda_with_two_reduced === nothing && nred == 2
            first_lambda_with_two_reduced = λ
        end

        @printf("accepted minima runs        = %d / %d\n", accepted_minima, n_inits)
        @printf("rejected (not minimum)      = %d\n", rejected_not_minimum)
        @printf("rejected (grad filter)      = %d\n", rejected_grad)
        @printf("unique minima raw           = %d\n", nraw)
        @printf("unique minima reduced       = %d\n", nred)

        println("top reduced minima:")
        for (j, item) in enumerate(red_list[1:min(end, 5)])
            @printf(
                "  #%d  count=%d  frac=%.4f  gn=%.3e  mineig=%.3e  Eround=%.6f\n",
                j,
                Int(item["count"]),
                Int(item["count"]) / max(accepted_minima, 1),
                Float64(item["best_grad_norm"]),
                Float64(item["best_hess_mineig"]),
                Float64(item["rounded_energy"]),
            )
        end

        # distance between first two reduced representatives, if present
        repdist12 = nothing
        if nred >= 2
            θ1 = red_list[1]["representative_theta"]
            θ2 = red_list[2]["representative_theta"]
        repdist12 = theta_distance_modpi(θ1, θ2)
    end

        push!(results_per_lambda, Dict{String, Any}(
            "lambda" => λ,
            "accepted_minima_runs" => accepted_minima,
            "rejected_not_minimum" => rejected_not_minimum,
            "rejected_grad_filter" => rejected_grad,
            "unique_minima_raw" => nraw,
            "unique_minima_reduced" => nred,
            "rep_distance_12_reduced" => repdist12,
            "raw_minima" => raw_list,
            "reduced_minima" => red_list,
        ))
    end

    # -----------------------
    # Final save
    # -----------------------
    save_data = Dict{String, Any}(
        "experiment" => "find_two_minima_onegraph",
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seed" => graph_seed,
        "graph_path" => gpath,
        "lambdas" => λs,
        "n_inits" => n_inits,
        "solver" => Dict(
            "name" => "grad",
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "save_params" => save_params,
            "compute_hessian" => compute_hessian,
            "hessian_tol" => hessian_tol,
            "require_grad_norm_max" => require_grad_norm_max,
            "grad_norm_max" => grad_norm_max,
        ),
        "angle_clustering" => Dict(
            "angle_bin" => angle_bin,
            "flip_equiv" => angle_flip_equiv,
            "wrap" => "mod_pi",
            "representative_policy" => "best grad norm within cluster",
        ),
        "first_lambda_with_two_raw" => (first_lambda_with_two_raw === nothing ? "none" : first_lambda_with_two_raw),
        "first_lambda_with_two_reduced" => (first_lambda_with_two_reduced === nothing ? "none" : first_lambda_with_two_reduced),
        "results_per_lambda" => results_per_lambda,
    )

    atomic_json_write(out_path, save_data)

    println()
    println("==============================================================")
    println("Done.")
    println("Saved: $out_path")
    println("first λ with 2 raw minima     = $(first_lambda_with_two_raw)")
    println("first λ with 2 reduced minima = $(first_lambda_with_two_reduced)")
    println("==============================================================")
end

main()