# scripts/print_small_lambda_rawdelta_vs_ratio.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function json_is_v7_like(path::AbstractString)
    data = JSON.parsefile(path)
    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false
    e = rpl[1]
    return haskey(e, "devTheta_abs_mean") && haskey(e, "devZ_abs_mean")
end

function conv_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function extract_conv_from_filename(path::AbstractString)
    m = match(r"_conv([^_]+)_abin", basename(path))
    m === nothing && return nothing
    try
        return conv_tag_to_float(m.captures[1])
    catch
        return nothing
    end
end

function list_matching_jsons(; N, k, ngraphs, n_inits, angle_bin)
    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    subdir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(ngraphs)_unweighted"
    )
    isdir(subdir) || error("Directory not found: $subdir")

    files = readdir(subdir; join=true)

    prefix = "qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d"
    abin_piece = "_abin$(abin_tag)_"
    ninit_piece = "_ngraphs$(ngraphs)_ninit$(n_inits)_outer1_inner5000_thr0p999.json"

    matches = filter(files) do f
        b = basename(f)
        startswith(b, prefix) &&
        occursin(abin_piece, b) &&
        occursin(ninit_piece, b) &&
        !occursin("checkpoint", lowercase(b))
    end

    matches = filter(json_is_v7_like, matches)
    isempty(matches) && error("No matching v7-like JSON files found.")
    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv)
    matches = list_matching_jsons(
        N=N, k=k, ngraphs=ngraphs, n_inits=n_inits, angle_bin=angle_bin
    )

    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=1e-14, rtol=0.0)
    end

    isempty(conv_matches) && error("No JSON with conv=$(angle_conv) found.")
    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function load_aggregated_table(path::AbstractString; λmax=0.25)
    data = JSON.parsefile(path)
    rows = Tuple{Float64,Float64,Float64,Float64}[]
    for entry in data["results_per_lambda"]
        λ = getnum(entry, "λ")
        mr = getnum(entry, "mean_ratio_mean")
        ps = getnum(entry, "success_rate_mean")
        dθ = getnum(entry, "devTheta_abs_mean")
        if isfinite(λ) && λ <= λmax
            push!(rows, (λ, mr, ps, dθ))
        end
    end
    sort!(rows, by = x -> x[1])
    return rows
end

function raw_mean_abs_delta_curve(
    W,
    N;
    λs,
    n_inits,
    seed_base,
    iterations,
    inner_iterations,
    tao,
    angle_conv,
    init_mode,
    compute_hessian,
    hessian_tol,
)
    out = Dict{Float64,Float64}()

    for λ in λs
        sum_abs_delta = 0.0
        n_kept = 0

        for r in 1:n_inits
            run_seed = seed_base + r * 10_000 + Int(round(Float64(λ) * 1000))

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
                save_params = true,
                progressbar = false,
                compute_hessian = compute_hessian,
                hessian_tol = hessian_tol,
            )

            if compute_hessian
                get(sol.metadata, :hess_is_minimum, false) || continue
            end

            θ = get(sol.metadata, :theta_converged, Float64[])
            isempty(θ) && error("Missing :theta_converged")

            @inbounds for i in eachindex(θ)
                sum_abs_delta += abs(θ[i] - (pi / 4))
            end
            n_kept += 1
        end

        out[Float64(λ)] = n_kept == 0 ? NaN : sum_abs_delta / (N * n_kept)
    end

    return out
end

function main()
    N = 50
    k = 3
    ngraphs = 1
    n_inits_json = 10000
    angle_bin = 0.01
    angle_conv = 1e-8
    λmax = 0.25

    # fresh recomputation just for the raw tiny drift
    n_inits_raw = 2000
    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    init_mode = :uniform
    compute_hessian = true
    hessian_tol = 1e-8
    graph_seed = 1

    path = newest_matching_json(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits_json,
        angle_bin=angle_bin,
        angle_conv=angle_conv,
    )

    println("Using aggregated file:")
    println(path)
    println()

    agg_rows = load_aggregated_table(path; λmax=λmax)
    λs = [row[1] for row in agg_rows]

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=false, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    rawδ = raw_mean_abs_delta_curve(
        W, N;
        λs=λs,
        n_inits=n_inits_raw,
        seed_base=graph_seed * 1_000_000,
        iterations=iterations,
        inner_iterations=inner_iterations,
        tao=tao,
        angle_conv=angle_conv,
        init_mode=init_mode,
        compute_hessian=compute_hessian,
        hessian_tol=hessian_tol,
    )

    println("λ        mean_ratio    P(success)   devTheta(JSON)      raw_mean|θ-π/4|")
    println("----------------------------------------------------------------------------")
    for (λ, mr, ps, dθjson) in agg_rows
        dθraw = get(rawδ, λ, NaN)
        @printf("%0.6f   %0.6f      %0.6f      %0.10f    %.6e\n",
            λ, mr, ps, dθjson, dθraw)
    end
end

main()