# scripts/print_panel_b_info.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const LAMBDA_TRANSITION = 0.25

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function getvec(entry, key::AbstractString)
    haskey(entry, key) || return Float64[]
    v = entry[key]
    v === nothing && return Float64[]
    return Float64.(v)
end

finite_mask(x) = .!isnan.(x) .&& .!isinf.(x)
finite_positive_mask(x) = finite_mask(x) .&& (x .> 0)

function conv_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function extract_conv_from_filename(path::AbstractString)
    b = basename(path)
    m = match(r"_conv([^_]+)_abin", b)
    m === nothing && return nothing
    try
        return conv_tag_to_float(m.captures[1])
    catch
        return nothing
    end
end

function tao_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function extract_tao_from_filename(path::AbstractString)
    b = basename(path)
    m = match(r"_tao([^_]+)_conv", b)
    m === nothing && return nothing
    try
        return tao_tag_to_float(m.captures[1])
    catch
        return nothing
    end
end

function json_tao(path::AbstractString)
    data = JSON.parsefile(path)

    if haskey(data, "tao")
        v = data["tao"]
        if !(v === nothing || v == "none")
            return Float64(v)
        end
    end

    if haskey(data, "solver")
        solver = data["solver"]
        if solver isa Dict && haskey(solver, "tao")
            v = solver["tao"]
            if !(v === nothing || v == "none")
                return Float64(v)
            end
        end
    end

    return nothing
end

function effective_tao(path::AbstractString)
    tf = extract_tao_from_filename(path)
    tf !== nothing && return tf
    return json_tao(path)
end

function json_is_new_like(path::AbstractString)
    data = JSON.parsefile(path)

    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false

    entry1 = rpl[1]
    has_cloud = haskey(entry1, "opt_min_hess_conds")
    has_summary = haskey(entry1, "opt_min_hess_cond_median") || haskey(entry1, "opt_min_hess_cond_mean")

    return haskey(entry1, "unique_angle_count_mean") &&
           haskey(entry1, "unique_angle_count_stderr") &&
           haskey(entry1, "Nstar_mean") &&
           haskey(entry1, "Nstar_stderr") &&
           haskey(entry1, "mean_ratio_mean") &&
           (has_cloud || has_summary)
end

function list_matching_jsons(; N, k, ngraphs, n_inits, angle_bin, require_hessian=true)
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
        ok = startswith(b, prefix) &&
             occursin(abin_piece, b) &&
             occursin(ninit_piece, b) &&
             !occursin("checkpoint", lowercase(b))
        if require_hessian
            ok &= occursin("_hess_on_", b)
        end
        ok
    end

    matches = filter(json_is_new_like, matches)

    isempty(matches) && error("No matching aggregated JSON found in:\n$subdir")
    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, tao, require_hessian=true)
    matches = list_matching_jsons(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        require_hessian = require_hessian,
    )

    tol = 1e-14

    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error("No matching aggregated JSON found for conv=$(angle_conv).")

    explicit_match = String[]
    legacy_unknown = String[]

    println("------------------------------------------------------------")
    println("Candidate files after conv filter:")
    for f in conv_matches
        tf = extract_tao_from_filename(f)
        tj = json_tao(f)
        te = effective_tao(f)
        println("file = ", basename(f))
        println("  tao from filename = ", tf)
        println("  tao from json     = ", tj)
        println("  effective tao     = ", te)

        if te === nothing
            push!(legacy_unknown, f)
        elseif isapprox(te, tao; atol=tol, rtol=0.0)
            push!(explicit_match, f)
        else
            println("  -> rejected (different tao)")
        end
    end
    println("------------------------------------------------------------")

    chosen_pool =
        if !isempty(explicit_match)
            explicit_match
        elseif !isempty(legacy_unknown)
            println("No file with explicit tao=$(tao) found.")
            println("Falling back only to legacy files with no tao in filename and no tao in JSON.")
            legacy_unknown
        else
            error("No matching aggregated JSON found for conv=$(angle_conv) and tao=$(tao).")
        end

    mtimes = [stat(f).mtime for f in chosen_pool]
    chosen = chosen_pool[argmax(mtimes)]

    println("Chosen file: ", chosen)
    println("Chosen effective tao: ", effective_tao(chosen))
    println("------------------------------------------------------------")

    return chosen
end

function load_panel_b_data(; N=50, k=3, ngraphs=100, n_inits=10000, angle_bin=0.01,
                             angle_conv_main=1e-8, tao_main=0.1)
    path_main = newest_matching_json(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        angle_conv = angle_conv_main,
        tao = tao_main,
        require_hessian = true,
    )

    data = JSON.parsefile(path_main)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    nmin = Float64[]
    nmin_err = Float64[]
    nstar = Float64[]
    nstar_err = Float64[]
    mean_ratio = Float64[]
    κopt_median_saved = Float64[]
    κopt_mean_saved = Float64[]
    opt_min_hess_conds = Vector{Vector{Float64}}()

    for entry in rpl
        λ    = getnum(entry, "λ")
        nm   = getnum(entry, "unique_angle_count_mean")
        nme  = getnum(entry, "unique_angle_count_stderr")
        ns   = getnum(entry, "Nstar_mean")
        nse  = getnum(entry, "Nstar_stderr")
        mr   = getnum(entry, "mean_ratio_mean")
        κmed = getnum(entry, "opt_min_hess_cond_median")
        κmn  = getnum(entry, "opt_min_hess_cond_mean")
        κs   = getvec(entry, "opt_min_hess_conds")

        if isfinite(λ)
            push!(λs, λ)
            push!(nmin, nm)
            push!(nmin_err, nme)
            push!(nstar, ns)
            push!(nstar_err, nse)
            push!(mean_ratio, mr)
            push!(κopt_median_saved, κmed)
            push!(κopt_mean_saved, κmn)
            push!(opt_min_hess_conds, κs)
        end
    end

    p = sortperm(λs)

    λs = λs[p]
    nmin = nmin[p]
    nmin_err = nmin_err[p]
    nstar = nstar[p]
    nstar_err = nstar_err[p]
    mean_ratio = mean_ratio[p]
    κopt_median_saved = κopt_median_saved[p]
    κopt_mean_saved = κopt_mean_saved[p]
    opt_min_hess_conds = opt_min_hess_conds[p]

    mask_b = isfinite.(λs) .&
             isfinite.(nmin) .&
             isfinite.(nmin_err) .&
             isfinite.(nstar) .&
             isfinite.(nstar_err) .&
             isfinite.(mean_ratio) .&
             (nmin .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    λk = Float64[]
    κk = Float64[]
    for (i, λ) in enumerate(λs)
        for κ in opt_min_hess_conds[i]
            if isfinite(κ) && κ > 0
                push!(λk, λ)
                push!(κk, κ)
            end
        end
    end

    return (
        path_main = path_main,
        λs = λs,
        nmin = nmin,
        nmin_err = nmin_err,
        nstar = nstar,
        nstar_err = nstar_err,
        mean_ratio = mean_ratio,
        κopt_median_saved = κopt_median_saved,
        κopt_mean_saved = κopt_mean_saved,
        λk = λk,
        κk = κk,
        mask_b = mask_b,
    )
end

function kappa_curve_from_cloud(λ, κ; κmax=Inf, stat=:mean)
    keep = finite_positive_mask(κ) .& (κ .<= κmax)

    λf = λ[keep]
    κf = κ[keep]

    isempty(λf) && return Float64[], Float64[], Float64[]

    λu = sort(unique(λf))
    y = Float64[]
    yerr = Float64[]

    for l in λu
        mask = λf .== l
        vals = κf[mask]

        if stat == :mean
            push!(y, mean(vals))
            push!(yerr, length(vals) > 1 ? std(vals) / sqrt(length(vals)) : 0.0)
        elseif stat == :median
            push!(y, median(vals))
            push!(yerr, 0.0)
        else
            error("stat must be :mean or :median")
        end
    end

    return λu, y, yerr
end

function print_panel_b_info(data)
    println()
    println("============================================================")
    println("PANEL (b): Number of minima, optimal minima, ratio, and κ inset")
    println("============================================================")
    println("JSON file used:")
    println(data.path_main)

    println()
    println("MAIN PANEL VALUES")
    println("--------------------------------------------------------------------------------------------------------------")
    @printf("%8s  %14s  %14s  %14s  %14s  %14s\n",
        "lambda", "Nmin", "stderr(Nmin)", "N*", "stderr(N*)", "mean_ratio")
    for i in eachindex(data.λs)
        if data.mask_b[i]
            @printf("%8.4f  %14.6e  %14.6e  %14.6e  %14.6e  %14.8f\n",
                data.λs[i],
                data.nmin[i],
                data.nmin_err[i],
                data.nstar[i],
                data.nstar_err[i],
                data.mean_ratio[i]
            )
        end
    end

    println()
    println("AROUND THE TRANSITION")
    println("--------------------------------------------------------------------------------------------------------------")
    idx_before = findlast(<(LAMBDA_TRANSITION), data.λs)
    idx_at     = findfirst(x -> isapprox(x, LAMBDA_TRANSITION; atol=1e-12), data.λs)
    idx_after  = findfirst(>(LAMBDA_TRANSITION), data.λs)

    if idx_before !== nothing
        @printf("largest lambda below %.4f : λ = %.4f, Nmin = %.6e, N* = %.6e, ratio = %.8f\n",
            LAMBDA_TRANSITION,
            data.λs[idx_before], data.nmin[idx_before], data.nstar[idx_before], data.mean_ratio[idx_before]
        )
    end
    if idx_at !== nothing
        @printf("at lambda %.4f            : λ = %.4f, Nmin = %.6e, N* = %.6e, ratio = %.8f\n",
            LAMBDA_TRANSITION,
            data.λs[idx_at], data.nmin[idx_at], data.nstar[idx_at], data.mean_ratio[idx_at]
        )
    end
    if idx_after !== nothing
        @printf("smallest lambda above %.4f: λ = %.4f, Nmin = %.6e, N* = %.6e, ratio = %.8f\n",
            LAMBDA_TRANSITION,
            data.λs[idx_after], data.nmin[idx_after], data.nstar[idx_after], data.mean_ratio[idx_after]
        )
    end

    println()
    println("GLOBAL SUMMARIES FOR MAIN PANEL")
    println("--------------------------------------------------------------------------------------------------------------")
    @printf("minimum Nmin = %.6e at lambda = %.4f\n",
        minimum(data.nmin[data.mask_b]), data.λs[findmin(data.nmin[data.mask_b])[2]]
    )
    @printf("maximum Nmin = %.6e\n", maximum(data.nmin[data.mask_b]))
    @printf("maximum N*   = %.6e\n", maximum(data.nstar[data.mask_b]))
    @printf("ratio range  = [%.8f, %.8f]\n",
        minimum(data.mean_ratio[data.mask_b]), maximum(data.mean_ratio[data.mask_b]))

    println()
    println("κ INSET FROM CLOUD (same source used by the plot)")
    println("--------------------------------------------------------------------------------------------------------------")
    λm, κm, _ = kappa_curve_from_cloud(data.λk, data.κk; stat=:median)

    if isempty(λm)
        println("No κ cloud found in JSON. Falling back to saved per-lambda summaries.")
        @printf("%8s  %18s  %18s\n", "lambda", "kappa_median_saved", "kappa_mean_saved")
        for i in eachindex(data.λs)
            @printf("%8.4f  %18.10e  %18.10e\n",
                data.λs[i],
                data.κopt_median_saved[i],
                data.κopt_mean_saved[i]
            )
        end
    else
        @printf("%8s  %18s\n", "lambda", "kappa_median_from_cloud")
        for i in eachindex(λm)
            @printf("%8.4f  %18.10e\n", λm[i], κm[i])
        end
    end

    println("============================================================")
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
data = load_panel_b_data(; tao_main = 0.1)
print_panel_b_info(data)