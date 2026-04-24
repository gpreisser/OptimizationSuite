# scripts/print_grad_and_hess.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function conv_tag_to_float(tag::AbstractString)
    s = replace(tag, "p" => ".")
    return parse(Float64, s)
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

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, require_hessian=true)
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
        return ok
    end

    isempty(matches) && error("No matching aggregated JSON found in $subdir")

    tol = 1e-14
    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error(
        "No matching JSON found for conv=$(angle_conv).\nAvailable conv values: " *
        join(sort(unique(filter(!isnothing, extract_conv_from_filename.(matches)))), ", ")
    )

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function load_grad_and_hess(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    haskey(data, "results_per_lambda") || error("Missing key results_per_lambda in $json_path")
    rpl = data["results_per_lambda"]

    λs = Float64[]
    gn_meaninner = Float64[]
    gn_meaninner_err = Float64[]
    hess_mineig = Float64[]
    hess_mineig_err = Float64[]

    for entry in rpl
        λ    = getnum(entry, "λ")
        gnm  = getnum(entry, "grad_norm_meaninner_mean")
        gne  = getnum(entry, "grad_norm_meaninner_stderr")
        hmin = getnum(entry, "hess_mineig_mean_mean")
        hime = getnum(entry, "hess_mineig_mean_stderr")

        if isfinite(λ)
            push!(λs, λ)
            push!(gn_meaninner, gnm)
            push!(gn_meaninner_err, gne)
            push!(hess_mineig, hmin)
            push!(hess_mineig_err, hime)
        end
    end

    p = sortperm(λs)

    return (
        λs[p],
        gn_meaninner[p],
        gn_meaninner_err[p],
        hess_mineig[p],
        hess_mineig_err[p],
    )
end

function main()
    N = 50
    k = 3
    ngraphs = 100
    n_inits = 10000
    angle_bin = 0.01
    angle_conv = 1e-8

    json_path = newest_matching_json(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv,
        require_hessian=true,
    )

    println("Using JSON:")
    println(json_path)

    λs, g, ge, h, he = load_grad_and_hess(json_path)

    println()
    println("λ        grad_meaninner      grad_stderr        hess_mineig_mean     hess_mineig_stderr")
    println("------------------------------------------------------------------------------------------")

    for i in eachindex(λs)
        @printf("%0.4f   %14.6e   %14.6e   %14.6e   %14.6e\n",
            λs[i], g[i], ge[i], h[i], he[i]
        )
    end

    println()
end

main()