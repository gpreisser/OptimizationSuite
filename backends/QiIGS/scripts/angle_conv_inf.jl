# scripts/print_success_rate_compare.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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

function list_matching_jsons(; N, k, ngraphs, n_inits, angle_bin)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")

    subdir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(ngraphs)_unweighted"
    )

    isdir(subdir) || error("Directory not found: $subdir")

    files = readdir(subdir; join=true)

    prefix = "qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d"

    matches = filter(files) do f
        b = basename(f)

        startswith(b, prefix) &&
        occursin("_abin$(abin_tag)_", b) &&
        occursin("_ngraphs$(ngraphs)_ninit$(n_inits)_outer1_inner5000_thr0p999.json", b) &&
        occursin("_hess_on_", b) &&
        !occursin("checkpoint", lowercase(b))
    end

    isempty(matches) && error("No JSON files found")

    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv)

    matches = list_matching_jsons(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
    )

    tol = 1e-14

    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error("No file found for conv=$angle_conv")

    mtimes = [stat(f).mtime for f in conv_matches]

    return conv_matches[argmax(mtimes)]
end

function load_success_curve(json_path)

    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    succ = Float64[]

    for entry in rpl

        λ = getnum(entry, "λ")
        sr = getnum(entry, "success_rate_mean")

        if isfinite(λ) && isfinite(sr)
            push!(λs, λ)
            push!(succ, sr)
        end

    end

    p = sortperm(λs)

    return λs[p], succ[p]
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()

    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.02

    conv1 = 1e-2
    conv2 = 1e-8

    path1 = newest_matching_json(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=conv1,
    )

    path2 = newest_matching_json(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=conv2,
    )

    println("\nUsing:")
    println(path1)
    println(path2)
    println()

    λ1, s1 = load_success_curve(path1)
    λ2, s2 = load_success_curve(path2)

    @printf("λ        success(conv=1e-2)   success(conv=1e-8)\n")
    println("------------------------------------------------------")

    for i in eachindex(λ1)

        @printf(
            "%6.3f     %10.6f           %10.6f\n",
            λ1[i],
            s1[i],
            s2[i],
        )

    end

end

main()