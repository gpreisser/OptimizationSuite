# scripts/print_ratio_compare.jl

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

    isempty(matches) && error("No matching aggregated JSON files found.")
    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv)
    matches = list_matching_jsons(;
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

    isempty(conv_matches) && error(
        "No matching JSON found for conv=$(angle_conv).\nAvailable conv values: " *
        join(sort(unique(filter(!isnothing, extract_conv_from_filename.(matches)))), ", ")
    )

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function load_ratio_curve(json_path)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    best_ratio = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]

    for entry in rpl
        λ  = getnum(entry, "λ")
        br = getnum(entry, "best_ratio_mean")
        mr = getnum(entry, "mean_ratio_mean")
        sr = getnum(entry, "success_rate_mean")

        if isfinite(λ)
            push!(λs, λ)
            push!(best_ratio, br)
            push!(mean_ratio, mr)
            push!(succ_rate, sr)
        end
    end

    p = sortperm(λs)
    return λs[p], best_ratio[p], mean_ratio[p], succ_rate[p]
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

    path1 = newest_matching_json(;
        N=N, k=k, ngraphs=ngraphs, n_inits=n_inits,
        angle_bin=angle_bin, angle_conv=conv1,
    )

    path2 = newest_matching_json(;
        N=N, k=k, ngraphs=ngraphs, n_inits=n_inits,
        angle_bin=angle_bin, angle_conv=conv2,
    )

    println("\nUsing:")
    println(path1)
    println(path2)
    println()

    λ1, br1, mr1, sr1 = load_ratio_curve(path1)
    λ2, br2, mr2, sr2 = load_ratio_curve(path2)

    length(λ1) == length(λ2) || error("Mismatch in λ-grid lengths.")
    all(isapprox.(λ1, λ2; atol=0.0, rtol=0.0)) || error("Mismatch in λ values between files.")

    println("λ        best(1e-2)   best(1e-8)   mean(1e-2)   mean(1e-8)   succ(1e-2)   succ(1e-8)")
    println(repeat("-", 98))

    for i in eachindex(λ1)
        @printf("%6.3f   %10.6f   %10.6f   %10.6f   %10.6f   %10.6f   %10.6f\n",
            λ1[i], br1[i], br2[i], mr1[i], mr2[i], sr1[i], sr2[i])
    end

    println()
    println("Focused window around the suspected anomaly:")
    println("λ        best(1e-2)   best(1e-8)   mean(1e-2)   mean(1e-8)   succ(1e-2)   succ(1e-8)")
    println(repeat("-", 98))

    for i in eachindex(λ1)
        if 0.225 <= λ1[i] <= 0.325
            @printf("%6.3f   %10.6f   %10.6f   %10.6f   %10.6f   %10.6f   %10.6f\n",
                λ1[i], br1[i], br2[i], mr1[i], mr2[i], sr1[i], sr2[i])
        end
    end

    println()
end

main()