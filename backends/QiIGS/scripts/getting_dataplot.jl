# scripts/print_small_lambda_diagnostics.jl

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

function main()
    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.01
    angle_conv = 1e-8
    λmax_print = 0.25

    path = newest_matching_json(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv,
    )

    println("Using file:")
    println(path)
    println()

    data = JSON.parsefile(path)
    rpl = data["results_per_lambda"]

    println("λ        P(success)   mean_ratio    unique_spin   unique_angle_raw   devTheta")
    println("-------------------------------------------------------------------------------")

    for entry in rpl
        λ   = getnum(entry, "λ")
        ps  = getnum(entry, "success_rate_mean")
        mr  = getnum(entry, "mean_ratio_mean")
        uc  = getnum(entry, "unique_count_mean")
        uar = getnum(entry, "unique_angle_count_raw_mean")
        dθ  = getnum(entry, "devTheta_abs_mean")

        if isfinite(λ) && λ <= λmax_print
            @printf("%0.6f   %0.6f     %0.6f      %8.1f      %8.1f         %0.10f\n",
                λ, ps, mr, uc, uar, dθ)
        end
    end
end

main()