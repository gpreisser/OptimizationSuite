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

    isempty(matches) && error("No matching aggregated JSON found.")

    matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, require_hessian=true)

    matches = list_matching_jsons(
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        require_hessian=require_hessian,
    )

    tol = 1e-14

    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error("No JSON found for conv=$(angle_conv)")

    mtimes = [stat(f).mtime for f in conv_matches]

    conv_matches[argmax(mtimes)]
end

function print_success(path)

    data = JSON.parsefile(path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    ps = Float64[]

    for entry in rpl

        λ = getnum(entry, "λ")
        s = getnum(entry, "success_rate_mean")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(ps, s)

    end

    p = sortperm(λs)

    println("------------------------------------------------")
    println(@sprintf("%8s %16s", "lambda", "P_success"))
    println("------------------------------------------------")

    for i in p
        println(@sprintf("%8.3f %16.6f", λs[i], ps[i]))
    end

end

function main()

    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.02

    convs = [1e-8, 1e-2]

    for conv in convs

        println()
        println("==============================================")
        println("Success rate for conv = ", conv)
        println("==============================================")

        path = newest_matching_json(
            N=N,
            k=k,
            ngraphs=ngraphs,
            n_inits=n_inits,
            angle_bin=angle_bin,
            angle_conv=conv,
            require_hessian=true,
        )

        println("Using JSON: ", path)
        println()

        print_success(path)

    end

end

main()