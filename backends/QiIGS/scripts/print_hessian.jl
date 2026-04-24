# scripts/print_hessian_panel_values.jl

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
        return ok
    end

    isempty(matches) && error("No matching aggregated JSON found in:\n$subdir")
    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, require_hessian=true)
    matches = list_matching_jsons(;
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

    isempty(conv_matches) && error(
        "No matching aggregated JSON found for conv=$(angle_conv).\n" *
        "Available conv values: " *
        join(sort(unique(filter(!isnothing, extract_conv_from_filename.(matches)))), ", ")
    )

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function main()
    # ------------------------------------------------------------
    # Parameters: adjust here if needed
    # ------------------------------------------------------------
    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.02
    angle_conv = 1e-2
    require_hessian = true

    path = newest_matching_json(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv,
        require_hessian=require_hessian,
    )

    println()
    println("Using JSON:")
    println(path)
    println()

    data = JSON.parsefile(path)
    rpl = data["results_per_lambda"]

    println("λ        Nangle_raw    λmin(H)        cond(H)        frac_min   frac_saddle   frac_max   frac_deg")
    println(repeat("-", 100))

    for e in rpl
        λ      = getnum(e, "λ")
        נארaw  = getnum(e, "unique_angle_count_raw_mean")
        λminH  = getnum(e, "hess_mineig_mean_mean")
        condH  = getnum(e, "hess_cond_mean_mean")
        fmin   = getnum(e, "hess_frac_min_mean")
        fsad   = getnum(e, "hess_frac_saddle_mean")
        fmax   = getnum(e, "hess_frac_max_mean")
        fdeg   = getnum(e, "hess_frac_degenerate_mean")

        @printf("%6.3f   %10.3f   %12.6e   %12.6e   %8.4f   %11.4f   %8.4f   %8.4f\n",
            λ, נארaw, λminH, condH, fmin, fsad, fmax, fdeg)
    end

    println()
end

main()