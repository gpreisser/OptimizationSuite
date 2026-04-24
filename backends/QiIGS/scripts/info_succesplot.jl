# scripts/print_hessian_angles_ab_info.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

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

function json_is_v9_like(path::AbstractString)
    data = JSON.parsefile(path)

    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false

    entry1 = rpl[1]

    haskey(entry1, "devTheta_abs_mean") || return false
    haskey(entry1, "devTheta_abs_stderr") || return false
    haskey(entry1, "unique_angle_count_raw_mean") || return false
    haskey(entry1, "unique_angle_count_raw_stderr") || return false
    haskey(entry1, "mean_ratio_mean") || return false
    haskey(entry1, "success_rate_mean") || return false
    haskey(entry1, "success_rate_stderr") || return false
    haskey(entry1, "grad_norm_init_mean") || return false
    haskey(entry1, "grad_norm_init_stderr") || return false
    haskey(entry1, "init_hess_mineig_mean_mean") || return false
    haskey(entry1, "init_hess_mineig_mean_stderr") || return false
    haskey(entry1, "hess_mineig_mean_mean") || return false
    haskey(entry1, "hess_mineig_mean_stderr") || return false

    haskey(data, "solver") || return false
    solver = data["solver"]
    haskey(solver, "outputs") || return false
    outs = solver["outputs"]

    return ("devTheta_abs" in outs) &&
           ("gn_init" in outs) &&
           ("init_hess_mineig" in outs) &&
           ("hess_mineig" in outs)
end

function load_curves(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    unique_angle_raw = Float64[]
    unique_angle_raw_err = Float64[]
    init_hess_mineig = Float64[]
    init_hess_mineig_err = Float64[]
    final_hess_mineig = Float64[]
    final_hess_mineig_err = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]
    succ_rate_err = Float64[]
    devTheta_abs = Float64[]
    devTheta_abs_err = Float64[]
    gn_init = Float64[]
    gn_init_err = Float64[]

    for entry in rpl
        λ      = getnum(entry, "λ")
        ua     = getnum(entry, "unique_angle_count_raw_mean")
        uae    = getnum(entry, "unique_angle_count_raw_stderr")
        ihm    = getnum(entry, "init_hess_mineig_mean_mean")
        ihme   = getnum(entry, "init_hess_mineig_mean_stderr")
        fhm    = getnum(entry, "hess_mineig_mean_mean")
        fhme   = getnum(entry, "hess_mineig_mean_stderr")
        mr     = getnum(entry, "mean_ratio_mean")
        sr     = getnum(entry, "success_rate_mean")
        sre    = getnum(entry, "success_rate_stderr")
        dθ     = getnum(entry, "devTheta_abs_mean")
        dθe    = getnum(entry, "devTheta_abs_stderr")
        gni    = getnum(entry, "grad_norm_init_mean")
        gnie   = getnum(entry, "grad_norm_init_stderr")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_raw, ua)
        push!(unique_angle_raw_err, uae)
        push!(init_hess_mineig, ihm)
        push!(init_hess_mineig_err, ihme)
        push!(final_hess_mineig, fhm)
        push!(final_hess_mineig_err, fhme)
        push!(mean_ratio, mr)
        push!(succ_rate, sr)
        push!(succ_rate_err, sre)
        push!(devTheta_abs, dθ)
        push!(devTheta_abs_err, dθe)
        push!(gn_init, gni)
        push!(gn_init_err, gnie)
    end

    p = sortperm(λs)

    return (
        λs[p],
        unique_angle_raw[p],
        unique_angle_raw_err[p],
        init_hess_mineig[p],
        init_hess_mineig_err[p],
        final_hess_mineig[p],
        final_hess_mineig_err[p],
        mean_ratio[p],
        succ_rate[p],
        succ_rate_err[p],
        devTheta_abs[p],
        devTheta_abs_err[p],
        gn_init[p],
        gn_init_err[p],
    )
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

    matches = filter(json_is_v9_like, matches)

    isempty(matches) && error(
        "No matching v9-like aggregated JSON found in:\n$subdir\n" *
        "Check angle_bin and whether the updated sweep was rerun."
    )

    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, require_hessian=true)
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

    isempty(conv_matches) && error(
        "No matching v9-like aggregated JSON found for conv=$(angle_conv).\n" *
        "Available conv values: " *
        join(sort(unique(filter(!isnothing, extract_conv_from_filename.(matches)))), ", ")
    )

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function print_separator(title::AbstractString)
    println()
    println("="^90)
    println(title)
    println("="^90)
end

function main()
    N = 50
    k = 3
    ngraphs = 100
    n_inits = 10000
    angle_bin = 0.01
    angle_conv_main = 1e-8

    path_main = newest_matching_json(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        angle_conv = angle_conv_main,
        require_hessian = true,
    )

    @info "Using main path" path_main

    λs, uniques_angle, uniques_angle_err,
    init_hess_mineig, init_hess_mineig_err,
    final_hess_mineig, final_hess_mineig_err,
    mean_ratio,
    succ_rate, succ_rate_err,
    devTheta_abs, devTheta_abs_err,
    gn_init, gn_init_err = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_all = isfinite.(λs)

    print_separator("FULL TABLE")
    @printf("%8s  %16s  %16s  %14s  %14s  %12s  %12s  %14s  %14s  %14s  %14s  %14s  %14s\n",
        "lambda",
        "(4/pi)|θ-π/4|",
        "stderr",
        "Nmin",
        "stderr",
        "r",
        "Psolved",
        "Psolved_err",
        "gn_init",
        "gn_init_err",
        "init_hess",
        "init_hess_err",
        "final_hess"
    )

    for i in eachindex(λs)
        mask_all[i] || continue
        @printf("%8.4f  %16.8e  %16.8e  %14.6e  %14.6e  %12.6f  %12.6f  %14.6e  %14.6e  %14.6e  %14.6e  %14.6e  %14.6e\n",
            λs[i],
            devTheta_abs[i] * (4 / pi),
            devTheta_abs_err[i] * (4 / pi),
            uniques_angle[i],
            uniques_angle_err[i],
            mean_ratio[i],
            succ_rate[i],
            succ_rate_err[i],
            gn_init[i],
            gn_init_err[i],
            init_hess_mineig[i],
            init_hess_mineig_err[i],
            final_hess_mineig[i]
        )
    end

    print_separator("PANEL (a): MAIN CURVE")
    @printf("%8s  %20s  %20s\n", "lambda", "(4/pi)|θ-π/4|", "stderr")
    for i in eachindex(λs)
        if isfinite(λs[i]) && isfinite(devTheta_abs[i]) && isfinite(devTheta_abs_err[i]) &&
           0.0 <= λs[i] <= 1.0
            @printf("%8.4f  %20.12e  %20.12e\n",
                λs[i],
                devTheta_abs[i] * (4 / pi),
                devTheta_abs_err[i] * (4 / pi)
            )
        end
    end

    print_separator("PANEL (a) INSET REGION: 0 <= λ <= 0.25 AND VALUE > 0")
    @printf("%8s  %20s  %20s\n", "lambda", "(4/pi)|θ-π/4|", "stderr")
    for i in eachindex(λs)
        if isfinite(λs[i]) && isfinite(devTheta_abs[i]) && isfinite(devTheta_abs_err[i]) &&
           devTheta_abs[i] > 0.0 && 0.0 <= λs[i] <= 0.25
            @printf("%8.4f  %20.12e  %20.12e\n",
                λs[i],
                devTheta_abs[i] * (4 / pi),
                devTheta_abs_err[i] * (4 / pi)
            )
        end
    end

    print_separator("PANEL (b): N_minima + color value r")
    @printf("%8s  %16s  %16s  %14s  %14s  %14s\n",
        "lambda", "Nmin", "stderr", "r", "Psolved", "Psolved_err")
    for i in eachindex(λs)
        if isfinite(λs[i]) && isfinite(uniques_angle[i]) && isfinite(uniques_angle_err[i]) &&
           isfinite(mean_ratio[i]) && uniques_angle[i] > 0.0 && 0.0 <= λs[i] <= 1.0
            @printf("%8.4f  %16.8e  %16.8e  %14.6f  %14.6f  %14.6e\n",
                λs[i],
                uniques_angle[i],
                uniques_angle_err[i],
                mean_ratio[i],
                succ_rate[i],
                succ_rate_err[i]
            )
        end
    end

    finite_r = findall(i -> isfinite(mean_ratio[i]), eachindex(mean_ratio))
    if !isempty(finite_r)
        ibest = finite_r[argmax(mean_ratio[finite_r])]

        print_separator("BEST-r SUMMARY")
        @printf("lambda                    = %.6f\n", λs[ibest])
        @printf("r                         = %.10f\n", mean_ratio[ibest])
        @printf("P_solved                  = %.10f ± %.3e\n", succ_rate[ibest], succ_rate_err[ibest])
        @printf("(4/pi)|θ-π/4|             = %.12e ± %.3e\n",
            devTheta_abs[ibest] * (4 / pi),
            devTheta_abs_err[ibest] * (4 / pi)
        )
        @printf("N_minima                  = %.10e ± %.3e\n", uniques_angle[ibest], uniques_angle_err[ibest])
        @printf("grad_norm_init_mean       = %.10e ± %.3e\n", gn_init[ibest], gn_init_err[ibest])
        @printf("init_hess_mineig_mean     = %.10e ± %.3e\n", init_hess_mineig[ibest], init_hess_mineig_err[ibest])
        @printf("final_hess_mineig_mean    = %.10e ± %.3e\n", final_hess_mineig[ibest], final_hess_mineig_err[ibest])
    end
end

main()