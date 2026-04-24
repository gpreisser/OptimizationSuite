# scripts/plot_hessian_angles_final.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using CSV
using DataFrames
using LaTeXStrings
using Printf
using Statistics

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(ROOT, "results", "plots")
mkpath(PLOTS_DIR)

const MAIN_TICKSIZE   = 16
const MAIN_LABELSIZE  = 16
const INSET_TICKSIZE  = 16
const INSET_LABELSIZE = 16
const PANEL_LABELSIZE = 18

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

function data_dir_for_seed(data_base_dir::AbstractString, n::Int, seed::Int)
    return joinpath(data_base_dir, "data_n$(n)_seed$(seed)")
end

function load_external_optimal_cond_cloud(; n=50, seed_start=1, seed_end=100, data_base_dir="")
    raw_cond_lambdas = Float64[]
    raw_cond_values = Float64[]

    println("============================================================")
    println(" Loading external optimal-condition cloud for panel (d) inset")
    println("============================================================")
    println("data_base_dir = ", data_base_dir)
    println("seed range    = ", seed_start, ":", seed_end)

    n_found_files = 0
    n_used_files = 0

    for s in seed_start:seed_end
        ddir = data_dir_for_seed(data_base_dir, n, s)
        detail_file = joinpath(ddir, "minima_detail.csv")

        if !isfile(detail_file)
            println("seed $(s): missing -> ", detail_file)
            continue
        end

        n_found_files += 1
        println("seed $(s): found -> ", detail_file)

        df = CSV.read(detail_file, DataFrame)
        colnames = names(df)

        if !all(["lambda", "condition_number", "rounds_to_global"] .∈ Ref(colnames))
            println("  skipped: missing expected columns")
            println("  columns present: ", colnames)
            continue
        end

        λv = Float64.(df[!, "lambda"])
        κv = Float64.(df[!, "condition_number"])
        rg = Int.(df[!, "rounds_to_global"])

        mask = (rg .== 1) .& finite_positive_mask(κv)
        nkeep = count(mask)

        println("  rows total   = ", nrow(df))
        println("  rows kept    = ", nkeep)

        if nkeep > 0
            n_used_files += 1
            append!(raw_cond_lambdas, λv[mask])
            append!(raw_cond_values, κv[mask])
        end
    end

    println("------------------------------------------------------------")
    println("external files found    = ", n_found_files)
    println("external files used     = ", n_used_files)
    println("external cloud npoints  = ", length(raw_cond_values))

    if !isempty(raw_cond_values)
        println("external λ range        = [", minimum(raw_cond_lambdas), ", ", maximum(raw_cond_lambdas), "]")
        println("external κ range        = [", minimum(raw_cond_values), ", ", maximum(raw_cond_values), "]")
    else
        println("WARNING: external cloud is empty")
    end

    max_condition_number = 100.0

    if !isempty(raw_cond_values)
        keep = finite_mask(raw_cond_values) .& (raw_cond_values .<= max_condition_number)
        dropped = length(raw_cond_values) - count(keep)

        raw_cond_lambdas = raw_cond_lambdas[keep]
        raw_cond_values = raw_cond_values[keep]

        println("Applied max_condition_number = ", max_condition_number)
        println("Dropped external κ points    = ", dropped)
        println("Remaining external points    = ", length(raw_cond_values))

        if !isempty(raw_cond_values)
            println("filtered external κ range   = [", minimum(raw_cond_values), ", ", maximum(raw_cond_values), "]")
        else
            println("WARNING: all external κ points were removed by the filter")
        end
    end

    println("============================================================")

    return raw_cond_lambdas, raw_cond_values
end

function json_is_new_like(path::AbstractString)
    data = JSON.parsefile(path)

    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false

    entry1 = rpl[1]

    return haskey(entry1, "devTheta_abs_mean") &&
           haskey(entry1, "devTheta_abs_stderr") &&
           haskey(entry1, "unique_angle_count_mean") &&
           haskey(entry1, "unique_angle_count_stderr") &&
           haskey(entry1, "Nstar_mean") &&
           haskey(entry1, "Nstar_stderr") &&
           haskey(entry1, "mean_ratio_mean") &&
           haskey(entry1, "opt_min_hess_conds") &&
           haskey(entry1, "hess_cond_mean_mean") &&
           haskey(entry1, "hess_cond_mean_stderr") &&
           haskey(entry1, "opt_min_probs_raw") &&
           haskey(entry1, "success_rate_mean") &&
           haskey(entry1, "success_rate_stderr") &&
           haskey(entry1, "grad_norm_init_mean") &&
           haskey(entry1, "grad_norm_init_stderr")
end

function load_panel_abc_curves(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]

    devTheta_abs = Float64[]
    devTheta_abs_err = Float64[]

    nmin = Float64[]
    nmin_err = Float64[]

    nstar = Float64[]
    nstar_err = Float64[]

    mean_ratio = Float64[]

    κmean_all = Float64[]
    κmean_all_err = Float64[]

    succ_rate = Float64[]
    succ_rate_err = Float64[]

    gn_init = Float64[]
    gn_init_err = Float64[]

    opt_min_hess_conds = Vector{Vector{Float64}}()
    opt_min_probs_raw = Vector{Vector{Float64}}()

    for entry in rpl
        λ   = getnum(entry, "λ")
        dθ  = getnum(entry, "devTheta_abs_mean")
        dθe = getnum(entry, "devTheta_abs_stderr")

        nm  = getnum(entry, "unique_angle_count_mean")
        nme = getnum(entry, "unique_angle_count_stderr")

        ns  = getnum(entry, "Nstar_mean")
        nse = getnum(entry, "Nstar_stderr")

        mr  = getnum(entry, "mean_ratio_mean")

        κm  = getnum(entry, "hess_cond_mean_mean")
        κme = getnum(entry, "hess_cond_mean_stderr")

        sr  = getnum(entry, "success_rate_mean")
        sre = getnum(entry, "success_rate_stderr")

        gni  = getnum(entry, "grad_norm_init_mean")
        gnie = getnum(entry, "grad_norm_init_stderr")

        κs = getvec(entry, "opt_min_hess_conds")
        ps = getvec(entry, "opt_min_probs_raw")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(devTheta_abs, dθ)
        push!(devTheta_abs_err, dθe)
        push!(nmin, nm)
        push!(nmin_err, nme)
        push!(nstar, ns)
        push!(nstar_err, nse)
        push!(mean_ratio, mr)
        push!(κmean_all, κm)
        push!(κmean_all_err, κme)
        push!(succ_rate, sr)
        push!(succ_rate_err, sre)
        push!(gn_init, gni)
        push!(gn_init_err, gnie)
        push!(opt_min_hess_conds, κs)
        push!(opt_min_probs_raw, ps)
    end

    p = sortperm(λs)

    return (
        λs = λs[p],
        devTheta_abs = devTheta_abs[p],
        devTheta_abs_err = devTheta_abs_err[p],
        nmin = nmin[p],
        nmin_err = nmin_err[p],
        nstar = nstar[p],
        nstar_err = nstar_err[p],
        mean_ratio = mean_ratio[p],
        κmean_all = κmean_all[p],
        κmean_all_err = κmean_all_err[p],
        succ_rate = succ_rate[p],
        succ_rate_err = succ_rate_err[p],
        gn_init = gn_init[p],
        gn_init_err = gn_init_err[p],
        opt_min_hess_conds = opt_min_hess_conds[p],
        opt_min_probs_raw = opt_min_probs_raw[p],
    )
end

function styled_axis(figpos; xlabel, ylabel, xticks=nothing, kwargs...)
    common = (;
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = MAIN_TICKSIZE,
        yticklabelsize = MAIN_TICKSIZE,
        xlabelsize = MAIN_LABELSIZE,
        ylabelsize = MAIN_LABELSIZE,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    if xticks === nothing
        return Axis(figpos; common..., kwargs...)
    else
        return Axis(figpos; xticks = xticks, common..., kwargs...)
    end
end

function conv_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function tao_tag_to_float(tag::AbstractString)
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

function list_matching_jsons(; N, k, ngraphs, n_inits, angle_bin, tao, require_hessian=true)
    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    tao_tag  = replace(@sprintf("%.3g", tao), "." => "p")

    subdir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(ngraphs)_unweighted"
    )
    isdir(subdir) || error("Directory not found: $subdir")

    files = readdir(subdir; join=true)

    prefix = "qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d"
    tao_piece = "_tao$(tao_tag)_"
    abin_piece = "_abin$(abin_tag)_"
    ninit_piece = "_ngraphs$(ngraphs)_ninit$(n_inits)_outer1_inner5000_thr0p999.json"

    matches = filter(files) do f
        b = basename(f)
        ok = startswith(b, prefix) &&
             occursin(tao_piece, b) &&
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

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, tao, angle_conv, require_hessian=true)
    matches = list_matching_jsons(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        tao = tao,
        require_hessian = require_hessian,
    )

    tol = 1e-14
    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        t = extract_tao_from_filename(f)
        c !== nothing &&
        t !== nothing &&
        isapprox(c, angle_conv; atol=tol, rtol=0.0) &&
        isapprox(t, tao; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error("No matching aggregated JSON found for tao=$(tao), conv=$(angle_conv).")

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function blank_panel_axis(figpos; xticks=nothing)
    ax = styled_axis(figpos; xlabel = "", ylabel = "", xticks = xticks)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

function load_plot_data(; N=50, k=3, ngraphs=100, n_inits=10000, angle_bin=0.01, tao=0.05, angle_conv_main=1e-8)
    path_main = newest_matching_json(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        tao = tao,
        angle_conv = angle_conv_main,
        require_hessian = true,
    )

    @info "Using main path" path_main

    d = load_panel_abc_curves(path_main)

    λs = d.λs
    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_a = isfinite.(λs) .&
             isfinite.(d.devTheta_abs) .&
             isfinite.(d.devTheta_abs_err) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_a_inset_zoom = isfinite.(λs) .&
                        isfinite.(d.devTheta_abs) .&
                        isfinite.(d.devTheta_abs_err) .&
                        (d.devTheta_abs .> 0.0) .&
                        (λs .>= 0.0) .&
                        (λs .<= 0.25)

    mask_b = isfinite.(λs) .&
             isfinite.(d.nmin) .&
             isfinite.(d.nmin_err) .&
             isfinite.(d.nstar) .&
             isfinite.(d.nstar_err) .&
             isfinite.(d.mean_ratio) .&
             (d.nmin .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_kmean = isfinite.(λs) .&
                 isfinite.(d.κmean_all) .&
                 isfinite.(d.κmean_all_err) .&
                 (d.κmean_all .> 0.0)

    mask_ratio = isfinite.(d.mean_ratio) .&
                 isfinite.(λs) .&
                 (λs .!= 0.0) .&
                 (λs .!= 1.0)

    mask_succ = isfinite.(λs) .&
                isfinite.(d.succ_rate) .&
                isfinite.(d.succ_rate_err) .&
                (λs .>= 0.0) .&
                (λs .<= 1.0)

    mask_gn = isfinite.(λs) .&
              isfinite.(d.gn_init) .&
              isfinite.(d.gn_init_err) .&
              (d.gn_init .> 0.0) .&
              (λs .>= 0.0) .&
              (λs .<= 1.0)

    any(mask_a) || error("No usable entries for panel (a).")
    any(mask_a_inset_zoom) || error("No usable positive entries for panel (a) inset.")
    any(mask_b) || error("No usable entries for panel (b).")
    any(mask_kmean) || error("No usable mean-κ entries for panel (b)/(d) inset.")
    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    any(mask_succ) || error("No usable success-rate entries for panel (c) inset.")
    any(mask_gn) || error("No usable grad-norm-init entries for panel (d).")

    ratio_min = minimum(d.mean_ratio[mask_ratio])
    ratio_max = maximum(d.mean_ratio[mask_ratio])

    λk = Float64[]
    κk = Float64[]
    for (i, λ) in enumerate(λs)
        for κ in d.opt_min_hess_conds[i]
            if isfinite(κ) && κ > 0
                push!(λk, λ)
                push!(κk, κ)
            end
        end
    end
    isempty(λk) && error("No positive optimal-minimum condition numbers found.")

    λp = Float64[]
    pp = Float64[]
    for (i, λ) in enumerate(λs)
        for p in d.opt_min_probs_raw[i]
            if isfinite(p) && p > 0
                push!(λp, λ)
                push!(pp, p)
            end
        end
    end
    isempty(λp) && error("No positive optimal-minimum probabilities found.")

    external_data_base_dir = "/Users/guillermo.preisser/Projects/data_mechanicsm_qiils/optimization"
    λk_ext, κk_ext = load_external_optimal_cond_cloud(;
        n = 50,
        seed_start = 1,
        seed_end = 100,
        data_base_dir = external_data_base_dir,
    )

    println("Panel (d) inset external cloud points loaded = ", length(κk_ext))
    if !isempty(κk_ext)
        println("First few external points:")
        for i in 1:min(10, length(κk_ext))
            @printf("  i=%d   λ=%.4f   κ=%.6e\n", i, λk_ext[i], κk_ext[i])
        end
    else
        println("WARNING: λk_ext / κk_ext is empty, so the inset cloud will not appear.")
    end

    return (
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        tao = tao,
        angle_conv_main = angle_conv_main,
        path_main = path_main,
        λs = λs,
        devTheta_abs = d.devTheta_abs,
        devTheta_abs_err = d.devTheta_abs_err,
        nmin = d.nmin,
        nmin_err = d.nmin_err,
        nstar = d.nstar,
        nstar_err = d.nstar_err,
        mean_ratio = d.mean_ratio,
        κmean_all = d.κmean_all,
        κmean_all_err = d.κmean_all_err,
        succ_rate = d.succ_rate,
        succ_rate_err = d.succ_rate_err,
        gn_init = d.gn_init,
        gn_init_err = d.gn_init_err,
        λk = λk,
        κk = κk,
        λk_ext = λk_ext,
        κk_ext = κk_ext,
        λp = λp,
        pp = pp,
        mask_a = mask_a,
        mask_a_inset_zoom = mask_a_inset_zoom,
        mask_b = mask_b,
        mask_kmean = mask_kmean,
        mask_succ = mask_succ,
        mask_gn = mask_gn,
        ratio_min = ratio_min,
        ratio_max = ratio_max,
    )
end

function mean_kappa_from_cloud(λ, κ; κmax=Inf)
    keep = finite_positive_mask(κ) .& (κ .<= κmax)

    λf = λ[keep]
    κf = κ[keep]

    isempty(λf) && return Float64[], Float64[], Float64[]

    λu = sort(unique(λf))
    μ = Float64[]
    σ = Float64[]

    for l in λu
        mask = λf .== l
        vals = κf[mask]
        push!(μ, mean(vals))
        push!(σ, length(vals) > 1 ? std(vals) / sqrt(length(vals)) : 0.0)
    end

    return λu, μ, σ
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

function weighted_basin_curve(data)
    λu = sort(unique(data.λp))
    y = Float64[]

    for λ in λu
        mask = data.λp .== λ
        ps = data.pp[mask]
        push!(y, sum(ps .^ 2) / data.ngraphs)
    end

    return λu, y
end

function trivial_curve(data)
    λu = sort(unique(data.λp))
    y = Float64[]

    for λ in λu
        mask = data.λp .== λ
        ps = data.pp[mask]
        push!(y, sum(ps) / data.ngraphs)
    end

    return λu, y
end

function plot_panel_ab(data)
    let
        fig = Figure(
            size = (600, 1000),
            tellwidth = false,
            tellheight = false,
            figure_padding = (8, -50, 8, 8),
        )

        xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

        # ------------------------------------------------------------
        # (a)
        # ------------------------------------------------------------
        axa = styled_axis(fig[1, 1:2];
            xlabel = L"\lambda",
            ylabel = L"\frac{4}{\pi} |\theta - \pi/4| ",
            xticks = xticks_to1
        )

        errorbars!(axa,
            data.λs[data.mask_a],
            data.devTheta_abs[data.mask_a] .* (4 / pi),
            data.devTheta_abs_err[data.mask_a] .* (4 / pi);
            whiskerwidth = 0
        )

        scatterlines!(axa,
            data.λs[data.mask_a],
            data.devTheta_abs[data.mask_a] .* (4 / pi);
            markersize = 8,
            marker = :circle
        )

        axa_inset = Axis(fig[1, 1:2];
            width = Relative(0.46),
            height = Relative(0.46),
            halign = 0.94,
            valign = 0.29,
            xlabel = L"\lambda",
            ylabel = L"\frac{4}{\pi} |\theta - \pi/4| ",
            xticks = (0.0:0.1:0.2, [@sprintf("%.1f", x) for x in 0.0:0.1:0.2]),
            yticks = ([10^-13.5, 1e-13, 10^-12.5],
          [L"10^{-13.5}", L"10^{-13.0}", L"10^{-12.5}"]),
            yscale = log10,
            xticklabelsize = INSET_TICKSIZE,
            yticklabelsize = INSET_TICKSIZE,
            xlabelsize = INSET_LABELSIZE,
            ylabelsize = INSET_LABELSIZE,
            xgridvisible = false,
            ygridvisible = false,
            xtickalign = 1,
            ytickalign = 1,
            xticksmirrored = true,
            yticksmirrored = true,
            backgroundcolor = (:white, 0.95),
        )
        axa_inset.xlabelpadding = -5
        axa_inset.ylabelpadding = 0

        ylims!(axa, -0.49, nothing)

        errorbars!(axa_inset,
            data.λs[data.mask_a_inset_zoom],
            data.devTheta_abs[data.mask_a_inset_zoom] .* (4 / pi),
            data.devTheta_abs_err[data.mask_a_inset_zoom] .* (4 / pi);
            whiskerwidth = 5
        )

        scatterlines!(axa_inset,
            data.λs[data.mask_a_inset_zoom],
            data.devTheta_abs[data.mask_a_inset_zoom] .* (4 / pi);
            marker = :circle,
            markersize = 6,
            linewidth = 1.5
        )

        # ------------------------------------------------------------
        # (b)
        # ------------------------------------------------------------
        axb = styled_axis(fig[2, 1];
            xlabel = L"\lambda",
            ylabel = L"N_{\mathrm{minima}}",
            xticks = xticks_to1,
            yscale = log10,
        )

        ylims!(axb, 0.02, nothing)

        x_b = data.λs[data.mask_b]
        y_b = data.nmin[data.mask_b]
        e_b = data.nmin_err[data.mask_b]
        c_b = data.mean_ratio[data.mask_b]

        mask_whisk = data.mask_b .& (data.λs .<= 0.45)
        mask_nowhisk = data.mask_b .& (data.λs .> 0.45)

        errorbars!(axb,
            data.λs[mask_whisk], data.nstar[mask_whisk], data.nstar_err[mask_whisk];
            color = :black,
            whiskerwidth = 6,
        )

        errorbars!(axb,
            data.λs[mask_nowhisk], data.nstar[mask_nowhisk], data.nstar_err[mask_nowhisk];
            color = :black,
            whiskerwidth = 0,
        )

        scatterlines!(axb,
            x_b, data.nstar[data.mask_b];
            color = :black,
            markercolor = :yellow,
            marker = :circle,
            markersize = 8,
            linewidth = 1.8,
            label = L"N^*",
            strokecolor = :black,
            strokewidth = 1.0,
        )

        errorbars!(axb,
            x_b, y_b, e_b;
            color = c_b,
            colormap = :viridis,
            colorrange = (data.ratio_min, data.ratio_max),
            whiskerwidth = 6
        )

        hm = scatter!(axb,
            x_b, y_b;
            color = c_b,
            colormap = :viridis,
            colorrange = (data.ratio_min, data.ratio_max),
            markersize = 8
        )

        axislegend(axb;
            position = :lt,
            labelsize = 16,
            padding = (4, 4, 4, 4),
            framevisible = false
        )

        cb = Colorbar(fig[2, 2], hm;
            label = L"r",
            width = 12,
            ticklabelsize = 14,
            labelsize = 15
        )
        colsize!(fig.layout, 2, Fixed(120))
        Label(fig[1, 2], ""; tellheight = false)
        Label(fig[3, 2], ""; tellheight = false)

        Box(fig[1, 2], visible = false)
        Box(fig[3, 2], visible = false)

        axb_inset = Axis(fig[2, 1];
            width = Relative(0.48),
            height = Relative(0.38),
            halign = 0.95,
            valign = 0.18,
            xlabel = L"\lambda",
            ylabel = L"\kappa",
            xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
            yticks = (
                [10^0.5, 10^1.0, 10^1.5],
                [L"10^{0.5}", L"10^{1.0}", L"10^{1.5}"]
            ),
            yscale = log10,
            xticklabelsize = INSET_TICKSIZE,
            yticklabelsize = INSET_TICKSIZE,
            xlabelsize = INSET_LABELSIZE,
            ylabelsize = INSET_LABELSIZE,
            xgridvisible = false,
            ygridvisible = false,
            xtickalign = 1,
            ytickalign = 1,
            xticksmirrored = true,
            yticksmirrored = true,
            backgroundcolor = (:white, 0.95),
        )
        axb_inset.xlabelpadding = -3

        scatter!(axb_inset,
            data.λk_ext, data.κk_ext;
            color = (Makie.wong_colors()[1], 0.10),
            markersize = 4.5
        )

        λm_ext, κm_ext, κm_ext_err = mean_kappa_from_cloud(
            data.λk_ext, data.κk_ext; κmax = 100.0
        )

        errorbars!(axb_inset,
            λm_ext, κm_ext, κm_ext_err;
            color = :black,
            whiskerwidth = 5
        )

        scatterlines!(axb_inset,
            λm_ext, κm_ext;
            color = :black,
            marker = :circle,
            markersize = 5,
            linewidth = 1.5
        )

        λm_json, κm_json, κm_json_err = kappa_curve_from_cloud(
            data.λk, data.κk; stat = :median
        )

        errorbars!(axb_inset,
            λm_json, κm_json, κm_json_err;
            color = Makie.wong_colors()[2],
            whiskerwidth = 5
        )

        scatterlines!(axb_inset,
            λm_json, κm_json;
            color = Makie.wong_colors()[2],
            marker = :rect,
            markersize = 5,
            linewidth = 1.5
        )

        # ------------------------------------------------------------
        # (c)
        # ------------------------------------------------------------
        axc = styled_axis(fig[3, 1:2];
            xlabel = L"\lambda",
            ylabel = L"P_{\mathrm{solved}}",
            xticks = xticks_to1,
            yticks = (0.0:0.2:1.0, [@sprintf("%.1f", y) for y in 0.0:0.2:1.0]),
        )
        ylims!(axc, nothing, 1.0)

        errorbars!(axc,
            data.λs[data.mask_succ],
            data.succ_rate[data.mask_succ],
            data.succ_rate_err[data.mask_succ];
            whiskerwidth = 6
        )

        scatterlines!(axc,
            data.λs[data.mask_succ],
            data.succ_rate[data.mask_succ];
            color = Makie.wong_colors()[1],
            marker = :circle,
            markersize = 8,
            linewidth = 1.8
        )

        axc_inset = Axis(fig[3, 1:2];
            width = Relative(0.46),
            height = Relative(0.46),
            halign = 0.95,
            valign = 0.95,
            xlabel = L"\lambda",
            ylabel = L"\left| \mathbf{g} \right|",
            xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
            xticklabelsize = INSET_TICKSIZE,
            yticklabelsize = INSET_TICKSIZE,
            xlabelsize = INSET_LABELSIZE,
            ylabelsize = INSET_LABELSIZE,
            xgridvisible = false,
            ygridvisible = false,
            xtickalign = 1,
            ytickalign = 1,
            xticksmirrored = true,
            yticksmirrored = true,
            backgroundcolor = (:white, 0.95),
        )
        axc_inset.xlabelpadding = -5

        errorbars!(axc_inset,
            data.λs[data.mask_gn],
            data.gn_init[data.mask_gn],
            data.gn_init_err[data.mask_gn];
            whiskerwidth = 5
        )

        scatterlines!(axc_inset,
            data.λs[data.mask_gn],
            data.gn_init[data.mask_gn];
            marker = :circle,
            markersize = 5,
            linewidth = 1.5
        )

        rowgap!(fig.layout, 28)
        colgap!(fig.layout, 1, -40)

        resize_to_layout!(fig)

        text!(fig.scene, "(a)"; position = (0.01, 0.97), space = :relative, fontsize = PANEL_LABELSIZE)
        text!(fig.scene, "(b)"; position = (0.01, 0.64), space = :relative, fontsize = PANEL_LABELSIZE)
        text!(fig.scene, "(c)"; position = (0.01, 0.31), space = :relative, fontsize = PANEL_LABELSIZE)

        abin_tag = replace(@sprintf("%.3g", data.angle_bin), "." => "p")
        conv_main_tag = replace(@sprintf("%.0e", data.angle_conv_main), "." => "p")
        tao_tag = replace(@sprintf("%.3g", data.tao), "." => "p")

        out_png = joinpath(
            PLOTS_DIR,
            "qiigs_abc_panel_vertical_unweighted_N$(data.N)_k$(data.k)_ngraphs$(data.ngraphs)_ninit$(data.n_inits)_tao$(tao_tag)_abin$(abin_tag)_mainconv$(conv_main_tag).png"
        )

        save(out_png, fig)
        @info "Saved $out_png"

        display(fig)
        return fig
    end
end

# Interactive usage:
data = load_plot_data(; tao=0.05)
fig = plot_panel_ab(data)

# Uncomment these only when you want full reload:
# data = load_plot_data(; tao=0.05)
# fig = plot_panel_ab(data)