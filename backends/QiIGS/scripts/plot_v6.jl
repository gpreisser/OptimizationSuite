# scripts/plot_hessian_angles_ab.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using LaTeXStrings
using Printf
using Random
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

function load_success_curve(json_path::AbstractString)
    λs, _, _, _, _, _, _, _, succ_rate, succ_rate_err, _, _, _, _ = load_curves(json_path)
    mask = isfinite.(λs) .& isfinite.(succ_rate) .& isfinite.(succ_rate_err)
    return λs[mask], succ_rate[mask], succ_rate_err[mask]
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

function main()
    N = 50
    k = 3
    ngraphs = 100
    n_inits = 10000
    angle_bin = 0.01

    angle_conv_main = 1e-8

    panelb_conv   = 1e-8
    panelb_marker = :circle
    panelb_color  = "#0072B2"

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
    succ_rate_main, succ_rate_err_main,
    devTheta_abs, devTheta_abs_err,
    gn_init, gn_init_err = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_a = isfinite.(λs) .&
             isfinite.(devTheta_abs) .&
             isfinite.(devTheta_abs_err) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_a_inset_zoom = isfinite.(λs) .&
                        isfinite.(devTheta_abs) .&
                        isfinite.(devTheta_abs_err) .&
                        (devTheta_abs .> 0.0) .&
                        (λs .>= 0.0) .&
                        (λs .<= 0.25)

    mask_b = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(uniques_angle_err) .&
             isfinite.(mean_ratio) .&
             (uniques_angle .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    any(mask_a) || error("No usable entries for panel (a).")
    any(mask_a_inset_zoom) || error("No usable positive entries for panel (a) inset on log scale.")
    any(mask_b) || error("No usable positive entries for panel (b).")

    mask_ratio = isfinite.(mean_ratio) .&
                 isfinite.(λs) .&
                 (λs .!= 0.0) .&
                 (λs .!= 1.0)

    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    ratio_min = minimum(mean_ratio[mask_ratio])
    ratio_max = maximum(mean_ratio[mask_ratio])

    fig = Figure(
        size = (930, 360),
        tellwidth = false,
        tellheight = false,
        figure_padding = (8, 8, 8, 8),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    # ------------------------------------------------------------
    # (a)
    # ------------------------------------------------------------
    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"{\frac{4}{\pi}} |\theta - \pi/4| ",
        xticks = xticks_to1
    )

    ylims!(axa, -0.49, nothing)

    errorbars!(axa,
        λs[mask_a],
        devTheta_abs[mask_a] .* (4 / pi),
        devTheta_abs_err[mask_a] .* (4 / pi);
        whiskerwidth = 0
    )

    scatterlines!(axa,
        λs[mask_a], devTheta_abs[mask_a] .* (4 / pi);
        markersize = 8,
        marker = :circle
    )

    axa_inset = Axis(fig[1, 1];
        width = Relative(0.46),
        height = Relative(0.46),
        halign = 0.94,
        valign = 0.19,
        xlabel = L"\lambda",
        ylabel = L"{\frac{4}{\pi}}|\theta - \pi/4| ",
        xticks = (0.0:0.1:0.2, [@sprintf("%.1f", x) for x in 0.0:0.1:0.2]),
        yticks = ([1e-14, 1e-13], [L"10^{-14}", L"10^{-13}"]),
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

    errorbars!(axa_inset,
        λs[mask_a_inset_zoom],
        devTheta_abs[mask_a_inset_zoom] .* (4 / pi),
        devTheta_abs_err[mask_a_inset_zoom] .* (4 / pi);
        whiskerwidth = 5
    )

    scatterlines!(axa_inset,
        λs[mask_a_inset_zoom],
        devTheta_abs[mask_a_inset_zoom] .* (4 / pi);
        marker = :circle,
        markersize = 6,
        linewidth = 1.5
    )

    # ------------------------------------------------------------
    # (b)
    # ------------------------------------------------------------
    right = GridLayout()
    fig[1, 2] = right
    colgap!(right, 8)

    axb = styled_axis(right[1, 1];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{minima}}",
        xticks = xticks_to1,
        yscale = log10,
    )

    ylims!(axb, 0.1, nothing)

    x_b = λs[mask_b]
    y_b = uniques_angle[mask_b]
    e_b = uniques_angle_err[mask_b]
    c_b = mean_ratio[mask_b]

    errorbars!(axb,
        x_b, y_b, e_b;
        color = c_b,
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        whiskerwidth = 6
    )

    hm = scatter!(axb,
        x_b, y_b;
        color = c_b,
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8
    )

    path_b = newest_matching_json(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        angle_conv = panelb_conv,
        require_hessian = true,
    )

    @info "Using inset path" panelb_conv path_b

    λins, succ_ins, succ_ins_err = load_success_curve(path_b)
    isempty(λins) && error("No usable success-rate entries for conv=$(panelb_conv).")

    axb_inset = Axis(right[1, 1];
        width = Relative(0.46),
        height = Relative(0.46),
        halign = 0.955,
        valign = 0.19,
        xlabel = L"\lambda",
        ylabel = L"\text{P_{solved}}",
        xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
        yticks = (0.0:0.2:0.6, [@sprintf("%.1f", x) for x in 0.0:0.2:0.6]),
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
    axb_inset.xlabelpadding = -5

    errorbars!(axb_inset,
        λins, succ_ins, succ_ins_err;
        whiskerwidth = 5
    )

    scatterlines!(axb_inset,
        λins, succ_ins;
        marker = panelb_marker,
        color = panelb_color,
        markersize = 6
    )

    Colorbar(right[1, 2], hm;
        label = L" r",
        width = 16,
        ticklabelsize = 14,
        labelsize = 15
    )
    colsize!(right, 2, Fixed(22))

    colgap!(fig.layout, 28)
    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position = (0.01, 0.94), space = :relative, fontsize = PANEL_LABELSIZE)
    text!(fig.scene, "(b)"; position = (0.48, 0.94), space = :relative, fontsize = PANEL_LABELSIZE)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_main_tag = replace(@sprintf("%.0e", angle_conv_main), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_ab_v9_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_mainconv$(conv_main_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()