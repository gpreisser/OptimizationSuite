# scripts/plot_conor_N50_unweighted_multiconv_panelc_devtheta.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using LaTeXStrings
using Printf

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

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

    entry1 = rpl[1]

    haskey(entry1, "devTheta_abs_mean") || return false
    haskey(entry1, "devTheta_abs_stderr") || return false
    haskey(entry1, "devZ_abs_mean") || return false
    haskey(entry1, "devZ_abs_stderr") || return false

    haskey(data, "solver") || return false
    solver = data["solver"]
    haskey(solver, "outputs") || return false
    outs = solver["outputs"]

    return ("devTheta_abs" in outs) && ("devZ_abs" in outs)
end

function load_curves(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    unique_angle_raw = Float64[]
    unique_angle_raw_err = Float64[]
    hess_mineig = Float64[]
    hess_mineig_err = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]
    succ_rate_err = Float64[]
    devTheta_abs = Float64[]
    devTheta_abs_err = Float64[]

    for entry in rpl
        λ    = getnum(entry, "λ")
        ua   = getnum(entry, "unique_angle_count_raw_mean")
        uae  = getnum(entry, "unique_angle_count_raw_stderr")
        hm   = getnum(entry, "hess_mineig_mean_mean")
        hme  = getnum(entry, "hess_mineig_mean_stderr")
        mr   = getnum(entry, "mean_ratio_mean")
        sr   = getnum(entry, "success_rate_mean")
        sre  = getnum(entry, "success_rate_stderr")
        dθ   = getnum(entry, "devTheta_abs_mean")
        dθe  = getnum(entry, "devTheta_abs_stderr")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_raw, ua)
        push!(unique_angle_raw_err, uae)
        push!(hess_mineig, hm)
        push!(hess_mineig_err, hme)
        push!(mean_ratio, mr)
        push!(succ_rate, sr)
        push!(succ_rate_err, sre)
        push!(devTheta_abs, dθ)
        push!(devTheta_abs_err, dθe)
    end

    p = sortperm(λs)

    return (
        λs[p],
        unique_angle_raw[p],
        unique_angle_raw_err[p],
        hess_mineig[p],
        hess_mineig_err[p],
        mean_ratio[p],
        succ_rate[p],
        succ_rate_err[p],
        devTheta_abs[p],
        devTheta_abs_err[p],
    )
end

function load_success_curve(json_path::AbstractString)
    λs, _, _, _, _, _, succ_rate, succ_rate_err, _, _ = load_curves(json_path)
    mask = isfinite.(λs) .& isfinite.(succ_rate) .& isfinite.(succ_rate_err)
    return λs[mask], succ_rate[mask], succ_rate_err[mask]
end

function styled_axis(figpos; xlabel, ylabel, xticks=nothing, kwargs...)
    common = (;
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 16,
        ylabelsize = 16,
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
        return Axis(figpos; xticks=xticks, common..., kwargs...)
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

    matches = filter(json_is_v7_like, matches)

    isempty(matches) && error(
        "No matching v7-like aggregated JSON found in:\n$subdir\n" *
        "Check angle_bin and whether the updated sweep was rerun."
    )

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
        "No matching v7-like aggregated JSON found for conv=$(angle_conv).\n" *
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
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv_main,
        require_hessian=true,
    )

    @info "Using main path" path_main

    λs, uniques_angle, uniques_angle_err,
    hess_mineig, hess_mineig_err, mean_ratio,
    succ_rate_main, succ_rate_err_main,
    devTheta_abs, devTheta_abs_err = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_a = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(uniques_angle_err) .&
             isfinite.(mean_ratio) .&
             (uniques_angle .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_c = isfinite.(λs) .&
             isfinite.(hess_mineig) .&
             isfinite.(hess_mineig_err) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_d = isfinite.(λs) .&
             isfinite.(devTheta_abs) .&
             isfinite.(devTheta_abs_err) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    any(mask_a) || error("No usable positive entries for panel (a).")
    any(mask_c) || error("No usable entries for panel (c).")
    any(mask_d) || error("No usable entries for panel (d).")

    mask_ratio = isfinite.(mean_ratio) .&
                 isfinite.(λs) .&
                 (λs .!= 0.0) .&
                 (λs .!= 1.0)

    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    ratio_min = minimum(mean_ratio[mask_ratio])
    ratio_max = maximum(mean_ratio[mask_ratio])

    fig = Figure(
        size = (1550, 320),
        tellwidth = false,
        tellheight = false,
        figure_padding = (6, 6, 6, 6),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    # ------------------------------------------------------------
    # (a) minima count
    # ------------------------------------------------------------
    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{minima}}",
        xticks = xticks_to1,
        yscale = log10,
    )

   x_a = λs[mask_a]
y_a = uniques_angle[mask_a]
e_a = uniques_angle_err[mask_a]
c_a = mean_ratio[mask_a]

errorbars!(axa,
    x_a, y_a, e_a;
    color = c_a,
    colormap = :viridis,
    colorrange = (ratio_min, ratio_max),
    whiskerwidth = 6
)

hm = scatter!(axa,
    x_a, y_a;
    color = c_a,
    colormap = :viridis,
    colorrange = (ratio_min, ratio_max),
    markersize = 8
)

    Colorbar(fig[1, 2], hm;
        label = L"\langle r\rangle",
        width = 14,
        ticklabelsize = 14,
        labelsize = 15
    )
    colsize!(fig.layout, 2, Auto(22))

    # ------------------------------------------------------------
    # (b) success probability
    # ------------------------------------------------------------
    axb = styled_axis(fig[1, 3];
        xlabel = L"\lambda",
        ylabel = L"P(\mathrm{success})",
        xticks = xticks_to1
    )

    path_b = newest_matching_json(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=panelb_conv,
        require_hessian=true,
    )

    @info "Using panel (b) path" panelb_conv path_b

    λb, succ_b, succ_b_err = load_success_curve(path_b)
    isempty(λb) && error("No usable success-rate entries for conv=$(panelb_conv).")

    errorbars!(axb,
        λb, succ_b, succ_b_err;
        whiskerwidth = 8
    )

    scatterlines!(axb,
        λb, succ_b;
        marker = panelb_marker,
        color = panelb_color,
        markersize = 9
    )

    # ------------------------------------------------------------
    # (c) Hessian minimum eigenvalue
    # ------------------------------------------------------------
    axc = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\langle \lambda_{\min}(\nabla^2 f_\lambda) \rangle",
        xticks = xticks_to1
    )

    errorbars!(axc,
        λs[mask_c], hess_mineig[mask_c], hess_mineig_err[mask_c];
        whiskerwidth = 8
    )

    scatterlines!(axc,
        λs[mask_c], hess_mineig[mask_c];
        markersize = 8,
        marker = :circle
    )

    # ------------------------------------------------------------
    # (d) mean angle deviation
    # ------------------------------------------------------------
    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\langle |\theta - \pi/4| \rangle",
        xticks = xticks_to1
    )

    errorbars!(axd,
        λs[mask_d], devTheta_abs[mask_d], devTheta_abs_err[mask_d];
        whiskerwidth = 8
    )

    scatterlines!(axd,
        λs[mask_d], devTheta_abs[mask_d];
        markersize = 8,
        marker = :circle
    )

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.00, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.30, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(c)"; position=(0.535, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(d)"; position=(0.77, 0.92), space=:relative, fontsize=18)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_main_tag = replace(@sprintf("%.0e", angle_conv_main), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_abcd_allerrbars_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_mainconv$(conv_main_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()