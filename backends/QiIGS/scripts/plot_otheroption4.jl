# scripts/plot_conor_graph12_unweighted_N50_story_ordered.jl

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

function load_curves(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    unique_angle_raw = Float64[]
    hess_mineig = Float64[]
    hess_cond = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]

    for entry in rpl
        λ   = getnum(entry, "λ")
        ua  = getnum(entry, "unique_angle_count_raw_mean")
        hm  = getnum(entry, "hess_mineig_mean_mean")
        hc  = getnum(entry, "hess_cond_mean_mean")
        mr  = getnum(entry, "mean_ratio_mean")
        sr  = getnum(entry, "success_rate_mean")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_raw, ua)
        push!(hess_mineig, hm)
        push!(hess_cond, hc)
        push!(mean_ratio, mr)
        push!(succ_rate, sr)
    end

    p = sortperm(λs)

    return (
        λs[p],
        unique_angle_raw[p],
        hess_mineig[p],
        hess_cond[p],
        mean_ratio[p],
        succ_rate[p],
    )
end

function load_success_curve(json_path::AbstractString)
    λs, _, _, _, _, succ_rate = load_curves(json_path)
    mask = isfinite.(λs) .& isfinite.(succ_rate)
    return λs[mask], succ_rate[mask]
end

function styled_axis(figpos; xlabel, ylabel, xticks=nothing, yscale=nothing)
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

    if xticks === nothing && yscale === nothing
        return Axis(figpos; common...)
    elseif xticks !== nothing && yscale === nothing
        return Axis(figpos; xticks=xticks, common...)
    elseif xticks === nothing && yscale !== nothing
        return Axis(figpos; yscale=yscale, common...)
    else
        return Axis(figpos; xticks=xticks, yscale=yscale, common...)
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

function conv_label(conv::Float64)
    if isapprox(conv, 1e-2; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-2}"
    elseif isapprox(conv, 1e-8; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-8}"
    else
        error("Unsupported conv = $conv")
    end
end

function main()
    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.02

    # Main conv for panels (a), (c), (d)
    angle_conv_main = 1e-8

    # Panel (b): success overlay
    panelb_convs   = [1e-2, 1e-8]
    panelb_markers = [:dtriangle, :circle]
    panelb_colors  = ["#E69F00", "#0072B2"]

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

    λs, uniques_angle, hess_mineig, hess_cond,
    mean_ratio, succ_rate_main = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_a = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(mean_ratio) .&
             (uniques_angle .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_c = isfinite.(λs) .&
             isfinite.(hess_mineig) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_d = isfinite.(λs) .&
             isfinite.(hess_cond) .&
             (hess_cond .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    any(mask_a) || error("No usable entries for panel (a).")
    any(mask_c) || error("No usable entries for panel (c).")
    any(mask_d) || error("No usable entries for panel (d).")

    # Ignore λ = 0 and λ = 1 only for color scaling
    mask_ratio = isfinite.(mean_ratio) .&
                 isfinite.(λs) .&
                 (λs .!= 0.0) .&
                 (λs .!= 1.0)

    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    ratio_min = minimum(mean_ratio[mask_ratio])
    ratio_max = maximum(mean_ratio[mask_ratio])

    fig = Figure(
        size = (1650, 340),
        tellwidth = false,
        tellheight = false,
        figure_padding = (6, 6, 6, 6),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    # ------------------------------------------------------------
    # (a) minima count (log y)
    # ------------------------------------------------------------
    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{minima}}",
        xticks = xticks_to1,
        yscale = log10
    )

    hm = scatter!(axa,
        λs[mask_a], uniques_angle[mask_a];
        color = mean_ratio[mask_a],
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
    # (b) success probability overlay
    # ------------------------------------------------------------
    axb = styled_axis(fig[1, 3];
        xlabel = L"\lambda",
        ylabel = L"P(\mathrm{success})",
        xticks = xticks_to1
    )

    plots_b = Any[]
    labels_b = Any[]
    marksize1 = 9

    for (conv, marker, col) in zip(panelb_convs, panelb_markers, panelb_colors)
        path_b = newest_matching_json(;
            N=N,
            k=k,
            ngraphs=ngraphs,
            n_inits=n_inits,
            angle_bin=angle_bin,
            angle_conv=conv,
            require_hessian=true,
        )

        @info "Using panel (b) path" conv path_b

        λb, ps = load_success_curve(path_b)
        isempty(λb) && error("No usable success-rate entries for conv=$(conv).")

        pb = scatterlines!(axb,
            λb, ps;
            marker = marker,
            color = col,
            markersize = marksize1
        )

        push!(plots_b, pb)
        push!(labels_b, conv_label(conv))
    end

    axislegend(axb, plots_b, labels_b;
        position = :rt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 12
    )

    # ------------------------------------------------------------
    # (c) minimum Hessian eigenvalue
    # ------------------------------------------------------------
    axc = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\langle \lambda_{\min}(\nabla^2 f_\lambda) \rangle",
        xticks = xticks_to1
    )

    scatterlines!(axc,
        λs[mask_c], hess_mineig[mask_c];
        markersize = 8,
        marker = :circle
    )

    # ------------------------------------------------------------
    # (d) condition number (log y)
    # ------------------------------------------------------------
    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\langle \kappa(\nabla^2 f_\lambda) \rangle",
        xticks = xticks_to1,
        yscale = log10
    )

    scatterlines!(axd,
        λs[mask_d], hess_cond[mask_d];
        markersize = 8,
        marker = :circle
    )

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.00, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.29, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(c)"; position=(0.53, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(d)"; position=(0.757, 0.92), space=:relative, fontsize=18)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_main_tag = replace(@sprintf("%.0e", angle_conv_main), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_story_abcd_ordered_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_mainconv$(conv_main_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()