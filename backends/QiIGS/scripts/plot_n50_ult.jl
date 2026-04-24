# scripts/plot_conor_graph12_unweighted_N50_multiconv_panelc.jl

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
    unique_round = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]
    gn_init = Float64[]
    gn_meaninner = Float64[]

    for entry in rpl
        λ   = getnum(entry, "λ")
        ua  = getnum(entry, "unique_angle_count_raw_mean")
        ur  = getnum(entry, "unique_count_mean")
        mr  = getnum(entry, "mean_ratio_mean")
        sr  = getnum(entry, "success_rate_mean")
        gni = getnum(entry, "grad_norm_init_mean")
        gnm = getnum(entry, "grad_norm_meaninner_mean")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_raw, ua)
        push!(unique_round, ur)
        push!(mean_ratio, mr)
        push!(succ_rate, sr)
        push!(gn_init, gni)
        push!(gn_meaninner, gnm)
    end

    p = sortperm(λs)

    return (
        λs[p],
        unique_angle_raw[p],
        unique_round[p],
        mean_ratio[p],
        succ_rate[p],
        gn_init[p],
        gn_meaninner[p],
    )
end

function load_success_curve(json_path::AbstractString)
    λs, _, _, _, succ_rate, _, _ = load_curves(json_path)
    mask = isfinite.(λs) .& isfinite.(succ_rate)
    return λs[mask], succ_rate[mask]
end

function styled_axis(figpos; xlabel, ylabel, xticks=nothing)
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
        return Axis(figpos; common...)
    else
        return Axis(figpos; xticks=xticks, common...)
    end
end

function conv_tag_to_float(tag::AbstractString)
    # Handles tags such as:
    #   0p01, 0p0001, 1e-06, 1e-08, ...
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
    abin_piece = "_abin$(abin_tag)_ngraphs$(ngraphs)_ninit$(n_inits)_outer1_inner5000_thr0p999.json"

    matches = filter(files) do f
        b = basename(f)
        startswith(b, prefix) &&
        occursin(abin_piece, b) &&
        !occursin("checkpoint", lowercase(b))
    end

    isempty(matches) && error("No matching aggregated JSON found in:\n$subdir")
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
    elseif isapprox(conv, 1e-4; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-4}"
    elseif isapprox(conv, 1e-6; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-6}"
    elseif isapprox(conv, 1e-8; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-8}"
    elseif isapprox(conv, 1e-10; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-10}"
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

    # Main conv used for panels (a), (b), (d)
    angle_conv_main = 1e-8

    # Panel (c) overlays these conv values
    panelc_convs   = [1e-2, 1e-4, 1e-6, 1e-8]
    panelc_markers = [:circle, :utriangle, :dtriangle, :diamond]
    panelc_colors  = [nothing, nothing, "green", "purple"]

    path_main = newest_matching_json(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv_main,
    )

    @info "Using main path" path_main

    λs, uniques_angle, uniques_round, mean_ratio,
    succ_rate_main, gn_init, gn_meaninner = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    # ------------------------------------------------------------
    # Panel-specific masks
    # ------------------------------------------------------------

    # Panel (a): keep 0 ≤ λ < 1
    mask_a = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(mean_ratio) .&
             (λs .>= 0.0) .&
             (λs .< 1.0)

    # Panel (b): keep λ ≥ 0 except exactly λ = 1.0
    mask_b = isfinite.(λs) .&
             isfinite.(uniques_round) .&
             isfinite.(mean_ratio) .&
             (λs .>= 0.0) .&
             (λs .!= 1.0)

    # Panel (d): natural x-axis
    mask_gni = isfinite.(λs) .& isfinite.(gn_init)
    mask_gnm = isfinite.(λs) .& isfinite.(gn_meaninner)

    any(mask_a) || error("No usable entries for panel (a) after removing λ = 1.0.")
    any(mask_b) || error("No usable entries for panel (b) after removing λ = 1.0.")
    (any(mask_gni) || any(mask_gnm)) || error("No usable entries for panel (d).")

    # Ignore λ = 0 and λ = 1 only for color scaling
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
    xticks_no1 = (0.0:0.2:0.8, [@sprintf("%.1f", x) for x in 0.0:0.2:0.8])

    # ------------------------------------------------------------
    # (a)
    # ------------------------------------------------------------
    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{unique}}^{\mathrm{angle,raw}}",
        xticks = xticks_no1
    )

    hm = scatter!(axa,
        λs[mask_a], uniques_angle[mask_a];
        color = mean_ratio[mask_a],
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8
    )

    # ------------------------------------------------------------
    # (b)
    # ------------------------------------------------------------
    axb = styled_axis(fig[1, 2];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{unique}}^{\mathrm{round}}",
        xticks = xticks_to1
    )

    scatter!(axb,
        λs[mask_b], uniques_round[mask_b];
        color = mean_ratio[mask_b],
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8
    )

    Colorbar(fig[1, 3], hm;
        label = L"\langle r\rangle",
        width = 14,
        ticklabelsize = 14,
        labelsize = 15
    )
    colsize!(fig.layout, 3, Auto(22))

    # ------------------------------------------------------------
    # (c) multi-conv success-rate overlay
    # ------------------------------------------------------------
    axc = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\mathrm{success\ rate}",
        xticks = xticks_to1
    )

    plots_c = Any[]
    labels_c = Any[]
    marksize1 = 9

    for (conv, marker, col) in zip(panelc_convs, panelc_markers, panelc_colors)
        path_c = newest_matching_json(;
            N=N,
            k=k,
            ngraphs=ngraphs,
            n_inits=n_inits,
            angle_bin=angle_bin,
            angle_conv=conv,
        )

        @info "Using panel (c) path" conv path_c

        λc, src = load_success_curve(path_c)
        isempty(λc) && error("No usable success-rate entries for conv=$(conv).")

        pc = if col === nothing
            scatterlines!(axc,
                λc, src;
                marker = marker,
                markersize = marksize1
            )
        else
            scatterlines!(axc,
                λc, src;
                marker = marker,
                color = col,
                markersize = marksize1
            )
        end

        push!(plots_c, pc)
        push!(labels_c, conv_label(conv))
    end

    axislegend(axc, plots_c, labels_c;
        position = :rt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 12
    )

    # ------------------------------------------------------------
    # (d)
    # ------------------------------------------------------------
    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\langle \Vert \nabla \Vert \rangle",
        xticks = xticks_to1
    )

    plots_d = Any[]
    labels_d = Any[]

    if any(mask_gni)
        p1 = scatterlines!(axd,
            λs[mask_gni], gn_init[mask_gni];
            markersize = 8,
            marker = :circle
        )
        push!(plots_d, p1)
        push!(labels_d, L"\mathrm{init}")
    end

    if any(mask_gnm)
        p2 = scatterlines!(axd,
            λs[mask_gnm], gn_meaninner[mask_gnm];
            markersize = 8,
            marker = :utriangle
        )
        push!(plots_d, p2)
        push!(labels_d, L"\mathrm{mean\ inner}")
    end

    if !isempty(plots_d)
        axislegend(axd, plots_d, labels_d;
            position = :lt,
            framevisible = false,
            backgroundcolor = :transparent,
            labelsize = 14
        )
    end

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.00, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.24, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(c)"; position=(0.52, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(d)"; position=(0.77, 0.92), space=:relative, fontsize=18)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_main_tag = replace(@sprintf("%.0e", angle_conv_main), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_abcd_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_mainconv$(conv_main_tag)_panelc_multiconv.png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()