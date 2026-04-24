# scripts/plot_conor_graph12.jl

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

function newest_matching_json(; N, k, graph_seed, n_inits, angle_bin, angle_conv)
    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "." => "p")

    subdir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_pm1weighted_N$(N)_k$(k)_seed$(graph_seed)"
    )
    isdir(subdir) || error("Directory not found: $subdir")

    files = readdir(subdir; join=true)

    prefix = "qiigs_unique_ratio_meanbest_succ_grad_pm1weighted_seed$(graph_seed)_lam0p000_to_1p000_d"
    suffix = "_conv$(conv_tag)_abin$(abin_tag)_ngraphs1_ninit$(n_inits)_outer1_inner5000_thr0p999.json"

    matches = filter(files) do f
        b = basename(f)
        startswith(b, prefix) &&
        endswith(b, suffix) &&
        !occursin("checkpoint", lowercase(b))
    end

    isempty(matches) && error("No matching aggregated JSON found in:\n$subdir")

    mtimes = [stat(f).mtime for f in matches]
    return matches[argmax(mtimes)]
end

function main()
    N = 12
    k = 3
    graph_seed = 2
    n_inits = 1000
    angle_bin = 0.03
    angle_conv = 1e-6

    path = newest_matching_json(;
        N=N,
        k=k,
        graph_seed=graph_seed,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv,
    )

    @info "Using" path

    λs, uniques_angle, uniques_round, mean_ratio,
    succ_rate, gn_init, gn_meaninner = load_curves(path)

    isempty(λs) && error("No usable λ entries found in JSON.")

    # ------------------------------------------------------------
    # Panel-specific masks
    # ------------------------------------------------------------

    # Panel (a): keep 0 ≤ λ < 1, so λ = 1.0 is removed
    mask_a = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(mean_ratio) .&
             (λs .>= 0.0) .&
             (λs .< 1.0)

    # Panel (b): keep everything with λ ≥ 0 except exactly λ = 1.0
    mask_b = isfinite.(λs) .&
             isfinite.(uniques_round) .&
             isfinite.(mean_ratio) .&
             (λs .>= 0.2) .&
             (λs .!= 1.0)

    # Panels (c) and (d): natural x-axis, only finite values
    mask_c = isfinite.(λs) .& isfinite.(succ_rate)
    mask_gni = isfinite.(λs) .& isfinite.(gn_init)
    mask_gnm = isfinite.(λs) .& isfinite.(gn_meaninner)

    any(mask_a) || error("No usable entries for panel (a) after removing λ = 1.0.")
    any(mask_b) || error("No usable entries for panel (b) after removing λ = 1.0.")
    any(mask_c) || error("No usable entries for panel (c).")
    (any(mask_gni) || any(mask_gnm)) || error("No usable entries for panel (d).")

    # Shared color range for panels (a) and (b), excluding λ = 1.0
    mask_ratio = isfinite.(mean_ratio) .& isfinite.(λs) .& (λs .!= 1.0)
    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    ratio_min = minimum(mean_ratio[mask_ratio])
    ratio_max = maximum(mean_ratio[mask_ratio])

    fig = Figure(
        size = (1550, 320),
        tellwidth = false,
        tellheight = false,
        figure_padding = (6, 6, 6, 6),
    )

    xticks_no1 = (0.0:0.2:0.8, [@sprintf("%.1f", x) for x in 0.0:0.2:0.8])
    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

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

    axc = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\mathrm{success\ rate}"
    )

    scatterlines!(axc,
        λs[mask_c], succ_rate[mask_c];
        markersize = 8
    )

    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\langle \Vert \nabla \Vert \rangle"
    )

    plots = Any[]
    labels = Any[]

    if any(mask_gni)
        p1 = scatterlines!(axd,
            λs[mask_gni], gn_init[mask_gni];
            markersize = 8,
            marker = :circle
        )
        push!(plots, p1)
        push!(labels, L"\mathrm{init}")
    end

    if any(mask_gnm)
        p2 = scatterlines!(axd,
            λs[mask_gnm], gn_meaninner[mask_gnm];
            markersize = 8,
            marker = :utriangle
        )
        push!(plots, p2)
        push!(labels, L"\mathrm{mean\ inner}")
    end

    if !isempty(plots)
        axislegend(axd, plots, labels;
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
    conv_tag = replace(@sprintf("%.3g", angle_conv), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_abcd_pm1weighted_N$(N)_k$(k)_seed$(graph_seed)_ninit$(n_inits)_abin$(abin_tag)_conv$(conv_tag)_latest_panelmasks.png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()