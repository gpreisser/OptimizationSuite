# scripts/plot_conorgraph.jl

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

        if !isfinite(λ) || !isfinite(ua) || !isfinite(ur)
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

function styled_axis(figpos; xlabel, ylabel, xλ::Bool=false)
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

    if xλ
        xt = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])
        return Axis(figpos; xticks=xt, common...)
    else
        return Axis(figpos; common...)
    end
end

function main()
    N = 12
    k = 3
    graph_seed = 1
    n_inits = 1000
    angle_bin = 0.03
    angle_conv = 1e-6

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "." => "p")

    path = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_pm1weighted_N$(N)_k$(k)_seed$(graph_seed)",
        "qiigs_unique_ratio_meanbest_succ_grad_pm1weighted_seed$(graph_seed)_lam0p000_to_1p000_d0p100_conv$(conv_tag)_abin$(abin_tag)_ngraphs1_ninit$(n_inits)_outer1_inner5000_thr0p999.json"
    )

    isfile(path) || error("File not found: $path")
    @info "Using" path

    λs, uniques_angle, uniques_round, mean_ratio,
    succ_rate, gn_init, gn_meaninner = load_curves(path)

    isempty(λs) && error("No usable entries.")

    mask_r = isfinite.(mean_ratio)
    ratio_min = minimum(mean_ratio[mask_r])
    ratio_max = maximum(mean_ratio[mask_r])

    fig = Figure(size=(1550, 320),
        tellwidth=false,
        tellheight=false,
        figure_padding=(6, 6, 6, 6)
    )

    axa = styled_axis(fig[1, 1];
        xlabel=L"\lambda",
        ylabel=L"N_{\mathrm{unique}}^{\mathrm{angle,raw}}",
        xλ=true
    )

    mask_a = isfinite.(λs) .& isfinite.(uniques_angle)
    hm = scatter!(axa,
        λs[mask_a], uniques_angle[mask_a];
        color=mean_ratio[mask_a],
        colormap=:viridis,
        colorrange=(ratio_min, ratio_max),
        markersize=8
    )

    axb = styled_axis(fig[1, 2];
        xlabel=L"\lambda",
        ylabel=L"N_{\mathrm{unique}}^{\mathrm{round}}",
        xλ=true
    )

    mask_b = isfinite.(λs) .& isfinite.(uniques_round)
    scatter!(axb,
        λs[mask_b], uniques_round[mask_b];
        color=mean_ratio[mask_b],
        colormap=:viridis,
        colorrange=(ratio_min, ratio_max),
        markersize=8
    )

    Colorbar(fig[1, 3], hm;
        label=L"\langle r\rangle",
        width=14,
        ticklabelsize=14,
        labelsize=15
    )
    colsize!(fig.layout, 3, Auto(22))

    axc = styled_axis(fig[1, 4];
        xlabel=L"\lambda",
        ylabel=L"\mathrm{success\ rate}",
        xλ=true
    )

    mask_c = isfinite.(λs) .& isfinite.(succ_rate)
    scatterlines!(axc,
        λs[mask_c], succ_rate[mask_c];
        markersize=8
    )

    axd = styled_axis(fig[1, 5];
        xlabel=L"\lambda",
        ylabel=L"\langle \Vert \nabla \Vert \rangle",
        xλ=true
    )

    plots = Plot[]
    labels = Any[]

    mask_gni = isfinite.(λs) .& isfinite.(gn_init)
    p1 = scatterlines!(axd, λs[mask_gni], gn_init[mask_gni];
        markersize=8, marker=:circle)
    push!(plots, p1); push!(labels, L"\mathrm{init}")

    mask_gnm = isfinite.(λs) .& isfinite.(gn_meaninner)
    p2 = scatterlines!(axd, λs[mask_gnm], gn_meaninner[mask_gnm];
        markersize=8, marker=:utriangle)
    push!(plots, p2); push!(labels, L"\mathrm{mean\ inner}")

    axislegend(axd, plots, labels;
        position=:lt,
        framevisible=false,
        backgroundcolor=:transparent,
        labelsize=14
    )

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.00, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.24, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(c)"; position=(0.52, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(d)"; position=(0.77, 0.92), space=:relative, fontsize=18)

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_abcd_pm1weighted_N$(N)_k$(k)_seed$(graph_seed)_ninit$(n_inits)_abin$(abin_tag)_conv$(conv_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()