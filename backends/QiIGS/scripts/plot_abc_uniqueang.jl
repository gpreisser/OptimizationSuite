# scripts/fig_qiigs_abcd.jl
#
# Layout (1×5):
#   (a)  (b)  [colorbar]  (c)  (d)
#
# Panels:
# (a) λ vs (unique_angle_count_mean / n_inits) (scatter, colored by mean_ratio_mean)
# (b) λ vs (unique_count_mean / n_inits)       (scatter, colored by mean_ratio_mean)
# (c) success_rate_mean vs λ
# (d) gn_init_mean & gn_meaninner_mean vs λ
#
# Uses newest FINAL aggregated QiIGS JSON (skips checkpoint files).
# Saves under ROOT/results/plots (ROOT = QiIGS repo root).

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

const ROOT = normpath(joinpath(@__DIR__, ".."))   # QiIGS repo root
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

# ------------------------------------------------------------
# File discovery (newest FINAL aggregated QiIGS file)
# ------------------------------------------------------------

function newest_final_qiigs(; N::Int=50, k::Int=3, weighted::Bool=false, n_inits::Int=1000)
    wtag = weighted ? "weighted" : "unweighted"
    dir_hint = "qiigs_unique_minima_N$(N)_k$(k)_graphs"
    name_hint = "qiigs_unique_ratio_meanbest_succ_grad_"
    ninit_hint = "_ninit$(n_inits)_"

    best_path = nothing
    best_mtime = -1.0

    for (root, _, files) in walkdir(RESULTS_DIR)
        occursin(dir_hint, root) || continue
        occursin(wtag, root) || continue

        for f in files
            endswith(lowercase(f), ".json") || continue
            occursin("checkpoint", lowercase(f)) && continue
            occursin(name_hint, f) || continue
            occursin(ninit_hint, f) || continue

            path = joinpath(root, f)
            mt = stat(path).mtime
            if mt > best_mtime
                best_mtime = mt
                best_path = path
            end
        end
    end

    return best_path
end

# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------

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
    unique_angle_mean = Float64[]
    unique_round_mean = Float64[]
    mean_ratio_mean = Float64[]
    succ_mean = Float64[]
    gn_init_mean = Float64[]
    gn_meaninner_mean = Float64[]

    for entry in rpl
        λ   = getnum(entry, "λ")
        ua  = getnum(entry, "unique_angle_count_mean")
        ur  = getnum(entry, "unique_count_mean")
        mr  = getnum(entry, "mean_ratio_mean")
        sr  = getnum(entry, "success_rate_mean")
        gni = getnum(entry, "grad_norm_init_mean")
        gnm = getnum(entry, "grad_norm_meaninner_mean")

        if !isfinite(λ) || !isfinite(ua) || !isfinite(ur)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_mean, ua)
        push!(unique_round_mean, ur)
        push!(mean_ratio_mean, mr)
        push!(succ_mean, sr)
        push!(gn_init_mean, gni)
        push!(gn_meaninner_mean, gnm)
    end

    p = sortperm(λs)
    return (
        λs[p],
        unique_angle_mean[p],
        unique_round_mean[p],
        mean_ratio_mean[p],
        succ_mean[p],
        gn_init_mean[p],
        gn_meaninner_mean[p],
    )
end

# ------------------------------------------------------------
# Axis style helper
# ------------------------------------------------------------

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
        return Axis(figpos; xticks = xt, common...)
    else
        return Axis(figpos; common...)
    end
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    N = 50
    k = 3
    weighted = false
    n_inits = 1000

    path = newest_final_qiigs(; N=N, k=k, weighted=weighted, n_inits=n_inits)
    path === nothing && error("No FINAL QiIGS aggregated JSON found under $(RESULTS_DIR).")
    @info "Using" path

    λs, uniques_angle, uniques_round, mean_ratio, succ_rate, gn_init, gn_meaninner = load_curves(path)
    isempty(λs) && error("No usable entries found in: $path")

    # color scale adapts to actual ratio range
    mask_r = isfinite.(mean_ratio)
    any(mask_r) || error("No finite mean_ratio values found in: $path")
    ratio_min = minimum(mean_ratio[mask_r])
    ratio_max = maximum(mean_ratio[mask_r])

    fig = Figure(size=(1550, 320),
        tellwidth=false,
        tellheight=false,
        figure_padding=(6, 6, 6, 6)
    )

    # (a) normalized
    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
       ylabel = L"\hat{N}_{\mathrm{unique}}^{\mathrm{angle}}",
        xλ = true
    )

    mask_a = isfinite.(λs) .& isfinite.(uniques_angle) .& isfinite.(mean_ratio)

    hm = scatter!(axa, λs[mask_a], uniques_angle[mask_a] ;
        color      = mean_ratio[mask_a],
        colormap   = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8,
    )

    # (b) normalized
    axb = styled_axis(fig[1, 2];
        xlabel = L"\lambda",
        ylabel = L"\hat{N}_{\mathrm{unique}}^{\mathrm{round}}",
        xλ = true
    )

    mask_b = isfinite.(λs) .& isfinite.(uniques_round) .& isfinite.(mean_ratio)

    scatter!(axb, λs[mask_b], uniques_round[mask_b];
        color      = mean_ratio[mask_b],
        colormap   = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8,
    )

    # shared colorbar (col 3)
    Colorbar(fig[1, 3], hm;
        label = L"\langle r\rangle",
        width = 14,
        ticklabelsize = 14,
        labelsize = 15,
        ticklabelspace = 18,
    )
    colsize!(fig.layout, 3, Auto(22))

    # (c)
    axc = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\mathrm{success\ rate}",
        xλ = true
    )
    mask_c = isfinite.(λs) .& isfinite.(succ_rate)
    scatterlines!(axc, λs[mask_c], succ_rate[mask_c]; markersize=8)

    # (d)
    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\langle \Vert \nabla \Vert \rangle",
        xλ = true
    )

    plots = Plot[]
    labels = Any[]

    mask_gni = isfinite.(λs) .& isfinite.(gn_init)
    p1 = scatterlines!(axd, λs[mask_gni], gn_init[mask_gni]; markersize=8, marker=:circle)
    push!(plots, p1); push!(labels, L"\mathrm{init}")

    mask_gnm = isfinite.(λs) .& isfinite.(gn_meaninner)
    p2 = scatterlines!(axd, λs[mask_gnm], gn_meaninner[mask_gnm]; markersize=8, marker=:utriangle)
    push!(plots, p2); push!(labels, L"\mathrm{mean\ inner}")

    axislegend(axd, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 14
    )

    resize_to_layout!(fig)

    # panel labels (approx positions for 5-column layout)
    text!(fig.scene, "(a)"; position=(0.00, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(b)"; position=(0.24, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(c)"; position=(0.52, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(d)"; position=(0.77, 0.98), space=:relative, fontsize=18, align=(:left, :top))

    out_png = joinpath(PLOTS_DIR,
        "qiigs_abcd_N$(N)_k$(k)_ninit$(n_inits)_$(weighted ? "weighted" : "unweighted").png"
    )
    save(out_png, fig)
    @info "Saved" out_png

    display(fig)
    return nothing
end

main()