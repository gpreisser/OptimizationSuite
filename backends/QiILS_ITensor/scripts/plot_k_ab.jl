# scripts/plot_k_ab.jl
#
# Compare k = 2,3,4,5 at fixed χ (= maxdim), using newest FINAL aggregated JSON per k.
# Panel (a): approx_ratio_mean vs λ
# Panel (b): devTheta_abs_mean/(π/4) vs λ
#
# Robust file discovery: uses walkdir (no Glob, no absolute glob issues).
# Saves under ROOT/results/plots (never pwd-dependent)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

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
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

# ------------------------------------------------------------
# File discovery (newest final aggregated file for given k)
# ------------------------------------------------------------

function newest_final_for_k(; N::Int=50, k::Int, weighted::Bool=false, maxdim::Int=1)
    weight_tag = weighted ? "weighted" : "unweighted"
    dir_hint = "itensor_random_regular_N$(N)_k$(k)_graphs"
    name_hint = "qiils_itensor_lambda_sweep_energy2_cut_hat_ratio_"
    md_hint = "_maxdim$(maxdim)_nsw"

    best_path = nothing
    best_mtime = -1.0

    for (root, _, files) in walkdir(RESULTS_DIR)
        occursin(dir_hint, root) || continue
        occursin(weight_tag, root) || continue

        for f in files
            endswith(f, ".json") || continue
            occursin("checkpoint", lowercase(f)) && continue
            occursin(name_hint, f) || continue
            occursin(md_hint, f) || continue

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

function load_ratio_curve(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    ys = Float64[]
    for entry in rpl
        r = entry["approx_ratio_mean"]
        (r == "none" || r === nothing) && continue
        push!(λs, Float64(entry["λ_sweep"]))
        push!(ys, Float64(r))
    end
    p = sortperm(λs)
    return λs[p], ys[p]
end

function load_devtheta_curve(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    ys = Float64[]
    for entry in rpl
        th = entry["devTheta_abs_mean"]
        (th == "none" || th === nothing) && continue
        push!(λs, Float64(entry["λ_sweep"]))
        push!(ys, Float64(th))
    end
    p = sortperm(λs)
    return λs[p], ys[p]
end

# ============================================================
# ----------------- HELPERS ---------------------------------
# ============================================================

marker_for_k(k::Int) =
    k == 2 ? :circle :
    k == 3 ? :utriangle :
    k == 4 ? :dtriangle :
             :diamond

color_for_k(k::Int) =
    k == 4 ? :green :
    k == 5 ? :purple :
             nothing

# ============================================================
# ----------------- DRAW PANEL A -----------------------------
# ============================================================

function draw_panel_a!(fig, row, col; ks, N=50, weighted=false, maxdim=1)
    ax = Axis(fig[row, col];
        xticks = (0.0:0.2:1.0, string.(0.0:0.2:1.0)),
        xlabel = L"\lambda",
        ylabel = L"r",
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 18,
        ylabelsize = 18,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    plots, labels = Plot[], Any[]

    for k in ks
        path = newest_final_for_k(; N=N, k=k, weighted=weighted, maxdim=maxdim)
        if path === nothing
            @warn "No FINAL json found for" k N weighted maxdim
            continue
        end

        λs, y = load_ratio_curve(path)
        isempty(λs) && continue

        color = color_for_k(k)

        if isnothing(color)
            p = scatterlines!(ax, λs, y;
                marker = marker_for_k(k),
                markersize = 9)
        else
            p = scatterlines!(ax, λs, y;
                color = color,
                marker = marker_for_k(k),
                markersize = 9)
        end

        push!(plots, p)
        push!(labels, L"k=%$(k)")
    end

    axislegend(ax, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 16)

    return ax
end

# ============================================================
# ----------------- DRAW PANEL B -----------------------------
# ============================================================

function draw_panel_b!(fig, row, col; ks, N=50, weighted=false, maxdim=1)
    ax = Axis(fig[row, col];
        xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
        xlabel = L"\lambda",
        ylabel = L"\frac{4}{\pi}\,\left\langle \left|\frac{1}{2}\arccos(|\langle Z_i\rangle|)-\frac{\pi}{4}\right|\right\rangle_i",
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 18,
        ylabelsize = 18,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    plots, labels = Plot[], Any[]

    for k in ks
        path = newest_final_for_k(; N=N, k=k, weighted=weighted, maxdim=maxdim)
        if path === nothing
            @warn "No FINAL json found for" k N weighted maxdim
            continue
        end

        λs, th = load_devtheta_curve(path)
        isempty(λs) && continue

        y = th ./ (pi/4)

        color = color_for_k(k)

        if isnothing(color)
            p = scatterlines!(ax, λs, y;
                marker = marker_for_k(k),
                markersize = 9)
        else
            p = scatterlines!(ax, λs, y;
                color = color,
                marker = marker_for_k(k),
                markersize = 9)
        end

        push!(plots, p)
        push!(labels, L"k=%$(k)")
    end

    axislegend(ax, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 16)

    return ax
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    N = 50
    ks = [2, 3, 4, 5]
    weighted = false
    maxdim = 1

    fig = Figure(size=(850, 300),
        tellwidth=false,
        tellheight=false,
        figure_padding=(6, 6, 6, 6)
    )

    draw_panel_a!(fig, 1, 1; ks=ks, N=N, weighted=weighted, maxdim=maxdim)
    draw_panel_b!(fig, 1, 2; ks=ks, N=N, weighted=weighted, maxdim=maxdim)

    resize_to_layout!(fig)

    text!(fig.scene, "(a)";
        position = (0.0, 0.98),
        space = :relative,
        fontsize = 18,
        align = (:left, :top)
    )
    text!(fig.scene, "(b)";
        position = (0.492, 0.98),
        space = :relative,
        fontsize = 18,
        align = (:left, :top)
    )

    out_png = joinpath(PLOTS_DIR,
        "lambda_sweep_k_compare_ab_N$(N)_chi$(maxdim)_$(weighted ? "weighted" : "unweighted").png"
    )
    save(out_png, fig)
    @info "Saved" out_png

    display(fig)
    return nothing
end

main()