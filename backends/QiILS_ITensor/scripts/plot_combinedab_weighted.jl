# scripts/fig_lambda_sweep_ab_weighted.jl
#
# Same plot as before, but:
#   - ONLY uses WEIGHTED result files
#   - χ=1 no longer needs any merge logic (you now have one λ-grid 0.0:0.05:1.0)
#
# Saves: scripts/lambda_sweep_ab_weighted.png

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON
using Glob
using LaTeXStrings
using Printf

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

# ============================================================
# ----------------- FILE DISCOVERY (WEIGHTED ONLY) -----------
# ============================================================

# Robust-ish check: matches ".../weighted/..." or "..._weighted..." but NOT "...unweighted..."
function is_weighted_path(path::AbstractString)
    s = lowercase(path)
    occursin("unweighted", s) && return false
    return occursin("/weighted/", s) || occursin("_weighted", s) || occursin("weighted_", s)
end

function all_jsons_for_chi(χ::Int)
    pat = "**/*_maxdim$(χ)_nsw*.json"
    files = Glob.glob(pat, RESULTS_DIR)
    filter(is_weighted_path, files)
end

function newest_final_or_checkpoint_for_chi(χ::Int)
    files = all_jsons_for_chi(χ)
    isempty(files) && return nothing

    finals = [f for f in files if !occursin("checkpoint", lowercase(f))]
    ckpts  = [f for f in files if  occursin("checkpoint", lowercase(f))]

    if !isempty(finals)
        mt = map(f -> stat(f).mtime, finals)
        return (finals[argmax(mt)], :final)
    elseif !isempty(ckpts)
        mt = map(f -> stat(f).mtime, ckpts)
        return (ckpts[argmax(mt)], :checkpoint)
    else
        return nothing
    end
end

# ============================================================
# ----------------- PANEL A LOADING --------------------------
# ============================================================

function load_a(json_path::AbstractString, kind::Symbol)
    data = JSON.parsefile(json_path)

    if kind == :final
        rpl = data["results_per_lambda"]
        λs, y = Float64[], Float64[]
        for entry in rpl
            r = entry["approx_ratio_mean"]
            (r == "none" || r === nothing) && continue
            push!(λs, Float64(entry["λ_sweep"]))
            push!(y,  Float64(r))
        end
        p = sortperm(λs)
        return λs[p], y[p]
    else
        λs = Float64.(data["λs"])
        sum_ratio = Float64.(data["sum_ratio"])
        n_ratio   = Int.(data["n_ratio"])

        λ_keep, y_keep = Float64[], Float64[]
        for i in eachindex(λs)
            if n_ratio[i] > 0
                push!(λ_keep, λs[i])
                push!(y_keep, sum_ratio[i] / n_ratio[i])
            end
        end
        p = sortperm(λ_keep)
        return λ_keep[p], y_keep[p]
    end
end

# ============================================================
# ----------------- PANEL B LOADING --------------------------
# ============================================================

function load_b_for_chi(χ::Int)
    pick = newest_final_or_checkpoint_for_chi(χ)
    pick === nothing && return nothing
    path, kind = pick
    data = JSON.parsefile(path)

    if kind == :final
        rpl = data["results_per_lambda"]
        λs, y = Float64[], Float64[]
        for entry in rpl
            th = entry["devTheta_abs_mean"]
            (th === nothing || th == "none") && continue
            push!(λs, Float64(entry["λ_sweep"]))
            push!(y,  Float64(th))
        end
        p = sortperm(λs)
        return λs[p], y[p]
    else
        λs = Float64.(data["λs"])
        sum_devTh = Float64.(data["sum_devTh"])
        n_graphs  = Int(data["n_graphs"])
        y = sum_devTh ./ n_graphs
        p = sortperm(λs)
        return λs[p], y[p]
    end
end

# ============================================================
# ----------------- HELPERS ---------------------------------
# ============================================================

marker_for_chi(χ::Int) =
    χ == 1 ? :circle :
    χ == 2 ? :utriangle :
    χ == 4 ? :dtriangle :
             :diamond

# ============================================================
# ----------------- DRAW PANEL A -----------------------------
# ============================================================

function draw_panel_a!(fig, row, col, chis)
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

    for χ in chis
        pick = newest_final_or_checkpoint_for_chi(χ)
        pick === nothing && continue
        path, kind = pick

        λs, y = load_a(path, kind)
        isempty(λs) && continue

        color = χ == 4 ? :green :
                χ == 8 ? :purple :
                nothing

        if isnothing(color)
            p = scatterlines!(ax, λs, y; marker=marker_for_chi(χ), markersize=9)
        else
            p = scatterlines!(ax, λs, y; color=color, marker=marker_for_chi(χ), markersize=9)
        end

        push!(plots, p)
        push!(labels, L"\chi=%$(χ)")
    end

    axislegend(ax, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 16
    )

    return ax
end

# ============================================================
# ----------------- DRAW PANEL B -----------------------------
# ============================================================

function draw_panel_b!(fig, row, col, chis)
    ax = Axis(fig[row, col];
        xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
        xlabel = L"\lambda",
        ylabel = L"\frac{4}{\pi}\,\langle | \frac{1}{2}\arccos(|\langle Z_i\rangle|) - \frac{\pi}{4} | \rangle_i",
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

    for χ in chis
        result = load_b_for_chi(χ)
        result === nothing && continue
        λs, y = result
        isempty(λs) && continue

        color = χ == 4 ? :green :
                χ == 8 ? :purple :
                nothing

        if isnothing(color)
            p = scatterlines!(ax, λs, y/(pi/4); marker=marker_for_chi(χ), markersize=9)
        else
            p = scatterlines!(ax, λs, y/(pi/4); color=color, marker=marker_for_chi(χ), markersize=9)
        end

        push!(plots, p)
        push!(labels, L"\chi=%$(χ)")
    end

    axislegend(ax, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 16
    )

    return ax
end

# ============================================================
# ----------------- MAIN FIGURE ------------------------------
# ============================================================

let
    fig = Figure(size=(850, 300),
                 tellwidth=false,
                 tellheight=false,
                 figure_padding=(6, 6, 6, 6))

    chis = [1, 2, 4, 8]

    draw_panel_a!(fig, 1, 1, chis)
    draw_panel_b!(fig, 1, 2, chis)

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

    save(joinpath(@__DIR__, "lambda_sweep_ab_weighted.png"), fig)

    fig
end