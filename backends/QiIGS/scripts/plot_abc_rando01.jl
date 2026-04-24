# scripts/fig_qiigs_abcd.jl
#
# Panels (1×4 horizontal):
# (a) λ vs unique_count_mean (scatter, colored by mean_ratio_mean)
# (b) (unique_count_mean / n_inits) vs λ
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

# returns Float64 or NaN if missing/"none"/nothing
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

    unique_mean = Float64[]
    mean_ratio_mean = Float64[]
    succ_mean = Float64[]

    gn_init_mean = Float64[]
    gn_meaninner_mean = Float64[]

    for entry in rpl
        λ  = getnum(entry, "λ")
        u  = getnum(entry, "unique_count_mean")
        mr = getnum(entry, "mean_ratio_mean")
        sr = getnum(entry, "success_rate_mean")

        gni = getnum(entry, "grad_norm_init_mean")
        gnm = getnum(entry, "grad_norm_meaninner_mean")

        if !isfinite(λ) || !isfinite(u)
            continue
        end

        push!(λs, λ)
        push!(unique_mean, u)
        push!(mean_ratio_mean, mr)
        push!(succ_mean, sr)
        push!(gn_init_mean, gni)
        push!(gn_meaninner_mean, gnm)
    end

    p = sortperm(λs)
    return (
        λs[p],
        unique_mean[p],
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

    λs, uniques, mean_ratio, succ_rate, gn_init, gn_meaninner = load_curves(path)
    isempty(λs) && error("No usable entries found in: $path")

    fig = Figure(size=(1400, 320),
        tellwidth=false,
        tellheight=false,
        figure_padding=(6, 6, 6, 6)
    )

    # ------------------------------------------------------------
    # (a) λ vs ⟨N_unique⟩ — scatter colored by ⟨r⟩
    # NOTE: NOT normalized here (you only wanted (b) normalized)
    # ------------------------------------------------------------
    suba = GridLayout()
    fig[1, 1] = suba

    axa = styled_axis(suba[1, 1];
        xlabel = L"\lambda",
        ylabel = L"\langle N_{\mathrm{unique}}\rangle",
        xλ = true
    )

    mask_a = isfinite.(λs) .& isfinite.(uniques) .& isfinite.(mean_ratio)

    hm = scatter!(axa, λs[mask_a], uniques[mask_a]/n_inits;
        color      = mean_ratio[mask_a],   # color by ⟨r⟩
        colormap   = :viridis,
        colorrange = (0.0, 1.0),           # assuming r ∈ [0,1]
        markersize = 8,
    )

    Colorbar(suba[1, 2], hm;
        label = L"\langle r\rangle",
        width = 12,
        ticklabelsize = 13,
        labelsize = 14,
        ticklabelspace = 20,
    )

    colgap!(suba, 1, 22)
    colsize!(suba, 2, Auto(26))

    # ------------------------------------------------------------
    # (b) (⟨N_unique⟩ / n_inits) vs λ  (scatter only, no lines)
    # ------------------------------------------------------------
    axb = styled_axis(fig[1, 2];
        xlabel = L"\lambda",
        ylabel = L"\langle N_{\mathrm{unique}}\rangle ",
        xλ = true
    )

    uniques_norm = uniques ./ n_inits
    mask_b = isfinite.(λs) .& isfinite.(uniques_norm)

    scatterlines!(axb, λs[mask_b], uniques_norm[mask_b]; markersize=8)
    #ylims!(axb, 0.0, 1.0)

    # ------------------------------------------------------------
    # (c) success rate vs λ  (scatter only, no lines)
    # ------------------------------------------------------------
    axc = styled_axis(fig[1, 3];
        xlabel = L"\lambda",
        ylabel = L"\mathrm{success\ rate}",
        xλ = true
    )
    mask_c = isfinite.(λs) .& isfinite.(succ_rate)
    scatterlines!(axc, λs[mask_c], succ_rate[mask_c]; markersize=8)

    # ------------------------------------------------------------
    # (d) grad norms vs λ  (scatter only, no lines)
    # ------------------------------------------------------------
    axd = styled_axis(fig[1, 4];
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
        position = :rt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 14
    )

    resize_to_layout!(fig)

    # panel labels
    text!(fig.scene, "(a)"; position=(0.00, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(b)"; position=(0.25, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(c)"; position=(0.50, 0.98), space=:relative, fontsize=18, align=(:left, :top))
    text!(fig.scene, "(d)"; position=(0.75, 0.98), space=:relative, fontsize=18, align=(:left, :top))

    out_png = joinpath(PLOTS_DIR,
        "qiigs_abcd_N$(N)_k$(k)_ninit$(n_inits)_$(weighted ? "weighted" : "unweighted").png"
    )
    save(out_png, fig)
    @info "Saved" out_png

    display(fig)
    return nothing
end

main()