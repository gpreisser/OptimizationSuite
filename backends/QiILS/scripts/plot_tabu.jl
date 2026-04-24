# scripts/plot_tabu_candidates_vs_sweeps.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON
using LaTeXStrings

try
    set_theme!(theme_latexfonts())
catch
end

# ---------------------------------------------------------
# Paths / parameters
# ---------------------------------------------------------
const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(
    ROOT,
    "results",
    "Tabu_Gset12_grid_weighted_ntrials100_sweeps1000000",
)

const GSET = 12
const TENURE = 10
const NTRIALS = 100
const NSWEEPS = 1_000_000
const CANDIDATE_SIZES = [8, 16, 32, 64]

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
error_histories = Dict{Int, Vector{Float64}}()

for cs in CANDIDATE_SIZES
    path = joinpath(
        RESULTS_DIR,
        "tenure$(TENURE)_cand$(cs)",
        "tabu_grid_results_G$(GSET)_tenure$(TENURE)_cand$(cs)_sweeps$(NSWEEPS)_ntrials$(NTRIALS).json",
    )

    data = JSON.parsefile(path)
    error_histories[cs] = Float64.(data["avg_error_history"])
end
plots_dir = "/Users/guillermo.preisser/Projects/QiILS/results/plots"
mkpath(plots_dir)

let
    fig = Figure(size = (900, 650))

    ax = Axis(
        fig[1, 1],
        xlabel = "Sweeps",
        ylabel = L"1-r",
        yscale = log10,
        xscale = log10,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
    )

    ylims!(ax, 1e-5, 1)

    for cs in CANDIDATE_SIZES
        err = error_histories[cs]
        sweeps = 1:length(err)

        lines!(
            ax,
            sweeps,
            err;
            linewidth = 3,
            label = "candidate size = $(cs)",
        )
    end

    axislegend(ax, position = :rt)

    save(joinpath(plots_dir, "tabu_candidates_error_vs_sweeps.png"), fig)

    display(fig)
end