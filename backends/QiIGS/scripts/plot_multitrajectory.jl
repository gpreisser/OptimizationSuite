# scripts/plot_neb_trajectories_mean_lambda300.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using LaTeXStrings
using Statistics

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
const NEB_DIR = joinpath(RESULTS_DIR, "neb_analysis")
mkpath(PLOTS_DIR)

function load_profile(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    prof = data["profiles"]
    s = Float64.(prof["s"])
    est = Float64.(prof["continuous_energy_straight_rel"])
    erl = Float64.(prof["continuous_energy_relaxed_rel"])
    return s, est, erl
end

function stack_profiles(paths::Vector{String})
    s_ref = nothing
    straight_curves = Vector{Vector{Float64}}()
    relaxed_curves = Vector{Vector{Float64}}()

    for p in paths
        @assert isfile(p) "Missing JSON file: $p"
        s, est, erl = load_profile(p)

        if s_ref === nothing
            s_ref = s
        else
            @assert length(s) == length(s_ref) "Mismatched profile length in $p"
            @assert all(isapprox.(s, s_ref; atol=0.0, rtol=0.0)) "Mismatched s grid in $p"
        end

        push!(straight_curves, est)
        push!(relaxed_curves, erl)
    end

    straight_mat = hcat(straight_curves...)
    relaxed_mat = hcat(relaxed_curves...)

    straight_mean = vec(mean(straight_mat; dims=2))
    relaxed_mean = vec(mean(relaxed_mat; dims=2))

    return s_ref, straight_curves, relaxed_curves, straight_mean, relaxed_mean
end

function main()
    # ------------------------------------------------------------
    # JSON files for λ = 0.300, graph seed = 1
    # ------------------------------------------------------------
    json_files = [
        joinpath(NEB_DIR, "neb_relaxed_vs_straight_clean_N50_k3_seed1_unweighted_lam0p3000_pair1_2_nimg21.json"),
        joinpath(NEB_DIR, "neb_relaxed_vs_straight_clean_N50_k3_seed1_unweighted_lam0p3000_pair1_3_nimg21.json"),
        joinpath(NEB_DIR, "neb_relaxed_vs_straight_clean_N50_k3_seed1_unweighted_lam0p3000_pair1_4_nimg21.json"),
    ]

    pair_labels = ["1↔2", "1↔3", "1↔4"]

    s, straight_curves, relaxed_curves, straight_mean, relaxed_mean =
        stack_profiles(json_files)

    fig = Figure(size = (900, 620))

    ax = Axis(fig[1, 1];
        xlabel = L"s",
        ylabel = L"E_{\mathrm{cont}}(s) - E_{\min}",
        title = L"\lambda = 0.300,\ \mathrm{graph\ seed}=1",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 18,
        yticklabelsize = 18,
        xlabelsize = 20,
        ylabelsize = 20,
        titlesize = 20,
    )

    # all straight trajectories in light blue
    for (k, y) in enumerate(straight_curves)
        lines!(ax, s, y;
            color = (:blue, 0.20),
            linewidth = 2.5,
            label = (k == 1 ? "straight trajectories" : nothing),
        )
    end

    # all relaxed trajectories in light orange
    for (k, y) in enumerate(relaxed_curves)
        lines!(ax, s, y;
            color = (:orange, 0.25),
            linewidth = 2.5,
            linestyle = :dash,
            label = (k == 1 ? "relaxed trajectories" : nothing),
        )
    end

    # mean straight in standard Julia blue
    lines!(ax, s, straight_mean;
        color = :blue,
        linewidth = 5,
        label = "mean straight",
    )

    # mean relaxed in standard Julia orange
    lines!(ax, s, relaxed_mean;
        color = :orange,
        linewidth = 5,
        linestyle = :dash,
        label = "mean relaxed",
    )

    axislegend(ax, position = :rt, labelsize = 15)

    # small annotation block
    annotation = "pairs: " * join(pair_labels, ", ")
    text!(ax, 0.03, 0.97;
        text = annotation,
        space = :relative,
        align = (:left, :top),
        fontsize = 16,
    )

    Label(
        fig[0, 1],
        "All λ = 0.300 trajectories with mean straight and relaxed profiles",
        fontsize = 22,
    )

    outpath = joinpath(PLOTS_DIR, "neb_trajectories_mean_lambda0p300_seed1.png")
    save(outpath, fig)
    display(fig)

    println("Saved figure: $outpath")
end

main()