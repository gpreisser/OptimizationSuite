# scripts/plot_tabu_standard.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON
using LaTeXStrings
using Statistics

try
    set_theme!(theme_latexfonts())
catch
end

let
    # ---------------------------------------------------------
    # Paths / parameters
    # ---------------------------------------------------------
    root_dir = normpath(joinpath(@__DIR__, ".."))

    tabu_results_dir = joinpath(
        root_dir,
        "results",
        "StandardTabu_Gset12_tenures_weighted_ntrials100_sweeps1000000",
    )

    plots_dir = "/Users/guillermo.preisser/Projects/QiILS/results/plots"
    mkpath(plots_dir)

    gset = 12
    ntrials = 100
    nsweeps = 1_000_000
    tenures = [100,150,200,250]

    # ---------------------------------------------------------
    # Load Tabu data robustly
    # ---------------------------------------------------------
    tabu_error_histories = Dict{Int, Vector{Float64}}()
    tabu_saved_indices = Dict{Int, Vector{Int}}()
    loaded_tenures = Int[]

    for tenure in tenures
        path = joinpath(
            tabu_results_dir,
            "tenure$(tenure)",
            "standard_tabu_results_G$(gset)_tenure$(tenure)_sweeps$(nsweeps)_ntrials$(ntrials).json",
        )

        if !isfile(path)
            @warn "Missing file, skipping tenure" tenure path
            continue
        end

        if filesize(path) == 0
            @warn "Empty file, skipping tenure" tenure path
            continue
        end

        try
            data = JSON.parsefile(path)
            tabu_error_histories[tenure] = Float64.(data["avg_error_history"])
            tabu_saved_indices[tenure] = Int.(data["saved_sweep_indices"])
            push!(loaded_tenures, tenure)
        catch err
            @warn "Could not parse JSON, skipping tenure" tenure path exception=(err, catch_backtrace())
        end
    end

    isempty(loaded_tenures) && error("No valid Tabu result files could be loaded.")
    sort!(loaded_tenures)
    println("Loaded Tabu tenures: ", loaded_tenures)

    # ---------------------------------------------------------
    # Load QiILS data
    # ---------------------------------------------------------
    instance = 12
    mixing = 0.2
    nsamples = 100
    optimal_cut = 556.0

    qils_results_dir = "/Users/guillermo.preisser/Projects/code_optimization/results/"

    all_maxcut = Vector{Vector{Float64}}()
    all_sweep = Vector{Vector{Float64}}()

    for i in 1:nsamples
        json_path = qils_results_dir *
            "gset$(instance)/lambda/mix$(mixing)/" *
            "gset$(instance)_mix$(mixing)_lambda_sample$(i).json"

        dat = JSON.parsefile(json_path)

        push!(all_maxcut, Float64.(dat["maxcut"]) ./ optimal_cut)
        push!(all_sweep, Float64.(dat["sweep_list"]))
    end

    mean_maxcut = mean(all_maxcut)
    std_maxcut = std(all_maxcut)
    mean_sweep = mean(all_sweep)

    function logscale_indices(max_val::Int)
        indices = Int[]

        if max_val >= 1
            append!(indices, 1:2:min(10, max_val))
        end
        if max_val >= 20
            append!(indices, 20:20:min(100, max_val))
        end
        if max_val >= 200
            append!(indices, 200:200:min(1000, max_val))
        end
        if max_val >= 2000
            append!(indices, 2000:2000:min(10000, max_val))
        end
        if max_val >= 20000
            append!(indices, 20000:20000:min(100000, max_val))
        end
        if max_val >= 200000
            append!(indices, 200000:200000:min(max_val, 1_000_000))
        end

        sort!(indices)
        unique!(indices)
        return indices
    end

    idx_qiils = logscale_indices(length(mean_sweep))

    qiils_x = mean_sweep[idx_qiils]
    qiils_y = 1 .- mean_maxcut[idx_qiils]
    qiils_yerr = std_maxcut[idx_qiils] ./ sqrt(nsamples)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
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

    for tenure in loaded_tenures
        x = tabu_saved_indices[tenure]
        y = tabu_error_histories[tenure]

        lines!(
            ax,
            x,
            y;
            linewidth = 3,
            label = "Tabu, tenure = $(tenure)",
        )
    end

    errorbars!(
        ax,
        qiils_x,
        qiils_y,
        qiils_yerr;
        whiskerwidth = 6,
    )

    scatterlines!(
        ax,
        qiils_x,
        qiils_y;
        marker = :circle,
        markersize = 9,
        linewidth = 3,
        label = "QiILS",
    )

    axislegend(ax, position = :rt)

    save(joinpath(plots_dir, "standard_tabu_with_qiils_error_vs_sweeps.png"), fig)
    save(joinpath(plots_dir, "standard_tabu_with_qiils_error_vs_sweeps.pdf"), fig)

    display(fig)
end