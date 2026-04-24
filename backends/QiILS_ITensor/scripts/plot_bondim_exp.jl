using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using Statistics
using Printf
using CairoMakie

# -------------------- paths --------------------

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const RESULTS_ROOT = joinpath(PROJECT_ROOT, "results")

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath(RESULTS_ROOT, "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

function runs_dir_lambda1(base_dir; sweeps_per_attempt, percentage, maxdim, sample_mode)
    tag = "lambda1_sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode)"
    return joinpath(base_dir, "runs", tag)
end

# -------------------- user params --------------------

N_values = [30, 40, 50]
k        = 3
seed     = 1
weighted = true

bond_dims = [1, 2, 4, 8, 16, 32, 64, 128]

# Parameters used in the sweep
sweeps_per_attempt = 80
percentage = 0.3
sample_mode = :entangled

# -------------------- load data --------------------

println("="^80)
println("BOND DIMENSION SWEEP ANALYSIS (Multiple N)")
println("="^80)
println()

# Store data for each N
all_data = Dict{Int, Dict}()

for N in N_values
    println("\n--- Loading data for N=$N ---")
    
    base = instance_dir(N, k, seed, weighted)
    
    chi_values = Int[]
    mean_1mr = Float64[]
    std_1mr = Float64[]
    sem_1mr = Float64[]
    mean_cut = Float64[]
    std_cut = Float64[]
    n_trials = Int[]
    
    for maxdim in bond_dims
        rdir = runs_dir_lambda1(base;
            sweeps_per_attempt=sweeps_per_attempt,
            percentage=percentage,
            maxdim=maxdim,
            sample_mode=sample_mode
        )
        
        if !isdir(rdir)
            @warn "Directory not found for N=$N, χ=$maxdim"
            continue
        end
        
        one_minus_rs = Float64[]
        cuts = Float64[]
        
        files = sort(filter(f -> endswith(f, ".json") && occursin("run_s", f), readdir(rdir)))
        
        for file in files
            filepath = joinpath(rdir, file)
            data = JSON.parsefile(filepath)
            push!(one_minus_rs, data["one_minus_r"])
            push!(cuts, data["best_cut"])
        end
        
        if isempty(one_minus_rs)
            continue
        end
        
        push!(chi_values, maxdim)
        push!(mean_1mr, mean(one_minus_rs))
        push!(std_1mr, std(one_minus_rs))
        push!(sem_1mr, std(one_minus_rs) / sqrt(length(one_minus_rs)))
        push!(mean_cut, mean(cuts))
        push!(std_cut, std(cuts))
        push!(n_trials, length(one_minus_rs))
        
        @printf("  χ=%3d: n=%2d  mean(1-r)=%.6e\n", maxdim, length(one_minus_rs), mean(one_minus_rs))
    end
    
    all_data[N] = Dict(
        "chi_values" => chi_values,
        "mean_1mr" => mean_1mr,
        "sem_1mr" => sem_1mr,
        "n_trials" => n_trials
    )
end

println()

# -------------------- plotting --------------------
let
    println("Creating combined plot...")
    
    # Update theme for LaTeX fonts
    update_theme!(
        fontsize = 14,
        fonts = (
            regular = "CMU Serif",
            bold = "CMU Serif Bold",
        )
    )
    
    fig = Figure(size=(400, 300))
    
    ax = Axis(
        fig[1, 1],
        xlabel = L"\chi",
        ylabel = L"1 - r",
      #  title = L"\text{DMRG Performance vs Bond Dimension} (\lambda=1.0)",
        yscale = log10,
        xscale = log10,
         xgridvisible = false,  # ← Remove vertical grid lines
    ygridvisible = false,  # ← Remove horizontal grid lines
        xticks = ([1, 2, 4, 8, 16, 32, 64, 128], 
              [L"2^0", L"2^1", L"2^2", L"2^3", L"2^4", L"2^5", L"2^6", L"2^7"]),
    )
    
    # Different markers for each N
    markers = [:circle, :utriangle, :dtriangle]
    
    for (idx, N) in enumerate(N_values)
        if !haskey(all_data, N) || isempty(all_data[N]["chi_values"])
            @warn "No data for N=$N, skipping"
            continue
        end
        
        chi_vals = all_data[N]["chi_values"]
        mean_vals = all_data[N]["mean_1mr"]
        sem_vals = all_data[N]["sem_1mr"]
        
        # Plot with automatic color cycling
        scatter!(ax, chi_vals, mean_vals, 
            markersize=10,
            marker=markers[idx],
            label=L"n=%$(N)")
        
        errorbars!(ax, chi_vals, mean_vals, sem_vals,
            whiskerwidth=8,
            linewidth=1.5)
        
        lines!(ax, chi_vals, mean_vals, 
            linewidth=1.5)
    end
    
    axislegend(ax, position=:lb)
    
    display(fig)
    
    # Save plot to results root
    outfile = joinpath(RESULTS_ROOT, "bond_dim_sweep_comparison.png")
    save(outfile, fig, px_per_unit=2)
    println("\nPlot saved to: $outfile")
    
    # Reset theme
    set_theme!()
end
println("\nDone!")