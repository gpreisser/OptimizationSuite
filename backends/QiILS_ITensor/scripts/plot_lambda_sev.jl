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

function runs_dir_lambda_sweep(base_dir; L, sweeps_per_attempt, percentage, maxdim, sample_mode, track, early_stop_at_one, tol)
    tag = "lambdasweep_L$(L)_sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode)" *
          "_track$(track)_stop$(early_stop_at_one)_tol$(tol)"
    return joinpath(base_dir, "runs", tag)
end

# -------------------- user params --------------------

N_values = [30,50, 70]  # Can add more: [30, 40, 50]
k        = 3
seed     = 1
weighted = true

# Lambda sweep parameters (must match what you ran)
L = 7
sweeps_per_attempt = 80
percentage = 0.3
maxdim = 1
sample_mode = :entangled
track = :best
early_stop_at_one = true
tol = 1e-12

# -------------------- load data --------------------

println("="^80)
println("LAMBDA SWEEP ANALYSIS (Multiple N)")
println("="^80)
println()

# Store data for each N
all_data = Dict{Int, Dict}()

for N in N_values
    println("\n--- Loading data for N=$N ---")
    
    base = instance_dir(N, k, seed, weighted)
    
    rdir = runs_dir_lambda_sweep(base;
        L=L,
        sweeps_per_attempt=sweeps_per_attempt,
        percentage=percentage,
        maxdim=maxdim,
        sample_mode=sample_mode,
        track=track,
        early_stop_at_one=early_stop_at_one,
        tol=tol
    )
    
    if !isdir(rdir)
        @warn "Directory not found for N=$N: $rdir"
        continue
    end
    
    println("Loading from: $rdir")
    
    # Load all run files
    files = sort(filter(f -> endswith(f, ".json") && occursin("run_s", f), readdir(rdir)))
    
    if isempty(files)
        @warn "No run files found for N=$N"
        continue
    end
    
    # Collect 1-r curves from all runs
    all_one_minus_r = []
    lambdas = nothing
    
    for file in files
    filepath = joinpath(rdir, file)
    data = JSON.parsefile(filepath)
    
    if lambdas === nothing
        lambdas = Float64.(data["lambdas"])  # ← ADD Float64.()
    end
    
    push!(all_one_minus_r, Float64.(data["one_minus_r_saved"]))  # ← ADD Float64.()
end
    
    n_runs = length(all_one_minus_r)
    n_lambdas = length(lambdas)
    
    @printf("  Found %d runs with %d lambda points\n", n_runs, n_lambdas)
    
    # Compute mean and SEM at each lambda
    mean_1mr = zeros(n_lambdas)
    sem_1mr = zeros(n_lambdas)
    
    for i in 1:n_lambdas
        values = [run[i] for run in all_one_minus_r]
        mean_1mr[i] = mean(values)
        sem_1mr[i] = std(values) / sqrt(n_runs)
    end
    
    all_data[N] = Dict(
    "lambdas" => Float64.(lambdas),  # ← ADD Float64.()
    "mean_1mr" => mean_1mr,
    "sem_1mr" => sem_1mr,
    "n_runs" => n_runs
)
    
    @printf("  λ range: [%.3f, %.3f]\n", lambdas[1], lambdas[end])
    @printf("  1-r range: [%.6e, %.6e]\n", minimum(mean_1mr), maximum(mean_1mr))
end

println()

if isempty(all_data)
    error("No data found! Run the lambda sweep experiment first.")
end

# -------------------- plotting --------------------

let
    println("Creating lambda sweep plot...")
    
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
        xlabel = L"\lambda",
        ylabel = L"1 - r",
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false,
    )
    
    # Different markers for each N
    markers = [:circle, :utriangle, :dtriangle]
    
    for (idx, N) in enumerate(N_values)
        if !haskey(all_data, N)
            @warn "No data for N=$N, skipping"
            continue
        end
        
        lambdas = all_data[N]["lambdas"]
        mean_vals = all_data[N]["mean_1mr"]
        sem_vals = all_data[N]["sem_1mr"]
        
        # Plot with automatic color cycling
        scatter!(ax, lambdas, mean_vals, 
            markersize=6,
            marker=markers[idx],
            label=L"n=%$(N)")
        
        # Error band instead of error bars (cleaner for many points)
        band!(ax, lambdas, mean_vals .- sem_vals, mean_vals .+ sem_vals,
            alpha=0.2)
        
        lines!(ax, lambdas, mean_vals, 
            linewidth=2)
    end
    
    axislegend(ax, position=:lb)
    
    display(fig)
    
    # Save plot to results root
    outfile = joinpath(RESULTS_ROOT, "lambda_sweep_1mr_vs_lambda.png")
    save(outfile, fig, px_per_unit=2)
    println("\nPlot saved to: $outfile")
    
    # Reset theme
    set_theme!()
end

println("\nDone!")