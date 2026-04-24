using Revise
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JSON
using Printf
using Random
using QiILS_ITensor
cd(@__DIR__)

# -------------------- helpers --------------------

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

run_filename(run_id::Int) = @sprintf("run_s%04d.json", run_id)

function next_run_id(rdir::AbstractString)
    if !isdir(rdir)
        return 1
    end
    files = readdir(rdir)
    ids = Int[]
    for f in files
        m = match(r"run_s(\d+)\.json", f)
        m === nothing && continue
        push!(ids, parse(Int, m.captures[1]))
    end
    return isempty(ids) ? 1 : maximum(ids) + 1
end

# -------------------- user params --------------------

N        = 50
k        = 3
seed     = 1
weighted = true

# Bond dimension sweep parameters
bond_dims = [1, 2, 4, 8, 16, 32, 64,128]
num_trials = 40  # trials per bond dimension
attempts = 1     # only 1 DMRG attempt per trial

sweeps_per_attempt = 80
percentage = 0.3
sample_mode = :entangled

λ = 1.0

GRAPH_BASE = "/Users/guillermo.preisser/Projects/QiILS_ITensor/graphs"

# -------------------- graph + optimal --------------------

println("="^80)
println("BOND DIMENSION SWEEP EXPERIMENT")
println("="^80)
println("▶ Graph N=$N k=$k seed=$seed weighted=$weighted")

wg, graphfile = create_and_save_graph_QiILS(
    N, k, seed;
    weighted = weighted,
    base_path = GRAPH_BASE
)

solution_path = solution_file_path(N, k, seed; weighted=weighted)
optimal_cut = load_optimal_cut(solution_path)
optimal_cut === nothing && error("Need optimal_cut. Run Python solver first.")

@printf("Graph: %s\n", graphfile)
@printf("Optimal cut = %.6f\n", optimal_cut)
@printf("λ = %.1f\n", λ)
println()

# -------------------- run sweep --------------------

base = instance_dir(N, k, seed, weighted)
total_runs = length(bond_dims) * num_trials

for (bond_idx, maxdim) in enumerate(bond_dims)
    println("="^80)
    @printf("Bond dimension χ = %d\n", maxdim)
    println("="^80)
    
    # Create directory for this bond dimension
    rdir = runs_dir_lambda1(base;
        sweeps_per_attempt=sweeps_per_attempt,
        percentage=percentage,
        maxdim=maxdim,
        sample_mode=sample_mode
    )
    mkpath(rdir)
    
    # Get starting run_id for this bond dimension
    start_id = next_run_id(rdir)
    @printf("Directory: %s\n", rdir)
    @printf("Starting run_id: %d\n\n", start_id)
    
    for trial in 1:num_trials
        current_run = (bond_idx - 1) * num_trials + trial
        run_id = start_id + trial - 1
        
        outpath = joinpath(rdir, run_filename(run_id))
        if isfile(outpath)
            @printf("[%3d/%3d] χ=%2d trial=%2d: SKIPPING (file exists)\n",
                    current_run, total_runs, maxdim, trial)
            continue
        end
        
        # Unique seed for each trial
        Random.seed!(10_000 + run_id)
        
        # Run solver
        best_history, cut_history, best_spins, energy_history = qiils_itensor_solver(
            wg;
            lambda_sweep       = λ,
            attempts           = attempts,
            sweeps_per_attempt = sweeps_per_attempt,
            maxdim             = maxdim,
            percentage         = percentage,
            sample_mode        = sample_mode,
            weighted           = weighted,
            verbose            = false,
        )
        
        best_cut = best_history[end]
        r = best_cut / optimal_cut
        one_minus_r = 1 - r
        
        # Save data (same format as regular runs)
        save_data = Dict(
            "experiment" => "lambda1_only_baseline",
            "run_id" => run_id,
            
            "N" => N,
            "k" => k,
            "seed" => seed,
            "weighted" => weighted,
            "graphfile" => graphfile,
            
            "lambda" => λ,
            
            "attempts" => attempts,
            "sweeps_per_attempt" => sweeps_per_attempt,
            "percentage" => percentage,
            "maxdim" => maxdim,
            "sample_mode" => String(sample_mode),
            
            "optimal_cut" => optimal_cut,
            
            "best_history" => best_history,
            "cut_history" => cut_history,
            "energy_history" => energy_history,
            "best_cut" => best_cut,
            "best_ratio" => r,
            "one_minus_r" => one_minus_r,
            "best_spins" => best_spins,
        )
        
        open(outpath, "w") do io
            JSON.print(io, save_data)
        end
        
        @printf("[%3d/%3d] χ=%2d trial=%2d: cut=%.2f  r=%.6f  1-r=%.3e -> %s\n",
                current_run, total_runs, maxdim, trial, best_cut, r, one_minus_r, 
                basename(outpath))
    end
    
    println()
end

println("="^80)
println("Experiment complete!")

