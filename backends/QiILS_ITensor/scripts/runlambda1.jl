using Revise
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # Activate the project environment
using JSON
using Printf
using Random
using QiILS_ITensor
cd(@__DIR__)  # sets working dir to the scripts/ folder

# -------------------- helpers --------------------

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const RESULTS_ROOT = joinpath(PROJECT_ROOT, "results")

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath(RESULTS_ROOT, "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

"Folder that stores per-run JSONs for λ=1 baseline runs."
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

N        = 30
k        = 3
seed     = 1
weighted = true

maxdim   = 256
S_new    = 1             # how many NEW runs to add now
attempts = 20            # <-- within a run: how many QiILS attempts (mix cycles)

sweeps_per_attempt = 80
percentage = 0.3
sample_mode = :entangled  # ← CHANGED from :local to :entangled (correct sampling)

λ = 1.0

GRAPH_BASE = "/Users/guillermo.preisser/Projects/QiILS_ITensor/graphs"

# -------------------- graph + optimal --------------------

println("▶ Graph N=$N k=$k seed=$seed weighted=$weighted")
wg, graphfile = create_and_save_graph_QiILS(
    N, k, seed;
    weighted = weighted,
    base_path = GRAPH_BASE
)
println("Graph file: $graphfile")

solution_path = solution_file_path(N, k, seed; weighted=weighted)
optimal_cut = load_optimal_cut(solution_path)
optimal_cut === nothing && error("Need optimal_cut. Run Python solver first.")
@printf("optimal_cut = %.6f\n", optimal_cut)

# -------------------- output dir --------------------

base = instance_dir(N, k, seed, weighted)
rdir = runs_dir_lambda1(base;
    sweeps_per_attempt=sweeps_per_attempt,
    percentage=percentage,
    maxdim=maxdim,
    sample_mode=sample_mode
)
mkpath(rdir)

start_id = next_run_id(rdir)
println("Saving λ=1 runs to: $rdir")
println("Next run id = $start_id")

# -------------------- run + save --------------------

for t in 0:(S_new-1)
    run_id = start_id + t
    outpath = joinpath(rdir, run_filename(run_id))
    isfile(outpath) && error("Refusing to overwrite: $outpath")

    # reproducible but distinct per run
    Random.seed!(10_000 + run_id)

    # run solver (λ=1 only)
    best_history, cut_history, best_spins, energy_history = qiils_itensor_solver(
        wg;
        lambda_sweep       = λ,
        attempts           = attempts,
        sweeps_per_attempt = sweeps_per_attempt,
        maxdim             = maxdim,
        percentage         = percentage,
        sample_mode        = sample_mode,
        weighted           = weighted,
        verbose            = true,
    )

    best_cut = best_history[end]
    r = best_cut / optimal_cut
    one_minus_r = 1 - r

    save_data = Dict(
        "experiment" => "lambda1_only_baseline",
        "run_id" => run_id,

        "N" => N, "k" => k, "seed" => seed,
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
        "cut_history"  => cut_history,
        "energy_history" => energy_history,
        "best_cut" => best_cut,
        "best_ratio" => r,
        "one_minus_r" => one_minus_r,
        "best_spins" => best_spins,
    )

    open(outpath, "w") do io
        JSON.print(io, save_data)
    end

    @printf("✔ saved run %d/%d | r=%.6f | 1-r=%.3e -> %s\n",
            t+1, S_new, r, one_minus_r, outpath)
end

println("Done.")