############################ RUN QiILS_ITensor (λ sweep, save per-run) ############################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using JSON
using Printf
using Graphs
using Random
using QiILS_ITensor   # your package
using ITensorMPS

########################## Helpers for paths #############################

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const RESULTS_ROOT = joinpath(PROJECT_ROOT, "results")

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath(RESULTS_ROOT, "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

"Folder that stores per-run JSONs for a given experimental setting."
function runs_dir_lambda_sweep(base_dir; L, sweeps_per_attempt, percentage, maxdim, sample_mode, track, early_stop_at_one, tol)
    tag = "lambdasweep_L$(L)_sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode)" *
          "_track$(track)_stop$(early_stop_at_one)_tol$(tol)"
    return joinpath(base_dir, "runs", tag)
end

"Filename for a single run."
run_filename(run_id::Int) = @sprintf("run_s%04d.json", run_id)

########################## One λ-sweep run (best-so-far up to λ) #############################

function run_lambda_sweep_continuation(wg, optimal_cut;
    L::Int,
    sweeps_per_attempt::Int,
    maxdim::Int,
    percentage::Float64,
    sample_mode::Symbol,
    verbose::Bool=false,
    track::Symbol = :best,              # :best | :current | :both
    early_stop_at_one::Bool = false,
    tol::Float64 = 1e-12,
    print_dash_after_one::Bool = true,  # cosmetic
)
    N = nv(wg)
    hilbert = siteinds("Qubit", N)

    λs = collect(range(0.0, 1.0; length=L))

    cuts_best    = Vector{Float64}(undef, L)
    cuts_current = (track == :current || track == :both) ? Vector{Float64}(undef, L) : nothing

    psi = nothing
    best_spins_at_λ1 = nothing

    best_cut_so_far = -Inf
    hit_one = false

    for (i, λ) in enumerate(λs)
        if early_stop_at_one && hit_one
            cuts_best[i] = best_cut_so_far
            if cuts_current !== nothing
                cuts_current[i] = NaN
            end

            if verbose
                if print_dash_after_one
                    @printf("λ = %.4f (%d/%d) | r(current) =  -      | r(best so far) = 1.000000\n", λ, i, L)
                else
                    @printf("λ = %.4f (%d/%d) | r(best so far) = 1.000000\n", λ, i, L)
                end
            end
            continue
        end

        best_history, cut_history, best_spins, psi, hilbert = qiils_itensor_solver(
            wg;
            lambda_sweep       = λ,
            attempts           = 1,
            sweeps_per_attempt = sweeps_per_attempt,
            maxdim             = maxdim,
            percentage         = percentage,
            sample_mode        = sample_mode,
            init_psi           = psi,
            return_psi         = true,
            hilbert            = hilbert,
            weighted           = true,  # ← ADD THIS (assuming weighted graph)
        )

        current_cut = cut_history[end]  # ← FIX: was hist[end]
        best_cut_so_far = max(best_cut_so_far, current_cut)

        cuts_best[i] = best_cut_so_far
        if cuts_current !== nothing
            cuts_current[i] = current_cut
        end

        r_current = current_cut / optimal_cut
        r_best    = best_cut_so_far / optimal_cut

        if verbose
            @printf("λ = %.4f (%d/%d) | r(current) = %.6f | r(best so far) = %.6f\n",
                    λ, i, L, r_current, r_best)
        end

        if r_best >= 1.0 - tol
            hit_one = true
        end

        if i == L
            best_spins_at_λ1 = best_spins  # ← FIX: was spins
        end
    end

    ratios_best = cuts_best ./ optimal_cut
    one_minus_r_best = 1 .- ratios_best

    if track == :best
        return λs, cuts_best, ratios_best, one_minus_r_best, best_spins_at_λ1
    elseif track == :current
        ratios_cur = cuts_current ./ optimal_cut
        one_minus_r_cur = 1 .- ratios_cur
        return λs, cuts_current, ratios_cur, one_minus_r_cur, best_spins_at_λ1
    elseif track == :both
        ratios_cur = cuts_current ./ optimal_cut
        one_minus_r_cur = 1 .- ratios_cur
        return (λs,
                cuts_best, ratios_best, one_minus_r_best,
                cuts_current, ratios_cur, one_minus_r_cur,
                best_spins_at_λ1)
    else
        error("track must be :best, :current, or :both")
    end
end

############################ Main ########################################

println("====================================================")
println("     QiILS_ITensor Runner (save per-run sweeps)     ")
println("====================================================")

# -------------------- user params --------------------
N        = 70
k        = 3
seed     = 1

weighted = true
maxdim   = 1

L = 7

# solver params
sweeps_per_attempt = 80
percentage = 0.3
sample_mode = :entangled  # ← CHANGED from :local

# sweep behavior
S_new = 20  # ← REDUCED for testing (was 200)
early_stop_at_one = true
track = :best
tol = 1e-12

# -------------------- graph + optimal --------------------
GRAPH_BASE = "/Users/guillermo.preisser/Projects/QiILS_ITensor/graphs"

println("▶ Loading graph N=$N, k=$k, seed=$seed (weighted=$weighted)")
wg, graphfile = create_and_save_graph_QiILS(
    N, k, seed;
    weighted = weighted,
    base_path = GRAPH_BASE
)
println("Graph saved at: $graphfile")

solution_path = solution_file_path(N, k, seed; weighted=weighted)
println("Optimal JSON: ", solution_path, " | exists? ", isfile(solution_path))

optimal_cut = load_optimal_cut(solution_path)
println("Loaded optimal_cut = ", optimal_cut)
optimal_cut === nothing && error("Need optimal_cut. Run the Python solver first.")

# -------------------- per-run folder --------------------
base = instance_dir(N, k, seed, weighted)
rdir = runs_dir_lambda_sweep(base;  # ← FIX: renamed function
    L=L,
    sweeps_per_attempt=sweeps_per_attempt,
    percentage=percentage,
    maxdim=maxdim,
    sample_mode=sample_mode,
    track=track,
    early_stop_at_one=early_stop_at_one,
    tol=tol
)
mkpath(rdir)
println("Saving runs to: $rdir")

# determine next run_id (append mode)
existing = filter(f -> endswith(f, ".json") && occursin("run_s", f), readdir(rdir))
existing_ids = Int[]
for f in existing
    m = match(r"run_s(\d+)\.json", f)
    m === nothing && continue
    push!(existing_ids, parse(Int, m.captures[1]))
end
next_id = isempty(existing_ids) ? 1 : maximum(existing_ids) + 1
println("Next run id = $next_id (found $(length(existing_ids)) existing runs)")

# -------------------- run + save --------------------
for t in 0:(S_new-1)
    run_id = next_id + t
    outpath = joinpath(rdir, run_filename(run_id))
    isfile(outpath) && error("Refusing to overwrite existing run file: $outpath")

    Random.seed!(10_000 + run_id)

    verbose = (t == 0)  # ← FIX: verbose only for first iteration
    if verbose
        println("\n--- verbose for first new run (run_id=$run_id) ---")
    end

    λs, cuts, ratios, one_minus_r, best_spins_at_λ1 = run_lambda_sweep_continuation(
        wg, optimal_cut;
        L=L,
        sweeps_per_attempt=sweeps_per_attempt,
        maxdim=maxdim,
        percentage=percentage,
        sample_mode=sample_mode,
        verbose=verbose,
        track=track,
        early_stop_at_one=early_stop_at_one,
        tol=tol
    )

    save_data = Dict(
        "experiment" => "lambda_sweep_continuation_one_attempt_per_lambda",
        "run_id" => run_id,

        "N" => N, "k" => k, "seed" => seed,
        "weighted" => weighted,
        "graphfile" => graphfile,

        "L" => L,
        "lambdas" => λs,

        "sweeps_per_attempt" => sweeps_per_attempt,
        "percentage" => percentage,
        "maxdim" => maxdim,
        "sample_mode" => String(sample_mode),

        "track" => String(track),
        "early_stop_at_one" => early_stop_at_one,
        "tol" => tol,

        "optimal_cut" => optimal_cut,

        "cuts_saved" => cuts,
        "ratios_saved" => ratios,
        "one_minus_r_saved" => one_minus_r,

        "final_best_cut" => cuts[end],
        "final_best_ratio" => ratios[end],
        "best_spins_at_lambda1" => best_spins_at_λ1,  # ← ADD THIS
    )

    open(outpath, "w") do io
        JSON.print(io, save_data)
    end

    @printf("✔ saved run %d/%d | final r=%.6f -> %s\n", 
            t+1, S_new, ratios[end], basename(outpath))
end

println("Done.")