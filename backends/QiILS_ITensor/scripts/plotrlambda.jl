############################ AGGREGATE + PLOT (1-r best-so-far) for maxdim=1 vs 2 ############################
# This script:
#   • loads ALL per-run JSONs for maxdim=1 and maxdim=2
#   • averages the stored curve (1 - r(best-so-far up to λ))
#   • plots mean with error bars (±1 std) as two lines
#
# Assumes your per-run files were saved by the per-run runner script, with keys:
#   "lambdas", "one_minus_r_saved", and metadata including L, etc.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using Statistics
using Printf
using CairoMakie

########################## Helpers (must match your folder conventions) #############################

# absolute project root (one level above scripts/)
const PROJROOT = normpath(joinpath(@__DIR__, ".."))

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath(PROJROOT, "results", "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

function runs_dir(base_dir; L, sweeps_per_attempt, percentage, maxdim, sample_mode, track, early_stop_at_one, tol)
    runs_root = joinpath(base_dir, "runs")
    isdir(runs_root) || error("No runs/ folder found at: $runs_root")

    dirs = sort(filter(d -> isdir(joinpath(runs_root, d)), readdir(runs_root)))
    isempty(dirs) && error("No run subfolders inside: $runs_root")

    # Core signature that should always exist in your folder names
    sig_core = "sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode)"

    # Prefer the full convention (includes L, track, stop, tol)
    sig_L     = "L$(L)_" * sig_core
    sig_track = "_track$(track)"
    sig_stop  = "_stop$(early_stop_at_one)"
    sig_tol   = "_tol$(tol)"   # may differ in string formatting, so we won't require exact match

    # 1) strict-ish: must match L + core + track + stop, and contain "_tol"
    candidates = filter(d ->
        occursin(sig_L, d) &&
        occursin(sig_track, d) &&
        occursin(sig_stop, d) &&
        occursin("_tol", d),
        dirs
    )

    # 2) looser: match L + core only
    if isempty(candidates)
        candidates = filter(d -> occursin(sig_L, d), dirs)
    end

    # 3) loosest: match core only
    if isempty(candidates)
        candidates = filter(d -> occursin(sig_core, d), dirs)
    end

    if isempty(candidates)
        println("No folder matched. Searched for (in order):")
        println("  1) $sig_L + $sig_track + $sig_stop + _tol*")
        println("  2) $sig_L")
        println("  3) $sig_core")
        println("Available run folders:")
        for d in dirs
            println("  - ", d)
        end
        error("Could not locate a runs folder for maxdim=$maxdim")
    end

    chosen = candidates[1]
    return joinpath(runs_root, chosen)
end

########################## Load all runs in a folder and aggregate #############################

function load_runs_and_aggregate(rdir::AbstractString)
    isdir(rdir) || error("Runs directory not found: $rdir")

    files = sort(filter(f -> endswith(f, ".json") && occursin("run_s", f), readdir(rdir)))
    isempty(files) && error("No run files found in: $rdir")

    # Load first to get λ grid length
    first_run = JSON.parsefile(joinpath(rdir, files[1]))
    λs = Float64.(first_run["lambdas"])
    L = length(λs)

    # Collect all curves into matrix (nruns × L)
    X = Matrix{Float64}(undef, length(files), L)

    run_ids = Int[]
    for (i, f) in enumerate(files)
        d = JSON.parsefile(joinpath(rdir, f))

        # sanity: same λ grid length
        λs_i = Float64.(d["lambdas"])
        length(λs_i) == L || error("Inconsistent L in file: $f")
        # (optional) check actual λ values too
        maximum(abs.(λs_i .- λs)) < 1e-12 || error("Inconsistent lambdas in file: $f")

        v = Float64.(d["one_minus_r_saved"])
        length(v) == L || error("Wrong curve length in file: $f")

        X[i, :] .= v

        # parse run_id if present
        if haskey(d, "run_id")
            push!(run_ids, Int(d["run_id"]))
        end
    end

    meanv = vec(mean(X; dims=1))
    stdv  = vec(std(X;  dims=1))

    return (λs=λs, X=X, mean=meanv, std=stdv, nruns=size(X,1), run_ids=run_ids)
end

########################## USER SETTINGS #############################

N = 30
k = 3
seed = 1
weighted = true

# must match what you used when generating runs
L = 21
sweeps_per_attempt = 80
percentage = 0.3
sample_mode = :local

track = :best
early_stop_at_one = true
tol = 1e-12

# compare these two
maxdims = [1]

########################## Locate run folders + load #############################

base = instance_dir(N, k, seed, weighted)

rdirs = Dict{Int,String}()
for D in maxdims
    rdirs[D] = runs_dir(base;
        L=L,
        sweeps_per_attempt=sweeps_per_attempt,
        percentage=percentage,
        maxdim=D,
        sample_mode=sample_mode,
        track=track,
        early_stop_at_one=early_stop_at_one,
        tol=tol
    )
end

println("====================================================")
println("Aggregate plot: 1 - r(best-so-far) vs λ")
println("N=$N k=$k seed=$seed weighted=$weighted")
println("L=$L sweeps=$sweeps_per_attempt pct=$percentage mode=$sample_mode track=$track stop=$early_stop_at_one tol=$tol")
println("====================================================")

agg = Dict{Int,Any}()

for D in maxdims
    println("\nLoading runs for maxdim=$D")
    println("  dir: ", rdirs[D])

    a = load_runs_and_aggregate(rdirs[D])
    agg[D] = a

    @printf("  nruns = %d\n", a.nruns)
    @printf("  mean (1-r) at λ=1 = %.6f\n", a.mean[end])
    @printf("  std  (1-r) at λ=1 = %.6f\n", a.std[end])
end

# sanity: same λ grid
λs = agg[maxdims[1]].λs
for D in maxdims[2:end]
    maximum(abs.(agg[D].λs .- λs)) < 1e-12 || error("Different λ grid between D=$(maxdims[1]) and D=$D")
end

########################## Plot #############################
let
fig = Figure(size=(900, 520))
ax  = Axis(fig[1,1],
    xlabel = "λ",
    ylabel = "1 - r(best so far up to λ)",
    title  = "QiILS continuation | weighted=$(weighted) | N=$(N), k=$(k), seed=$(seed) | L=$(L)",
    yscale = log10,
)

# For each D: line + error bars (±1 std)
for D in maxdims
    m = agg[D].mean
   # println("m is ",m)
    s = agg[D].std

    lines!(ax, λs, m, label="maxdim=$D (n=$(agg[D].nruns))")
    errorbars!(ax, λs, m, s)   # vertical error bars (±std)
end

axislegend(ax; position=:rt)

display(fig)

# Save to a common aggregate folder (inside results/.../weighted|unweighted/)
outdir = joinpath(base, "aggregate_plots")
mkpath(outdir)

outbase = joinpath(outdir,
    "agg_1mr_L$(L)_sw$(sweeps_per_attempt)_pct$(percentage)_mode$(sample_mode)_track$(track)_stop$(early_stop_at_one)_tol$(tol)"
)

save(outbase * "_D1vsD2.png", fig)
save(outbase * "_D1vsD2.pdf", fig)

println("\nSaved:")
println(outbase * "_D1vsD2.png")
println(outbase * "_D1vsD2.pdf")
println("Done.")
end