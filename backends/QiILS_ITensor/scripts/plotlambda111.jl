############################ PLOT ILS: (1 - r) vs attempts ############################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using Statistics
using Printf
using CairoMakie

function instance_dir(N, k, seed, weighted)
    wtag = weighted ? "weighted" : "unweighted"
    return joinpath("results", "N$(N)", "k$(k)", "seed$(seed)", wtag)
end

# If your λ=1 runs live in a different tag, edit this to match your folder naming.
function runs_dir_any(base_dir)
    runs_root = joinpath(base_dir, "runs")
    isdir(runs_root) || error("No runs/ folder found at: $runs_root")

    dirs = sort(filter(d -> isdir(joinpath(runs_root, d)), readdir(runs_root)))
    isempty(dirs) && error("No run subfolders inside: $runs_root")

    println("Available run folders:")
    for d in dirs
        println("  - ", d)
    end

    # Prefer lambda1 if present, else take the first folder
    idx = findfirst(d -> occursin("lambda1_", d), dirs)
    chosen = isnothing(idx) ? dirs[1] : dirs[idx]

    return joinpath(runs_root, chosen)
end

function main()
    # -------------------- user params --------------------
    N        = 50
    k        = 3
    seed     = 1
    weighted = false

    maxdim   = 1
    sweeps_per_attempt = 80
    percentage = 0.3
    sample_mode = :local

    use_first_K_runs = nothing  # e.g. 30 for debugging
    clip_floor = 0.0            # keeps tiny negatives at 0

    # -------------------- paths --------------------
    base = instance_dir(N, k, seed, weighted)
    function runs_dir_any(base_dir; sweeps_per_attempt, percentage, maxdim, sample_mode)
    runs_root = joinpath(base_dir, "runs")
    isdir(runs_root) || error("No runs/ folder found at: $runs_root")

    dirs = sort(filter(d -> isdir(joinpath(runs_root, d)), readdir(runs_root)))
    isempty(dirs) && error("No run subfolders inside: $runs_root")

    # build a signature that matches your folder naming chunks
    sig = "sw$(sweeps_per_attempt)_pct$(percentage)_maxd$(maxdim)_mode$(sample_mode)"
    # examples it should match:
    #   lambda1_sw80_pct0.3_maxd1_modelocal
    #   L21_sw80_pct0.3_maxd1_modelocal_trackbest_stoptrue_tol1.0e-12

    # prefer folders that match the signature
    matches = filter(d -> occursin(sig, d), dirs)

    if !isempty(matches)
        # If multiple match, prefer lambda1 over others, else take first
        idx = findfirst(d -> startswith(d, "lambda1_"), matches)
        chosen = isnothing(idx) ? matches[1] : matches[idx]
        return joinpath(runs_root, chosen)
    end

    # fallback: print what's available and pick something sensible
    println("No folder matched signature: $sig")
    println("Available run folders:")
    for d in dirs
        println("  - ", d)
    end

    idx = findfirst(d -> startswith(d, "lambda1_"), dirs)
    chosen = isnothing(idx) ? dirs[1] : dirs[idx]
    println("Using fallback: $chosen")

    return joinpath(runs_root, chosen)
end

    println("====================================================")
    println("Plot ILS: mean ± std of (1 - r_best_so_far) vs attempts")
    @printf("N=%d k=%d seed=%d weighted=%s maxdim=%d\n", N, k, seed, string(weighted), maxdim)
    @printf("dir: %s\n", rdir)
    println("====================================================\n")

    isdir(rdir) || error("Runs directory not found: $rdir")

    files = sort(filter(f -> endswith(f, ".json") && occursin("run_s", f), readdir(rdir)))
    isempty(files) && error("No run JSON files found in $rdir")

    if use_first_K_runs !== nothing
        files = files[1:min(use_first_K_runs, length(files))]
    end

    nruns = length(files)
    println("Found nruns = $nruns")

    # -------------------- load first run for sizes --------------------
    first = JSON.parsefile(joinpath(rdir, files[1]))
    optimal_cut = Float64(first["optimal_cut"])

    # We expect ILS runs to store best_history (best-so-far cut per attempt)
    haskey(first, "best_history") || error("First JSON missing key 'best_history'. Keys: $(collect(keys(first)))")

    h0 = Float64.(first["best_history"])
    T  = length(h0)

    println("Attempts per run T = $T")
    @printf("optimal_cut = %.6f\n\n", optimal_cut)

    # matrix of (1-r) curves: T x nruns
    Y = Matrix{Float64}(undef, T, nruns)

    for (j, f) in enumerate(files)
        d = JSON.parsefile(joinpath(rdir, f))
        haskey(d, "best_history") || error("Run $f missing key 'best_history'")

        hist = Float64.(d["best_history"])
        length(hist) == T || error("Run $f has attempts=$(length(hist)) but expected $T")

        r = hist ./ optimal_cut
        one_minus_r = 1 .- r
        Y[:, j] = one_minus_r
    end

    m = vec(mean(Y; dims=2))
    s = vec(std(Y; dims=2))

    if clip_floor !== nothing
        m = max.(m, clip_floor)
    end

    attempts = collect(1:T)

    # -------------------- plot --------------------
    fig = Figure(size=(900, 520))
    ax = Axis(
        fig[1, 1],
        xlabel = "attempt",
        ylabel = "1 - r (best-so-far)",
        title  = @sprintf("ILS baseline (λ=1): mean ± std over %d runs", nruns),
        yscale = log10,
    )

    lines!(ax, attempts, m, label="mean")
    #errorbars!(ax, attempts, m, s)

    axislegend(ax; position=:rt)

    display(fig)  # explicit

    outpng = joinpath(rdir, "plot_ils_one_minus_r_vs_attempts_maxd$(maxdim).png")
    save(outpng, fig)
    println("Saved plot to: $outpng")
end

main()