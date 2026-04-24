# scripts/plot_tabu_standard_with_qiils_vs_time.jl

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

# ---------------------------------------------------------
# Paths / parameters
# ---------------------------------------------------------
const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

const GSET = 12
const NTRIALS = 100
const NSWEEPS = 10_000_000

# Match what you actually ran
const TENURES = [20, 50, 100, 150, 200, 250]

const TABU_RESULTS_DIR = joinpath(
    RESULTS_DIR,
    "StandardTabu_Gset$(GSET)_tenures_weighted_ntrials$(NTRIALS)_sweeps$(NSWEEPS)",
)

# QiILS timing assumption
const QIILS_TIME_PER_SWEEP_SEC = 0.05e-3  # 0.05 ms = 5e-5 s

# QiILS data location / parameters
const QIILS_RESULTS_DIR = "/Users/guillermo.preisser/Projects/code_optimization/results"
const INSTANCE = 12
const MIXING = 0.2
const NSAMPLES = 100
const OPTIMAL_CUT = 556.0

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
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

# Sparse subset for Tabu markers / error bars only
function sparse_plot_indices(n::Int)
    idx = Int[]

    if n >= 1
        append!(idx, 1:min(8, n))
    end
    if n >= 10
        append!(idx, 10:5:min(50, n))
    end
    if n >= 60
        append!(idx, 60:10:min(120, n))
    end
    if n >= 150
        append!(idx, 150:20:n)
    end

    sort!(idx)
    unique!(idx)
    return idx
end

function pow10_ticks(a::Int, b::Int)
    vals = 10.0 .^ (a:b)
    labs = [L"10^{%$i}" for i in a:b]
    return (vals, labs)
end

function custom_log_ticks(exponents::AbstractVector{<:Real})
    exps = Float64.(exponents)
    vals = 10.0 .^ exps

    labs = [
        isapprox(e, round(e); atol=1e-12) ?
        LaTeXString("10^{$(Int(round(e)))}") :
        LaTeXString("10^{$(round(e, digits=1))}")
        for e in exps
    ]

    return (vals, labs)
end

# ---------------------------------------------------------
# Load Tabu data ONCE
# ---------------------------------------------------------
println("------------------------------------------------------------")
println("Loading Tabu data...")
println("------------------------------------------------------------")

const TABU_ERROR_HISTORIES = Dict{Int, Vector{Float64}}()
const TABU_SEM_ERROR_HISTORIES = Dict{Int, Vector{Float64}}()
const TABU_SAVED_INDICES = Dict{Int, Vector{Int}}()
const TABU_MEDIAN_TPS = Dict{Int, Float64}()
const LOADED_TENURES = Int[]

for tenure in TENURES
    path = joinpath(
        TABU_RESULTS_DIR,
        "tenure$(tenure)",
        "standard_tabu_results_G$(GSET)_tenure$(tenure)_sweeps$(NSWEEPS)_ntrials$(NTRIALS).json",
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

        avg_error_history =
            haskey(data, "avg_error_history") ? Float64.(data["avg_error_history"]) : Float64[]
        sem_error_history =
            haskey(data, "sem_error_history") ? Float64.(data["sem_error_history"]) : Float64[]
        saved_sweep_indices =
            haskey(data, "saved_sweep_indices") ? Int.(data["saved_sweep_indices"]) : Int[]
        avg_median_time_per_sweep_sec =
            haskey(data, "avg_median_time_per_sweep_sec") ? Float64(data["avg_median_time_per_sweep_sec"]) : NaN

        if isempty(avg_error_history) || isempty(sem_error_history) || isempty(saved_sweep_indices)
            @warn "Missing or empty arrays, skipping tenure" tenure path
            continue
        end

        if length(avg_error_history) != length(saved_sweep_indices) ||
           length(sem_error_history) != length(saved_sweep_indices)
            @warn "Length mismatch, skipping tenure" tenure path length(avg_error_history) length(sem_error_history) length(saved_sweep_indices)
            continue
        end

        if !isfinite(avg_median_time_per_sweep_sec) || avg_median_time_per_sweep_sec <= 0
            @warn "Invalid avg_median_time_per_sweep_sec, skipping tenure" tenure path avg_median_time_per_sweep_sec
            continue
        end

        TABU_ERROR_HISTORIES[tenure] = avg_error_history
        TABU_SEM_ERROR_HISTORIES[tenure] = sem_error_history
        TABU_SAVED_INDICES[tenure] = saved_sweep_indices
        TABU_MEDIAN_TPS[tenure] = avg_median_time_per_sweep_sec
        push!(LOADED_TENURES, tenure)
    catch err
        @warn "Could not parse JSON, skipping tenure" tenure path exception=(err, catch_backtrace())
    end
end

isempty(LOADED_TENURES) && error("No valid Tabu result files could be loaded.")
sort!(LOADED_TENURES)

println("Loaded Tabu tenures: ", LOADED_TENURES)
for tenure in LOADED_TENURES
    println("  tenure = $(tenure), avg median time/sweep = $(TABU_MEDIAN_TPS[tenure]) s")
end

# ---------------------------------------------------------
# Load QiILS data ONCE
# ---------------------------------------------------------
println("------------------------------------------------------------")
println("Loading QiILS data...")
println("------------------------------------------------------------")

const ALL_MAXCUT = Vector{Vector{Float64}}()
const ALL_SWEEP  = Vector{Vector{Float64}}()

for i in 1:NSAMPLES
    json_path = joinpath(
        QIILS_RESULTS_DIR,
        "gset$(INSTANCE)",
        "lambda",
        "mix$(MIXING)",
        "gset$(INSTANCE)_mix$(MIXING)_lambda_sample$(i).json",
    )

    if !isfile(json_path)
        @warn "Missing QiILS file, skipping sample" i json_path
        continue
    end

    try
        dat = JSON.parsefile(json_path)

        maxcut_vec = Float64.(dat["maxcut"]) ./ OPTIMAL_CUT
        sweep_vec  = Float64.(dat["sweep_list"])

        push!(ALL_MAXCUT, maxcut_vec)
        push!(ALL_SWEEP, sweep_vec)
    catch err
        @warn "Could not parse QiILS JSON, skipping sample" i json_path exception=(err, catch_backtrace())
    end
end

isempty(ALL_MAXCUT) && error("No valid QiILS result files could be loaded.")
length(ALL_MAXCUT) == length(ALL_SWEEP) || error("Mismatch between loaded QiILS maxcut and sweep samples.")

maxcut_lengths = length.(ALL_MAXCUT)
sweep_lengths  = length.(ALL_SWEEP)

length(unique(maxcut_lengths)) == 1 || error("QiILS maxcut samples do not all have the same length.")
length(unique(sweep_lengths))  == 1 || error("QiILS sweep samples do not all have the same length.")
first(maxcut_lengths) == first(sweep_lengths) || error("QiILS maxcut and sweep vectors have different lengths.")

const MAXCUT_MAT = hcat(ALL_MAXCUT...)
const SWEEP_MAT  = hcat(ALL_SWEEP...)

const MEAN_MAXCUT = vec(mean(MAXCUT_MAT, dims=2))
const STD_MAXCUT  = vec(std(MAXCUT_MAT, dims=2))
const MEAN_SWEEP  = vec(mean(SWEEP_MAT, dims=2))
const NLOADED_QIILS = size(MAXCUT_MAT, 2)

println("Loaded QiILS samples: ", NLOADED_QIILS)

const IDX_QIILS = logscale_indices(length(MEAN_SWEEP))

const QIILS_SWEEPS = MEAN_SWEEP[IDX_QIILS]
const QIILS_X_TIME = QIILS_SWEEPS .* QIILS_TIME_PER_SWEEP_SEC
const QIILS_Y      = 1 .- MEAN_MAXCUT[IDX_QIILS]
const QIILS_YERR   = STD_MAXCUT[IDX_QIILS] ./ sqrt(NLOADED_QIILS)

println("------------------------------------------------------------")
println("Data loading complete.")
println("Re-run only the let ... end plot block below when tweaking.")
println("------------------------------------------------------------")

# =========================================================
# PLOT BLOCK ONLY
# =========================================================
let
    fig = Figure(
        size = (430, 315),
        tellwidth = false,
        tellheight = false,
        figure_padding = (1, 2, 2, 8),
    )

    ax = Axis(
        fig[1, 1];
        xscale = log10,
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false,
        xticksmirrored = true,
        yticksmirrored = true,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 16,
        ylabelsize = 16,
        xlabel = L"\mathrm{time\ (s)}",
        ylabel = L"1-r",
        yticks = pow10_ticks(-5, 0),
        xticks = custom_log_ticks([-6, -4, -2, 0, 2]),
    )

    ylims!(ax, 1e-5, 1.3)
    xlims!(ax, 10.0^-6, 10.0^2.2)

    marksize1 = 9
    plots = Any[]
    labels = Any[]

    # --------------------------------
    # Explicit colors
    # --------------------------------
    qiils_color = Makie.wong_colors()[1]  # blue

    # Requested Tabu pattern:
    # after QiILS use up triangle, down triangle, diamond, ...
    # and make tenure = 100 purple
    tabu_colors = [
        Makie.wong_colors()[2],  # tenure 20
        Makie.wong_colors()[3],  # tenure 50
        :purple,                 # tenure 100
        Makie.wong_colors()[5],  # tenure 150
        Makie.wong_colors()[6],  # tenure 200
        Makie.wong_colors()[7],  # tenure 250
    ]

    tabu_markers = [
        :utriangle,  # tenure 20
        :dtriangle,  # tenure 50
        :diamond,    # tenure 100
        :rect,       # tenure 150
        :xcross,     # tenure 200
        :star5,      # tenure 250
    ]

    # -----------------------------
    # QiILS -- keep as before
    # -----------------------------
    qiils_mask = (QIILS_X_TIME .> 0) .& (QIILS_Y .> 0)
    qx   = QIILS_X_TIME[qiils_mask]
    qy   = QIILS_Y[qiils_mask]
    qerr = QIILS_YERR[qiils_mask]

    errorbars!(
        ax,
        qx,
        qy,
        qerr;
        whiskerwidth = 6,
        color = qiils_color,
    )

    p_qiils = scatterlines!(
        ax,
        qx,
        qy;
        marker = :circle,
        markersize = marksize1,
        linewidth = 2.5,
        color = qiils_color,
    )

    push!(plots, p_qiils)
    push!(labels, L"\textrm{QiILS}")

    # -----------------------------
    # Tabu -- full line, sparse markers/error bars
    # -----------------------------
    for (k, tenure) in enumerate(LOADED_TENURES)
        sweeps_saved = TABU_SAVED_INDICES[tenure]
        x_full = sweeps_saved .* TABU_MEDIAN_TPS[tenure]
        y_full = TABU_ERROR_HISTORIES[tenure]
        yerr_full = TABU_SEM_ERROR_HISTORIES[tenure]

        mask = (x_full .> 0) .& (y_full .> 0)
        x_full = x_full[mask]
        y_full = y_full[mask]
        yerr_full = yerr_full[mask]

        isempty(x_full) && continue

        idx_sparse = sparse_plot_indices(length(x_full))
        xs = x_full[idx_sparse]
        ys = y_full[idx_sparse]
        yerrs = yerr_full[idx_sparse]

        c = tabu_colors[k]
        m = tabu_markers[k]

        errorbars!(
            ax,
            xs,
            ys,
            yerrs;
            whiskerwidth = 0,
            color = c,
        )

        p_line = lines!(
            ax,
            x_full,
            y_full;
            linewidth = 2.2,
            color = c,
        )

        p_scatter = scatter!(
            ax,
            xs,
            ys;
            marker = m,
            markersize = marksize1 - 1,
            color = c,
        )

        push!(plots, [p_line, p_scatter])
        push!(labels, L"\textrm{Tabu}\;(\mathrm{tenure}=%$(tenure))")
    end

    axislegend(
        ax,
        plots,
        labels;
        position = :lb,
        framevisible = false,
        backgroundcolor = :transparent,
        patchsize = (22, 12),
        rowgap = 2,
        labelsize = 16,
    )

    png_path = joinpath(PLOTS_DIR, "standard_tabu_with_qiils_error_vs_time_median_tabu_vertsem_sparsemarkers.png")
    pdf_path = joinpath(PLOTS_DIR, "standard_tabu_with_qiils_error_vs_time_median_tabu_vertsem_sparsemarkers.pdf")

    save(png_path, fig)
    save(pdf_path, fig)

    println("Saved plot to:")
    println("  ", png_path)
    println("  ", pdf_path)

    display(fig)
end