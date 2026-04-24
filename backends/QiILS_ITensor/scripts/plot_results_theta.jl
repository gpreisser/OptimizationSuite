using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON
using Glob
using LaTeXStrings
using Printf

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

# -----------------------------
# File discovery
# -----------------------------
function all_final_jsons_for_chi(χ::Int)
    pat = "**/*_maxdim$(χ)_nsw*.json"
    files = Glob.glob(pat, RESULTS_DIR)
    filter!(f -> !occursin("checkpoint", lowercase(f)), files)
    return files
end

function newest_checkpoint_for_chi(χ::Int)
    pat = "**/*_maxdim$(χ)_nsw*.json"
    files = Glob.glob(pat, RESULTS_DIR)
    ckpts = [f for f in files if occursin("checkpoint", lowercase(f))]
    isempty(ckpts) && return nothing
    mt = map(f -> stat(f).mtime, ckpts)
    return ckpts[argmax(mt)]
end

function print_checkpoint_progress(ck, path::AbstractString)
    n_done   = Int(get(ck, "n_graphs", 0))
    next_idx = Int(get(ck, "next_seed_index", 1))
    seeds_any = get(ck, "graph_seeds", Any[])
    seeds = isempty(seeds_any) ? Int[] : Int.(seeds_any)
    ts     = get(ck, "timestamp_utc", "unknown")

    println("--------------------------------------------------")
    println("Using CHECKPOINT file:")
    println("  ", path)
    println("  timestamp_utc: ", ts)
    println("  graphs completed (n_graphs): ", n_done)
    println("  next_seed_index: ", next_idx, " / ", length(seeds))
    if !isempty(seeds) && 1 <= next_idx <= length(seeds)
        println("  next graph seed to run: ", seeds[next_idx])
        if next_idx > 1
            println("  last completed graph seed: ", seeds[next_idx - 1])
        end
    end
    println("--------------------------------------------------")
end

# -----------------------------
# Loaders
# -----------------------------
function load_devtheta_from_final(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = get(data, "results_per_lambda", nothing)
    rpl === nothing && error("Final results JSON missing key \"results_per_lambda\": $json_path")

    λs = Float64[]
    y  = Float64[]

    for entry in rpl
        λ = Float64(entry["λ_sweep"])
        th = entry["devTheta_abs_mean"]
        (th === nothing || th == "none") && continue
        push!(λs, λ)
        push!(y, Float64(th))
    end

    p = sortperm(λs)
    return λs[p], y[p]
end

function load_devtheta_from_checkpoint(json_path::AbstractString)
    ck = JSON.parsefile(json_path)
    print_checkpoint_progress(ck, json_path)

    λs_any = get(ck, "λs", nothing)
    λs_any === nothing && error("Checkpoint missing key \"λs\": $json_path")
    λs = Float64.(λs_any)

    sum_devTh_any = get(ck, "sum_devTh", nothing)
    sum_devTh_any === nothing && error("Checkpoint missing key \"sum_devTh\": $json_path")
    sum_devTh = Float64.(sum_devTh_any)

    n_graphs = Int(get(ck, "n_graphs", 0))
    n_graphs <= 0 && error("Checkpoint has n_graphs <= 0; cannot compute means: $json_path")

    length(sum_devTh) == length(λs) || error("Checkpoint sum_devTh length != λs length")

    y = sum_devTh ./ n_graphs

    p = sortperm(λs)
    return λs[p], y[p]
end

# Merge ALL final files for chi; if none exist, fall back to newest checkpoint
function merged_series_for_chi(χ::Int)
    finals = all_final_jsons_for_chi(χ)

    if !isempty(finals)
        @info "Merging FINAL JSONs" χ count=length(finals)
        λs_all = Float64[]
        y_all  = Float64[]
        for f in finals
            @info "  + file" χ path=f
            λs, y = load_devtheta_from_final(f)
            append!(λs_all, λs)
            append!(y_all, y)
        end
        p = sortperm(λs_all)
        return λs_all[p], y_all[p], :final_merged
    end

    ckpt = newest_checkpoint_for_chi(χ)
    ckpt === nothing && return nothing
    @info "No final JSONs found; using CHECKPOINT" χ path=ckpt
    λs, y = load_devtheta_from_checkpoint(ckpt)
    return λs, y, :checkpoint
end

# -----------------------------
# Plot
# -----------------------------
function main()
    chis = [1, 2, 4, 8]

    marksize = 9
    lw = 2

    function pi_label(x::Real)
        if isapprox(x, 0.0; atol=1e-12)
            return L"0"
        end
        ratio = x / π
        for d in (1, 2, 4, 8, 16)
            n = round(Int, ratio * d)
            if isapprox(ratio, n / d; atol=1e-8)
                if d == 1
                    return n == 1 ? L"\pi" : L"%$n\pi"
                else
                    return n == 1 ? L"\pi/%$d" : L"%$n\pi/%$d"
                end
            end
        end
        return L"%$(round(ratio, digits=3))\pi"
    end

    ytick_vals = [0, π/8, π/4, 3π/8, π/2]

    fig = Figure(size=(400, 300))
    ax = Axis(fig[1, 1];
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
        xticks = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0]),
        yticks = (ytick_vals, pi_label.(ytick_vals)),
        xlabel = L"\lambda",
        ylabel = L"\mathrm{dev}\,\theta_{\mathrm{abs}}",
    )

    plots = Plot[]
    labels = Any[]

    for χ in chis
        result = merged_series_for_chi(χ)
        result === nothing && (@warn "No data found (finals or checkpoint) for χ=$(χ)"; continue)
        λs, y, kind = result

        marker = χ == 1 ? :circle :
                 χ == 2 ? :utriangle :
                 χ == 4 ? :dtriangle :
                          :diamond

        p = scatterlines!(ax, λs, y;
            marker = marker,
            markersize = marksize,
            linewidth = lw,
        )
        push!(plots, p)
        push!(labels, kind == :final_merged ? L"\chi = %$(χ)" : L"\chi = %$(χ)\ (\textrm{checkpoint})")
    end

    if !isempty(plots)
        axislegend(ax, plots, labels;
            position = :rb,
            framevisible = false,
            backgroundcolor = :transparent,
        )
    end

    display(fig)

    out = joinpath(@__DIR__, "lambda_sweep_devTheta_abs_mean_vs_lambda.png")
    save(out, fig)
    @info "Saved figure" out
end

main()