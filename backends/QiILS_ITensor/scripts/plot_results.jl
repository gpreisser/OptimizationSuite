using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON
using Glob
using LaTeXStrings

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

# -----------------------------
# Pick newest final if exists else newest checkpoint
# -----------------------------
function newest_final_or_checkpoint_for_chi(χ::Int)
    pat = "**/*_maxdim$(χ)_nsw*.json"
    files = Glob.glob(pat, RESULTS_DIR)
    isempty(files) && return nothing

    finals = [f for f in files if !occursin("checkpoint", lowercase(f))]
    ckpts  = [f for f in files if occursin("checkpoint", lowercase(f))]

    if !isempty(finals)
        mt = map(f -> stat(f).mtime, finals)
        return (finals[argmax(mt)], :final)
    elseif !isempty(ckpts)
        mt = map(f -> stat(f).mtime, ckpts)
        return (ckpts[argmax(mt)], :checkpoint)
    else
        return nothing
    end
end

# -----------------------------
# Checkpoint progress print (accept JSON.Object or Dict-like)
# -----------------------------
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
    if !isempty(seeds)
        if 1 <= next_idx <= length(seeds)
            println("  next graph seed to run: ", seeds[next_idx])
            if next_idx > 1
                println("  last completed graph seed: ", seeds[next_idx - 1])
            end
        else
            println("  next_seed_index out of range (maybe finished).")
        end
    end
    println("--------------------------------------------------")
end

# -----------------------------
# Load from FINAL json (results_per_lambda)
# -----------------------------
function load_from_final(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = get(data, "results_per_lambda", nothing)
    rpl === nothing && error("Final results JSON missing key \"results_per_lambda\": $json_path")

    λs = Float64[]
    y  = Float64[]
    for entry in rpl
        λ = Float64(entry["λ_sweep"])
        r = entry["approx_ratio_mean"]
        (r == "none" || r === nothing) && continue
        push!(λs, λ)
        push!(y, 1.0 - Float64(r))
    end

    p = sortperm(λs)
    return λs[p], y[p]
end

# -----------------------------
# Load from CHECKPOINT json (sum_ratio, n_ratio, λs)
# -----------------------------
function load_from_checkpoint(json_path::AbstractString)
    ck = JSON.parsefile(json_path)
    print_checkpoint_progress(ck, json_path)

    λs_any = get(ck, "λs", nothing)
    λs_any === nothing && error("Checkpoint missing key \"λs\": $json_path")
    λs = Float64.(λs_any)

    sum_ratio_any = get(ck, "sum_ratio", nothing)
    n_ratio_any   = get(ck, "n_ratio", nothing)
    (sum_ratio_any === nothing || n_ratio_any === nothing) && error(
        "Checkpoint missing sum_ratio or n_ratio (cannot reconstruct approx ratio): $json_path"
    )

    sum_ratio = Float64.(sum_ratio_any)
    n_ratio   = Int.(n_ratio_any)

    length(sum_ratio) == length(λs) || error("Checkpoint sum_ratio length != λs length")
    length(n_ratio)   == length(λs) || error("Checkpoint n_ratio length != λs length")

    λ_keep = Float64[]
    y_keep = Float64[]
    for i in eachindex(λs)
        if n_ratio[i] > 0
            rmean = sum_ratio[i] / n_ratio[i]
            push!(λ_keep, λs[i])
            push!(y_keep, 1.0 - rmean)
        end
    end

    p = sortperm(λ_keep)
    return λ_keep[p], y_keep[p]
end

function load_lambda_and_1minus_ratio(json_path::AbstractString, kind::Symbol)
    if kind == :final
        return load_from_final(json_path)
    elseif kind == :checkpoint
        return load_from_checkpoint(json_path)
    else
        error("Unknown kind = $kind")
    end
end

function main()
    chis = [1, 2, 4, 8]

    # ---- style knobs (match your example) ----
    marksize = 9
    whiskerwidth = 6
    lw = 2

    fig = Figure(size=(400, 300))
    ax = Axis(fig[1, 1];
        yscale = log10,
        # xscale = log10,              # (NOT for λ, keep linear)
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
        xlabel = L"\lambda",
        ylabel = L"1-r",
        xticks = (0.0:0.2:1.0, string.(0.0:0.2:1.0)),
    )
   # ylims!(ax, 1e-5, 1.0)

    plots = Plot[]
    labels = Any[]

    for χ in chis
        pick = newest_final_or_checkpoint_for_chi(χ)
        if pick === nothing
            @warn "No JSON (final or checkpoint) found for χ=$(χ) under $(RESULTS_DIR)"
            continue
        end

        path, kind = pick
        @info "Using file" χ kind path

        λs, y = load_lambda_and_1minus_ratio(path, kind)
        if isempty(λs)
            @warn "No plottable data (no valid ratios yet?)" χ kind path
            continue
        end

        # simple per-χ marker variety (similar vibe to your example)
        marker = χ == 1 ? :circle :
                 χ == 2 ? :utriangle :
                 χ == 4 ? :dtriangle :
                          :diamond

        # If you *have* error bars for final JSON (ratio_stderr) we can add them,
        # but for now we just plot points/lines like your example style.
        p = scatterlines!(ax, λs, y;
            marker = marker,
            markersize = marksize,
            linewidth = lw,
        )

        push!(plots, p)
        push!(labels, L"\chi = %$(χ)\ (\textrm{%$(String(kind))})")
    end

    if !isempty(plots)
        axislegend(ax, plots, labels;
            position = :lb,
            framevisible = false,
            backgroundcolor = :transparent,
        )
    end

    display(fig)

    out = joinpath(@__DIR__, "lambda_sweep_1minus_ratio_vs_lambda.png")
    save(out, fig)
    @info "Saved figure" out
end

main()