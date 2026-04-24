# scripts/make_regular_graphs_weighted01.jl
#
# Generate random regular graphs with random weights ~ Uniform(0,1),
# stored under:
#   QiILS_ITensor/graphs/N/k/graph_N{N}_k{k}_seed{seed}_seedb{seed}_weighted.txt
#
# Defensive behavior:
# - Calls QiILS_ITensor.create_and_save_graph_QiILS (to match topology/format expectations)
# - Ensures final filename is correct (renames legacy/wrong output)
# - Ensures weights are truly random in (0,1) by rewriting edge weights if needed
#
# NOTE: If your file format differs from "u v w" per line for weighted graphs,
# adjust read_edges/write_edges accordingly.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QiILS_ITensor
using Printf
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))
const GRAPHS_DIR = joinpath(ROOT, "graphs")

# -----------------------------
# I/O helpers (assume 1-based nodes)
# -----------------------------
function read_edges(path::AbstractString)
    # returns edges as (u,v) and ignores existing weights
    uv = Vector{Tuple{Int,Int}}()

    open(path, "r") do io
        for (ln, line) in enumerate(eachline(io))
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue

            # split on commas OR whitespace
            parts = split(s, r"[,\s]+"; keepempty=false)

            if length(parts) != 3
                error("Unexpected edge line format at $path:$ln → '$line'")
            end

            # Detect whether weight is first or last
            # Your observed format is: w,u,v  (w is Float64)
            if occursin(r"[\.eE]", parts[1])
                # w,u,v
                u = parse(Int, parts[2])
                v = parse(Int, parts[3])
                push!(uv, (u, v))
            elseif occursin(r"[\.eE]", parts[3])
                # u,v,w
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                push!(uv, (u, v))
            else
                # If all ints, assume u,v,w-as-int is not intended → fail loudly
                error("Could not detect weight position at $path:$ln → '$line'")
            end
        end
    end

    return uv
end

function write_edges_weighted01(path::AbstractString, uv::Vector{Tuple{Int,Int}})
    open(path, "w") do io
        for (u, v) in uv
            w = rand()
            @printf(io, "%.17g,%d,%d\n", w, u, v)  # w,u,v
        end
    end
end

function ensure_weighted01!(path::AbstractString; rngseed::Int)
    Random.seed!(rngseed)

    uv = read_edges(path)   # Vector{Tuple{Int,Int}}
    write_edges_weighted01(path, uv)

    return nothing
end

# -----------------------------
# Main
# -----------------------------
function main()
    N = 50
    k = 3
    weighted = true
    seeds = 1:100

    @info "Saving graphs to" GRAPHS_DIR
    dir_path = joinpath(GRAPHS_DIR, string(N), string(k))
    mkpath(dir_path)

    wtag = weighted ? "weighted" : "unweighted"  # here it's always weighted

    for seed in seeds
        # Ask the package to create/save (gets the right topology and "expected" format)
        _, retpath = QiILS_ITensor.create_and_save_graph_QiILS(
            N, k, seed;
            weighted = weighted,
            base_path = GRAPHS_DIR,
        )

        expected = joinpath(dir_path, "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt")
        legacy   = joinpath(dir_path, "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt")

        # ---- Ensure we end with the correct name on disk ----
        if isfile(expected)
            # ok
        elseif isfile(retpath)
            mv(retpath, expected; force=true)
        elseif isfile(legacy)
            mv(legacy, expected; force=true)
        else
            error(
                "Graph creation failed for seed=$(seed).\n" *
                "Expected: $(expected)\n" *
                "Returned: $(retpath) (exists? $(isfile(retpath)))\n" *
                "Legacy:   $(legacy) (exists? $(isfile(legacy)))\n"
            )
        end

        @assert isfile(expected) "Graph not found after fix-up: $expected"

        # ---- Force weights to be i.i.d. Uniform(0,1) ----
        # Use a reproducible per-seed RNG choice. If you want different weights
        # for same topology with same seed, change the mixing below.
        ensure_weighted01!(expected; rngseed = 10_000 + seed)

        if seed ≤ 5 || seed % 50 == 0
            @printf("✔ seed=%d → %s (weights ~ U(0,1))\n", seed, expected)
        end
    end

    println("Done. Generated $(length(seeds)) weighted graphs with random weights in (0,1).")
end

main()