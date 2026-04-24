# scripts/make_regular_graphs.jl
#
# Generate random regular graphs in the SAME format + naming as
# GraphLoader.create_and_save_graph_QiILS and store them under:
#   QiILS_ITensor/graphs/N/k/graph_N{N}_k{k}_seed{seed}_seedb{seed}_{weighted|unweighted}.txt
#
# This script is defensive:
# - It calls QiILS_ITensor.create_and_save_graph_QiILS
# - Then it *verifies* the expected filename exists
# - If the library wrote an older “wrong” filename (missing _unweighted/_weighted),
#   it renames it to the correct one.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QiILS_ITensor
using Printf

function main()
    N = 10
    k = 3
    weighted = false
    seeds = 1:1

    graphs_base = normpath(joinpath(@__DIR__, "..", "graphs"))
    @info "Saving graphs to" graphs_base

    dir_path = joinpath(graphs_base, string(N), string(k))
    mkpath(dir_path)

    wtag = weighted ? "weighted" : "unweighted"

    for seed in seeds
        # Ask the package to create/save
        _, retpath = QiILS_ITensor.create_and_save_graph_QiILS(
            N, k, seed;
            weighted = weighted,
            base_path = graphs_base,
        )

        # The ONLY correct/expected path (matches your Python loader)
        expected = joinpath(
            dir_path,
            "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
        )

        # Common legacy “wrong” name we want to eliminate (no _unweighted/_weighted)
        legacy = joinpath(
            dir_path,
            "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt"
        )

        # ---- Ensure we end with the correct name on disk ----
        if isfile(expected)
            # all good
        elseif isfile(retpath)
            # Library wrote something, but not where/what we expect → rename/move
            mv(retpath, expected; force=true)
        elseif isfile(legacy)
            # Library wrote legacy name → rename
            mv(legacy, expected; force=true)
        else
            # Hard fail with useful context
            error(
                "Graph creation failed for seed=$(seed).\n" *
                "Expected: $(expected)\n" *
                "Returned: $(retpath) (exists? $(isfile(retpath)))\n" *
                "Legacy:   $(legacy) (exists? $(isfile(legacy)))\n"
            )
        end

        # Final assertion: must exist with correct name
        @assert isfile(expected) "Graph not found after fix-up: $expected"

        if seed ≤ 5 || seed % 50 == 0
            @printf("✔ seed=%d → %s\n", seed, expected)
        end
    end

    println("Done. Generated $(length(seeds)) graphs with correct names.")
end

main()