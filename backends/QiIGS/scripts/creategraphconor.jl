# scripts/make_regular_graphs_N12_k3_pm1.jl
#
# Generate random regular graphs in the SAME format + naming as
# GraphLoader.create_and_save_graph_QiILS and store them under:
#   QiIGS/graphs/N/k/graph_N{N}_k{k}_seed{seed}_seedb{seed}_{weighted|unweighted}.txt
#
# This script is defensive:
# - It calls QiILS_ITensor.create_and_save_graph_QiILS
# - Then it verifies the expected filename exists
# - If the library wrote an older wrong filename (missing _weighted/_unweighted),
#   it renames it to the correct one
#
# Run from the QiIGS repo.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QiILS_ITensor
using Printf

function main()
    N = 12
    k = 3
    weighted = true
    seeds = 1:1

    ROOT = normpath(joinpath(@__DIR__, ".."))
    graphs_base = joinpath(ROOT, "graphs")
    @info "Saving graphs to" graphs_base

    dir_path = joinpath(graphs_base, string(N), string(k))
    mkpath(dir_path)

    wtag = weighted ? "weighted" : "unweighted"

    for seed in seeds
        _, retpath = QiILS_ITensor.create_and_save_graph_QiILS(
            N, k, seed;
            weighted = weighted,
            base_path = graphs_base,
        )

        expected = joinpath(
            dir_path,
            "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
        )

        legacy = joinpath(
            dir_path,
            "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt"
        )

        if isfile(expected)
            nothing
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

        if seed ≤ 5 || seed % 25 == 0
            @printf("✔ seed=%d → %s\n", seed, expected)
        end
    end

    println("Done. Generated $(length(seeds)) graphs with correct names.")
end

main()