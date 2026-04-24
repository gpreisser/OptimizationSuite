# scripts/make_regular_graphs_N12_k3_pm1weights.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Graphs
using Printf

function random_3regular_pm1_graph_file(path::AbstractString, N::Int, k::Int, seed::Int)
    rng = MersenneTwister(seed)

    g = random_regular_graph(N, k; seed=seed)

    open(path, "w") do io
        for e in edges(g)
            u = src(e) - 1   # save as 0-based to match your existing files
            v = dst(e) - 1
            w = rand(rng, Bool) ? 1 : -1
            println(io, "$(w),$(u),$(v)")
        end
    end
end

function main()
    N = 12
    k = 3
    seeds = 1:100

    ROOT = normpath(joinpath(@__DIR__, ".."))
    graphs_base = joinpath(ROOT, "graphs")
    dir_path = joinpath(graphs_base, string(N), string(k))
    mkpath(dir_path)

    for seed in seeds
        outpath = joinpath(
            dir_path,
            "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_pm1weighted.txt"
        )

        random_3regular_pm1_graph_file(outpath, N, k, seed)

        @assert isfile(outpath) "Missing graph file: $outpath"

        if seed ≤ 5 || seed % 25 == 0
            @printf("✔ seed=%d → %s\n", seed, outpath)
        end
    end

    println("Done.")
end

main()