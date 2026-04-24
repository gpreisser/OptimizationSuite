# src/graphs/load_graph.jl

using Random
using Graphs
using SimpleWeightedGraphs
using JSON

function _default_qiils_graphs_root()
    return get(
        ENV,
        "QIILS_GRAPHS_ROOT",
        joinpath(dirname(dirname(@__DIR__)), "graphs"),
    )
end

# -------------------------------------------------------
# Detect Gset format
# -------------------------------------------------------
function _is_gset_file(path::AbstractString)
    open(path, "r") do f
        first = split(strip(readline(f)))
        return length(first) == 2 &&
               all(x -> tryparse(Int, x) !== nothing, first)
    end
end

# -------------------------------------------------------
# Custom CSV-like "<w,u,v>" graph format loader
# (your QiILS graphs: zero-based u,v in file)
# -------------------------------------------------------
function _load_custom_graph(path::AbstractString; weighted::Bool=true)
    lines = readlines(path)
    edges_parsed = [split(strip(ln), ",") for ln in lines]

    # infer number of nodes (0-based → shift to 1-based)
    nodes = Int[]
    for row in edges_parsed
        @assert length(row) == 3 "Invalid line (expected w,u,v): $(join(row, ","))"
        _, u, v = row
        push!(nodes, parse(Int, u) + 1)
        push!(nodes, parse(Int, v) + 1)
    end
    N = maximum(nodes)

    g = SimpleWeightedGraph(N)

    for row in edges_parsed
        w_str, u_str, v_str = row
        u = parse(Int, u_str) + 1
        v = parse(Int, v_str) + 1
        w = weighted ? parse(Float64, w_str) : 1.0
        add_edge!(g, u, v, w)
    end

    return g
end

# -------------------------------------------------------
# Gset loader (supports both 2- and 3-column formats)
# -------------------------------------------------------
function _load_gset(path::AbstractString)
    open(path, "r") do f
        first = split(strip(readline(f)))
        N = parse(Int, first[1])

        g = SimpleWeightedGraph(N)

        for line in eachline(f)
            parts = split(strip(line))

            if length(parts) == 2
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = 1.0
            elseif length(parts) == 3
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = parse(Float64, parts[3])
            else
                error("Invalid Gset line: $line")
            end

            add_edge!(g, u, v, w)
        end

        return g
    end
end

# -------------------------------------------------------
# Helper: canonical QiILS graph path (YOUR original naming)
# -------------------------------------------------------
@inline function _graph_path_qiils(base_path::AbstractString, N::Int, k::Int, seed::Int)
    dir_path = joinpath(base_path, string(N), string(k))
    filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt"
    return joinpath(dir_path, filename)
end

# -------------------------------------------------------
# Generate + save QiILS-compatible random regular graph
# IMPORTANT FIX: do NOT call add_edge! to "set" weights on an existing edge.
# We write directly into the weights matrix.
# -------------------------------------------------------
function create_and_save_graph_QiILS(
    N::Int, k::Int, seed::Int;
    weighted::Bool = true,
    base_path::AbstractString = _default_qiils_graphs_root(),
)
    # 1) generate unweighted regular graph structure
    g = Graphs.random_regular_graph(N, k; seed=seed)
    wg = SimpleWeightedGraph(g)

    # 2) deterministically assign weights
    Random.seed!(seed)
    W = wg.weights

    if weighted
        rlist = rand(length(edges(wg)))
        i = 1
        for e in edges(wg)
            u, v = src(e), dst(e)
            w = rlist[i]
            W[u, v] = w
            W[v, u] = w
            i += 1
        end
    else
        for e in edges(wg)
            u, v = src(e), dst(e)
            W[u, v] = 1.0
            W[v, u] = 1.0
        end
    end

    # 3) save to disk (YOUR original convention)
    gpath = _graph_path_qiils(base_path, N, k, seed)
    mkpath(dirname(gpath))

    open(gpath, "w") do io
        for e in edges(wg)
            u0 = src(e) - 1   # zero-based indexing to match Python
            v0 = dst(e) - 1
            w  = W[src(e), dst(e)]
            println(io, "$(w),$(u0),$(v0)")
        end
    end

    return wg, gpath
end

# -------------------------------------------------------
# Unified graph loader
# Behavior:
#  - gset=... loads a Gset file
#  - path=... loads either Gset or custom CSV
#  - (N,k,seed) ensures a QiILS graph file exists; if not, creates it; then loads it
# -------------------------------------------------------
function load_graph(; gset=nothing,
                     path=nothing,
                     N=nothing,
                     k=nothing,
                     weighted::Bool=true,
                     seed::Int=1,
                     base_path::AbstractString = _default_qiils_graphs_root())

    if gset !== nothing
        fname = "G$(gset)"
        isfile(fname) || error("Gset file '$fname' not found.")
        return _load_gset(fname)
    end

    if path !== nothing
        return _is_gset_file(path) ? _load_gset(path) :
                                     _load_custom_graph(path; weighted=weighted)
    end

    if N !== nothing && k !== nothing
        gpath = _graph_path_qiils(base_path, N, k, seed)
        if !isfile(gpath)
            create_and_save_graph_QiILS(N, k, seed; weighted=weighted, base_path=base_path)
            @assert isfile(gpath) "Graph creation failed; expected file not found: $gpath"
        end
        return _load_custom_graph(gpath; weighted=weighted)
    end

    error("load_graph requires gset=Int, path=String, or (N,k) plus seed.")
end
