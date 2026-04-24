
using Random
using Graphs
using SimpleWeightedGraphs
using JSON

function _default_qiils_itensor_graphs_root()
    return get(
        ENV,
        "QIILS_ITENSOR_GRAPHS_ROOT",
        get(
            ENV,
            "QIILS_GRAPHS_ROOT",
            joinpath(dirname(dirname(@__DIR__)), "..", "QiILS", "graphs"),
        ),
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
# Save QiILS-compatible random regular graph
# -------------------------------------------------------
function create_and_save_graph_QiILS(N::Int, k::Int, seed::Int;
        weighted::Bool = true,
        base_path::AbstractString = _default_qiils_itensor_graphs_root())

    g = Graphs.random_regular_graph(N, k; seed=seed)
    g_weighted = SimpleWeightedGraph(g)

    Random.seed!(seed)

    # Assign weights
    if weighted
        rlist = rand(length(edges(g)))
        i = 1
        for e in edges(g)
            u, v = src(e), dst(e)
            add_edge!(g_weighted, u, v, rlist[i])
            i += 1
        end
    else
        # Unweighted: all weights = 1.0
        for e in edges(g)
            u, v = src(e), dst(e)
            add_edge!(g_weighted, u, v, 1.0)
        end
    end

    dir_path = joinpath(base_path, string(N), string(k))
    mkpath(dir_path)

    wtag = weighted ? "weighted" : "unweighted"
    filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
    full_path = joinpath(dir_path, filename)

    open(full_path, "w") do io
        for e in edges(g_weighted)
            u = src(e) - 1
            v = dst(e) - 1
            w = g_weighted.weights[src(e), dst(e)]
            println(io, "$(w),$(u),$(v)")
        end
    end

    return g_weighted, full_path
end

# -------------------------------------------------------
# Custom CSV "<w,u,v>" loader
# -------------------------------------------------------
function _load_custom_graph(path::AbstractString; weighted::Bool=true)
    lines = readlines(path)
    edges_parsed = [split(strip(l), ",") for l in lines]

    # infer max node index
    nodes = Int[]
    for row in edges_parsed
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
# Gset loader
# -------------------------------------------------------
function _load_gset(path::AbstractString)
    open(path, "r") do f
        first = split(strip(readline(f)))
        N = parse(Int, first[1])
        g = SimpleWeightedGraph(N)

        for line in eachline(f)
            parts = split(strip(line))

            if length(parts) == 2
                u, v = parse.(Int, parts)
                w = 1.0
            elseif length(parts) == 3
                u, v = parse.(Int, parts[1:2])
                w = parse(Float64, parts[3])
            else
                error("Invalid Gset format line: $line")
            end

            add_edge!(g, u, v, w)
        end

        return g
    end
end

# -------------------------------------------------------
# Fallback: random regular generator
# -------------------------------------------------------
function _generate_random_regular(N::Int, k::Int; weighted=true, seed=1)
    g = Graphs.random_regular_graph(N, k; seed=seed)
    wg = SimpleWeightedGraph(g)

    Random.seed!(seed)
    for e in edges(g)
        u, v = src(e), dst(e)
        w = weighted ? rand() : 1.0
        add_edge!(wg, u, v, w)
    end

    return wg
end

# -------------------------------------------------------
# Unified loader
# -------------------------------------------------------
function load_graph(; gset=nothing, path=nothing,
                     N=nothing, k=nothing,
                     weighted::Bool=true, seed::Int=1)

    if gset !== nothing
        f = "G$(gset)"
        isfile(f) || error("Gset file '$f' not found.")
        return _load_gset(f)
    end

    if path !== nothing
        return _is_gset_file(path) ?
               _load_gset(path) :
               _load_custom_graph(path; weighted=weighted)
    end

    if N !== nothing && k !== nothing
        return _generate_random_regular(N, k; weighted=weighted, seed=seed)
    end

    error("load_graph requires gset=Int, path=String or (N,k).")
end

export load_graph, create_and_save_graph_QiILS
