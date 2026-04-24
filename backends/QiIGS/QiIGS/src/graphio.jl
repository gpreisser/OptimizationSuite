# src/graphio.jl
#
# Loader for your QiILS-compatible graph text files:
# each line: "w,u,v" where u,v are 0-based node indices.

using SparseArrays

"""
    load_weight_matrix(path; force_symmetric=true) -> SparseMatrixCSC{Float64,Int}

Read a graph file where each line is `w,u,v` (comma-separated) with 0-based u,v.
Returns an `N×N` sparse weight matrix `W` (1-based internally).

- If `force_symmetric=true` (default), writes both (u,v) and (v,u).
"""
function load_weight_matrix(path::AbstractString; force_symmetric::Bool = true)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    maxnode = 0

    open(path, "r") do io
        for (ln, line) in enumerate(eachline(io))
            s = strip(line)
            isempty(s) && continue

            parts = split(s, ',')
            if length(parts) != 3
                throw(ArgumentError("Invalid line $ln in $path (expected 'w,u,v'): '$line'"))
            end

            w = parse(Float64, strip(parts[1]))
            u0 = parse(Int, strip(parts[2]))
            v0 = parse(Int, strip(parts[3]))

            u = u0 + 1
            v = v0 + 1

            maxnode = max(maxnode, u, v)

            push!(rows, u); push!(cols, v); push!(vals, w)
            if force_symmetric
                push!(rows, v); push!(cols, u); push!(vals, w)
            end
        end
    end

    N = maxnode
    return sparse(rows, cols, vals, N, N)
end


"""
    graph_path(N, k, seed; weighted=false, graphs_root)

Return the canonical path for a QiILS-format graph file.
"""
function graph_path(N::Int, k::Int, seed::Int; weighted::Bool=false,
                    graphs_root::AbstractString)
    wtag = weighted ? "weighted" : "unweighted"

    return joinpath(
        graphs_root,
        string(N),
        string(k),
        "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
    )
end