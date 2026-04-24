function _default_gset_root()
    return joinpath(dirname(@__DIR__), "..", "QiILS", "graphs", "gset")
end

function _gset_path(gset::Integer; gset_root::Union{Nothing,AbstractString}=nothing)
    root = gset_root === nothing ?
        get(ENV, "OPTIMIZATIONSUITE_GSET_ROOT", _default_gset_root()) :
        gset_root
    return joinpath(root, "G$(gset)")
end

function load_instance_graph(; instance_type::Symbol, kwargs...)
    if instance_type === :gset
        gset = get(kwargs, :gset, nothing)
        gset === nothing && error("For instance_type=:gset, provide gset=Int.")

        gset_root = get(kwargs, :gset_root, nothing)
        path = _gset_path(gset; gset_root=gset_root)
        isfile(path) || error("Gset file not found at $path")
        return QiILS.load_graph(path=path), Dict("gset" => gset, "graph_path" => path)
    elseif instance_type === :random_regular
        if !all(k -> haskey(kwargs, k), (:N, :k, :seed))
            error("For instance_type=:random_regular, provide N, k, and seed.")
        end

        weighted = get(kwargs, :weighted, true)
        N = kwargs[:N]
        k = kwargs[:k]
        seed = kwargs[:seed]
        graph = QiILS.load_graph(N=N, k=k, seed=seed, weighted=weighted)
        return graph, Dict(
            "N" => N,
            "k" => k,
            "seed" => seed,
            "weighted" => weighted,
        )
    else
        error("Unsupported instance_type=$instance_type. Use :gset or :random_regular.")
    end
end

function graph_to_sparse_matrix(wg)
    N = nv(wg)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for e in edges(wg)
        u = src(e)
        v = dst(e)
        w = Float64(wg.weights[u, v])
        push!(rows, u); push!(cols, v); push!(vals, w)
        push!(rows, v); push!(cols, u); push!(vals, w)
    end

    return sparse(rows, cols, vals, N, N)
end
