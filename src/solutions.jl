const GSET_OPTIMAL_CUTS = Dict{Int, Float64}(
    1 => 11624.0,
    2 => 11620.0,
    3 => 11622.0,
    4 => 11646.0,
    5 => 11631.0,
    6 => 2178.0,
    7 => 2006.0,
    8 => 2005.0,
    9 => 2054.0,
    10 => 2000.0,
    11 => 564.0,
    12 => 556.0,
)

function load_known_optimal_cut(; instance_type::Symbol, kwargs...)
    if instance_type === :gset
        gset = get(kwargs, :gset, nothing)
        gset === nothing && error("For instance_type=:gset, provide gset=Int.")
        return get(GSET_OPTIMAL_CUTS, gset, nothing)
    elseif instance_type === :random_regular
        if !all(k -> haskey(kwargs, k), (:N, :k, :seed))
            error("For instance_type=:random_regular, provide N, k, and seed.")
        end

        weighted = get(kwargs, :weighted, true)
        return QiILS_ITensor.load_optimal_cut(
            QiILS_ITensor.solution_file_path(
                kwargs[:N],
                kwargs[:k],
                kwargs[:seed];
                weighted=weighted,
            ),
        )
    else
        error("Unsupported instance_type=$instance_type. Use :gset or :random_regular.")
    end
end
