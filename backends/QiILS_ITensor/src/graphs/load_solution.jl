using JSON

function solution_file_path(N::Int, k::Int, seed::Int; weighted::Bool=true)
    # load_solution.jl lives in:  <root>/src/graphs/load_solution.jl
    # so root is:                <root>
    root = normpath(joinpath(@__DIR__, "..", ".."))

    base = joinpath(root, "solutions", "random_regular", string(N), string(k))
    weight_tag = weighted ? "weighted" : "unweighted"
    filename = "akmaxdata_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(weight_tag).json"
    return joinpath(base, filename)
end

function load_optimal_cut(path::AbstractString)
    if !isfile(path)
        @info "Solution file missing: $path"
        return nothing
    end

    data = JSON.parsefile(path)
    if haskey(data, "maxcut_value")
        return Float64(data["maxcut_value"])
    else
        @warn "JSON file missing field 'maxcut_value': $path"
        return nothing
    end
end

const GSET_OPTIMAL_CUTS = Dict{Int, Float64}(
    1  => 11624.0,
    2  => 11620.0,
    3  => 11622.0,
    4  => 11646.0,
    5  => 11631.0,
    6  => 2178.0,
    7  => 2006.0,
    8  => 2005.0,
    9  => 2054.0,
    10 => 2000.0,
    11 => 564.0,
    12 => 556.0,
    13 => 582.0,
    14 => 3064.0,
    15 => 3050.0,
    16 => 2927.0,
    17 => 3047.0,
    18 => 992.0,
    19 => 906.0,
    20 => 941.0,
    21 => 931.0,
    22 => 14123.0,
    23 => 14129.0,
    24 => 13507.0,
    25 => 13386.0,
    26 => 13294.0,
    27 => 3341.0,
    28 => 3298.0,
    29 => 3405.0,
    30 => 3413.0,
    31 => 3310.0,
    32 => 6889.0,
    33 => 6799.0,
    34 => 6748.0,
    35 => 6756.0,
    36 => 6382.0,
    37 => 7687.0,
    38 => 7681.0,
    39 => 7673.0,
    40 => 7673.0,
    41 => 7032.0,
    42 => 1382.0,
    43 => 666.0,
    44 => 665.0,
    45 => 665.0,
    46 => 133.0,
    47 => 12084.0,
    48 => 6000.0,
    49 => 6000.0,
    50 => 5988.0,
    51 => 3848.0,
    52 => 3007.0,
    53 => 3006.0,
    54 => 3005.0,
)

function get_gset_optimal_cut(gset::Int)
    return get(GSET_OPTIMAL_CUTS, gset, nothing)
end