using JSON

"""
    solution_file_path(N, k, seed; weighted=true)

Return the path where the Python solver stores the exact MaxCut solution.
"""
function solution_file_path(N::Int, k::Int, seed::Int; weighted::Bool=true)
    base = joinpath(@__DIR__, "..", "..", "solutions", "random_regular",
                    string(N), string(k))

    weight_tag = weighted ? "weighted" : "unweighted"

    filename =
        "akmaxdata_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(weight_tag).json"

    return joinpath(base, filename)
end


"""
    load_optimal_cut(path)

Load the MaxCut optimum from a JSON file produced by the Python solver.

Returns:
- Float64 value
- `nothing` if file not found or invalid
"""
function load_optimal_cut(path::AbstractString)
    if !isfile(path)
        @info "No solution file found at: $path"
        return nothing
    end

    data = JSON.parsefile(path)

    if haskey(data, "maxcut_value")
        return Float64(data["maxcut_value"])
    end

    @warn "JSON file at $path does not contain key \"maxcut_value\"."
    return nothing
end