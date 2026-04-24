function _json_ready(value)
    if value isa Dict
        return Dict(string(k) => _json_ready(v) for (k, v) in value)
    elseif value isa AbstractVector
        return [_json_ready(v) for v in value]
    elseif value isa Tuple
        return [_json_ready(v) for v in value]
    elseif value isa AbstractFloat
        return isfinite(value) ? value : nothing
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        return nothing
    else
        return value
    end
end

function save_result_json(path::AbstractString, data::Dict)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, _json_ready(data), 2)
    end
    return path
end
