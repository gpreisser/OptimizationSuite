# src/types.jl

struct Solution{T<:AbstractFloat}
    energy::T
    configuration::Vector{Int8}
    grad_norm::T
    theta::Vector{T}
    metadata::Dict{Symbol,Any}
end