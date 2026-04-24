# src/metrics.jl

"""
    canonical_spin_config(s)

Return a canonical representative of the global-flip equivalence class {s, -s}.
We choose the representative with s[1] == +1 (flip all spins if s[1] == -1).

Input must be a ±1 vector (typically `Vector{Int8}`).
"""
function canonical_spin_config(s::AbstractVector{Int8})
    @inbounds if s[1] == Int8(-1)
        return Int8.(-s)
    else
        return Int8.(s)  # ensure a fresh Vector{Int8}
    end
end

"""
    spin_config_key(s) -> UInt64

Hash-like key for a ±1 spin configuration *after canonicalization*.
Uses a simple rolling hash; good enough for counting uniques.
"""
function spin_config_key(s::AbstractVector{Int8})
    c = canonical_spin_config(s)
    h = UInt64(1469598103934665603)  # FNV offset basis
    @inbounds for x in c
        # map -1 -> 0x00, +1 -> 0x01
        b = (x == Int8(1)) ? UInt64(1) : UInt64(0)
        h ⊻= b
        h *= UInt64(1099511628211)   # FNV prime
    end
    return h
end