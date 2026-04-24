"""
    maxcut_value(wg, spins)

Compute MaxCut value for spins ∈ {+1,-1}.
"""
function maxcut_value(wg, spins::Vector{Int8})
    cut = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        w = wg.weights[i, j]
        cut += w * (1 - spins[i] * spins[j]) / 2
    end
    return cut
end

"""
    delta_cut_flip(wg, spins, i)

Change in MaxCut value if spin i is flipped.
"""
function delta_cut_flip(wg, spins::Vector{Int8}, i::Int)
    si = spins[i]
    Δ = 0.0
    @inbounds for j in neighbors(wg, i)
        Δ += wg.weights[i, j] * (si * spins[j])
    end
    return Δ
end
