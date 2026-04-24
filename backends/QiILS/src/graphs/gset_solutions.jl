"""
Known optimal solutions for some Gset instances.
"""
const GSET_OPT = [
    11624, 11620, 11622, 11646, 11631,   # gset1–5
    2178,  2006,  2005,  2054,  2000,   # gset6–10
    564,   556                          # gset11–12
]

"""
    get_optimal_cut(i)

Return the optimal cut for Gset instance `i`.
"""
function get_optimal_cut(i::Int)
    if 1 ≤ i ≤ length(GSET_OPT)
        return GSET_OPT[i]
    else
        return nothing
    end
end