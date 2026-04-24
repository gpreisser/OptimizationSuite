using JSON
using SparseArrays

function _default_qiigs_solutions_root()
    return get(
        ENV,
        "QIIGS_SOLUTIONS_ROOT",
        joinpath(dirname(dirname(@__DIR__)), "..", "QiILS_ITensor", "solutions"),
    )
end

"Canonical akmaxdata path (matches the filenames you showed)."
function akmax_solution_path(
    N::Int,
    k::Int,
    seed::Int;
    weighted::Bool=false,
    solutions_root::AbstractString=_default_qiigs_solutions_root(),
)
    wtag = weighted ? "weighted" : "unweighted"
    fname = "akmaxdata_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).json"
    return joinpath(solutions_root, "random_regular", string(N), string(k), fname)
end

"Load optimal cut from akmaxdata JSON (key: maxcut_value). Returns `nothing` if missing."
function load_optimal_cut(path::AbstractString)
    isfile(path) || return nothing
    d = JSON.parsefile(path)
    return haskey(d, "maxcut_value") ? Float64(d["maxcut_value"]) : nothing
end

"Load optimal Ising energy from akmaxdata JSON (key: ising_energy). Returns `nothing` if missing."
function load_optimal_ising_energy(path::AbstractString)
    isfile(path) || return nothing
    d = JSON.parsefile(path)
    return haskey(d, "ising_energy") ? Float64(d["ising_energy"]) : nothing
end

"Total edge weight Wtot = sum_{i<j} W[i,j] for symmetric sparse matrix."
function total_edge_weight_upper(W::SparseMatrixCSC{T}) where {T<:Real}
    n = size(W, 1)
    @assert size(W,2) == n
    s = 0.0
    @inbounds for col in 1:n
        for idx in W.colptr[col]:(W.colptr[col+1]-1)
            row = W.rowval[idx]
            if row < col
                s += float(W.nzval[idx])
            end
        end
    end
    return s
end

"From Ising energy E and optimal cut, compute cut_hat and ratio."
function cut_hat_and_ratio(W::SparseMatrixCSC, E::Real, optimal_cut::Real)
    Wtot = total_edge_weight_upper(W)
    cut_hat = (Wtot - float(E)) / 2
    ratio = cut_hat / float(optimal_cut)
    return cut_hat, ratio
end
