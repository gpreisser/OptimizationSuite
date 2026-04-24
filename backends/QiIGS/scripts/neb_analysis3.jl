# scripts/interpolation_profile_two_minima.jl
#
# Purpose:
#   For one graph and one λ, load the first two representative minima from the
#   minima-scan JSON, linearly interpolate between them in angle space, and
#   evaluate the continuous energy profile along the path.
#
# Output:
#   - PNG figure under ROOT/results/plots/
#   - JSON data under ROOT/results/neb_analysis/
#
# Notes:
#   - This is NOT NEB yet.
#   - It is just the straight-line interpolation profile between the two minima.
#   - Uses the continuous energy consistent with the solver gradient.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using SparseArrays
using CairoMakie

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
const NEB_DIR = joinpath(RESULTS_DIR, "neb_analysis")
mkpath(PLOTS_DIR)
mkpath(NEB_DIR)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

function vector_float(x)
    return Float64.(x)
end

function get_result_for_lambda(results_per_lambda, λtarget::Float64; atol::Float64=1e-12)
    for entry in results_per_lambda
        λ = Float64(entry["lambda"])
        if isapprox(λ, λtarget; atol=atol, rtol=0.0)
            return entry
        end
    end
    error("No entry found for λ = $λtarget")
end

function theta_distance_modpi(
    θ1::AbstractVector{<:AbstractFloat},
    θ2::AbstractVector{<:AbstractFloat},
)
    @assert length(θ1) == length(θ2)
    s = 0.0
    @inbounds for i in eachindex(θ1)
        d = abs(mod(Float64(θ1[i]) - Float64(θ2[i]), pi))
        d = min(d, pi - d)
        s += d^2
    end
    return sqrt(s)
end

# Continuous objective consistent with the solver gradient:
#
# grad_i = -2 λ a_i sin(2θ_i) - 2 (1-λ) cos(2θ_i),
# a_i = Σ_j W_ij cos(2θ_j)
#
# One compatible energy is:
# F(θ; λ) = λ Σ_{i<j} W_ij cos(2θ_i) cos(2θ_j) - (1-λ) Σ_i sin(2θ_i)
#
function continuous_energy(
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
    λ::T,
) where {T<:AbstractFloat}
    N = length(θ)
    @assert size(W, 1) == N && size(W, 2) == N

    c2 = Vector{T}(undef, N)
    s2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
        s2[i] = sin(2 * θ[i])
    end

    Epair = zero(T)
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                Epair += W.nzval[idx] * c2[row] * c2[col]
            end
        end
    end

    Efield = zero(T)
    @inbounds for i in 1:N
        Efield += s2[i]
    end

    return λ * Epair - (one(T) - λ) * Efield
end

function continuous_gradient!(
    g::AbstractVector{T},
    θ::AbstractVector{T},
    λ::T,
    W::SparseMatrixCSC{T},
) where {T<:AbstractFloat}
    N = length(θ)
    @assert length(g) == N

    c2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
    end

    @inbounds for i in 1:N
        a = zero(T)
        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            a += W.nzval[idx] * c2[j]
        end
        g[i] = -2 * λ * a * sin(2 * θ[i]) - 2 * (1 - λ) * cos(2 * θ[i])
    end
    return g
end

function continuous_grad_norm(
    θ::AbstractVector{T},
    λ::T,
    W::SparseMatrixCSC{T},
) where {T<:AbstractFloat}
    g = zeros(T, length(θ))
    continuous_gradient!(g, θ, λ, W)
    return norm(g)
end

function interpolate_theta(
    θ1::AbstractVector{T},
    θ2::AbstractVector{T},
    s::T,
) where {T<:AbstractFloat}
    @assert length(θ1) == length(θ2)
    θ = similar(θ1)
    @inbounds for i in eachindex(θ1)
        θ[i] = (one(T) - s) * θ1[i] + s * θ2[i]
    end
    return θ
end

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

function main()
    println("==============================================================")
    println("  Straight-line interpolation profile between two minima")
    println("==============================================================")

    # -----------------------
    # User choices
    # -----------------------
    N = 50
    k = 3
    weighted = false
    graph_seed = 1
    λtarget = 0.275
    n_images = 101

    # saved minima scan
    mins_path = joinpath(
        RESULTS_DIR,
        "minima_scan_onegraph",
        "onegraph_minima_scan_N50_k3_seed1_unweighted_lam0p000_to_1p000_d0p025_conv1e-08_abin0p01_afliptrue_hess_on_htol1e-08_gfiltertrue_gnmax1e-06_ninit2000_outer1_inner5000.json"
    )

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    wtag = weighted ? "weighted" : "unweighted"
    λtag = replace(@sprintf("%.4f", λtarget), "." => "p")

    png_path = joinpath(
        PLOTS_DIR,
        "interp_profile_two_minima_N$(N)_k$(k)_seed$(graph_seed)_$(wtag)_lam$(λtag)_nimg$(n_images).png"
    )

    json_path = joinpath(
        NEB_DIR,
        "interp_profile_two_minima_N$(N)_k$(k)_seed$(graph_seed)_$(wtag)_lam$(λtag)_nimg$(n_images).json"
    )

    # -----------------------
    # Load minima scan
    # -----------------------
    @assert isfile(mins_path) "Minima-scan JSON not found: $mins_path"
    mins_data = JSON.parsefile(mins_path)
    entry = get_result_for_lambda(mins_data["results_per_lambda"], λtarget)
    red = entry["reduced_minima"]
    length(red) >= 2 || error("Need at least two reduced minima at λ = $λtarget")

    θ1 = vector_float(red[1]["representative_theta"])
    θ2 = vector_float(red[2]["representative_theta"])

    # -----------------------
    # Load graph
    # -----------------------
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)
    Wsp = (W isa SparseMatrixCSC) ? SparseMatrixCSC{Float64, Int}(W) : sparse(Float64.(W))

    # -----------------------
    # Build interpolation profile
    # -----------------------
    svals = collect(range(0.0, 1.0; length=n_images))
    Econt = zeros(Float64, n_images)
    Eround = zeros(Float64, n_images)
    gnorms = zeros(Float64, n_images)

    for (i, s) in enumerate(svals)
        θ = interpolate_theta(θ1, θ2, s)
        Econt[i] = continuous_energy(Wsp, θ, λtarget)
        Eround[i] = QiIGS.energy_from_angles(Wsp, θ)
        gnorms[i] = continuous_grad_norm(θ, λtarget, Wsp)
    end

    Emin_cont = min(Econt[1], Econt[end])
    Econt_rel = Econt .- Emin_cont

    # identify highest point along straight interpolation
    imax = argmax(Econt_rel)
    s_peak = svals[imax]
    Ebarrier = Econt_rel[imax]

    # endpoint diagnostics
    gn1 = continuous_grad_norm(θ1, λtarget, Wsp)
    gn2 = continuous_grad_norm(θ2, λtarget, Wsp)
    hs1 = QiIGS.hessian_summary(Wsp, θ1, λtarget; tol=1e-8)
    hs2 = QiIGS.hessian_summary(Wsp, θ2, λtarget; tol=1e-8)

    println("Graph path: $gpath")
    println("Minima path: $mins_path")
    @printf("λ = %.4f\n", λtarget)
    @printf("n_images = %d\n", n_images)
    @printf("distance(θ1, θ2) mod π = %.6f\n", theta_distance_modpi(θ1, θ2))
    @printf("endpoint continuous energies: %.12f , %.12f\n", Econt[1], Econt[end])
    @printf("endpoint rounded energies:   %.12f , %.12f\n", Eround[1], Eround[end])
    @printf("endpoint grad norms:         %.12e , %.12e\n", gn1, gn2)
    @printf("endpoint mineigs:            %.12e , %.12e\n",
        Float64(hs1[:hess_mineig]), Float64(hs2[:hess_mineig]))
    @printf("straight-path peak at s = %.4f with barrier %.12f\n", s_peak, Ebarrier)

    # -----------------------
    # Save JSON
    # -----------------------
    save_data = Dict(
        "experiment" => "interp_profile_two_minima",
        "graph_seed" => graph_seed,
        "graph_path" => gpath,
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "lambda" => λtarget,
        "n_images" => n_images,
        "mins_path" => mins_path,
        "theta_distance_modpi" => theta_distance_modpi(θ1, θ2),
        "endpoint_summary" => Dict(
            "continuous_energy_1" => Econt[1],
            "continuous_energy_2" => Econt[end],
            "rounded_energy_1" => Eround[1],
            "rounded_energy_2" => Eround[end],
            "grad_norm_1" => gn1,
            "grad_norm_2" => gn2,
            "hess_mineig_1" => Float64(hs1[:hess_mineig]),
            "hess_mineig_2" => Float64(hs2[:hess_mineig]),
            "hess_is_minimum_1" => hs1[:hess_is_minimum],
            "hess_is_minimum_2" => hs2[:hess_is_minimum],
        ),
        "straight_path_summary" => Dict(
            "continuous_energy_min_reference" => Emin_cont,
            "peak_index" => imax,
            "peak_s" => s_peak,
            "barrier_height" => Ebarrier,
        ),
        "profile" => Dict(
            "s" => svals,
            "continuous_energy" => Econt,
            "continuous_energy_relative" => Econt_rel,
            "rounded_energy" => Eround,
            "continuous_grad_norm" => gnorms,
        ),
    )

    atomic_json_write(json_path, save_data)

    # -----------------------
    # Plot
    # -----------------------
    fig = Figure(size = (900, 800))

    ax1 = Axis(fig[1, 1];
        xlabel = L"s",
        ylabel = L"E_{\mathrm{cont}}(s)-E_{\min}",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
        title = L"\lambda = 0.275 \ \mathrm{straight\ interpolation}",
        titlesize = 18,
    )

    lines!(ax1, svals, Econt_rel, linewidth = 3)
    scatter!(ax1, [svals[1], s_peak, svals[end]], [Econt_rel[1], Econt_rel[imax], Econt_rel[end]], markersize = 12)

    ax2 = Axis(fig[2, 1];
        xlabel = L"s",
        ylabel = L"E_{\mathrm{round}}(s)",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    lines!(ax2, svals, Eround, linewidth = 3)

    ax3 = Axis(fig[3, 1];
        xlabel = L"s",
        ylabel = L"\mathrm{grad\ norm}",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    lines!(ax3, svals, gnorms, linewidth = 3)

    Label(fig[0, 1],
        @sprintf("graph seed = %d,  barrier along straight path = %.6f at s = %.3f", graph_seed, Ebarrier, s_peak),
        fontsize = 18
    )

    rowgap!(fig.layout, 12)

    save(png_path, fig)
    display(fig)

    println("Saved figure: $png_path")
    println("Saved data:   $json_path")
    println("==============================================================")
end

main()