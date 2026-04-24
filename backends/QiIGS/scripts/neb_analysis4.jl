# scripts/neb_relax_two_minima_clean.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using LinearAlgebra
using SparseArrays
using CairoMakie

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(RESULTS_DIR, "plots")
const NEB_DIR     = joinpath(RESULTS_DIR, "neb_analysis")
mkpath(PLOTS_DIR)
mkpath(NEB_DIR)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

vector_float(x) = Float64.(x)

function get_result_for_lambda(results_per_lambda, λtarget; atol=1e-12)
    for entry in results_per_lambda
        λ = Float64(entry["lambda"])
        if isapprox(λ, λtarget; atol=atol, rtol=0.0)
            return entry
        end
    end
    error("No entry found for λ = $λtarget")
end

function wrap_mod_pi!(θ::Vector{Float64})
    @inbounds for i in eachindex(θ)
        θ[i] = mod(θ[i], pi)
    end
    return θ
end

function theta_distance_modpi(θ1::Vector{Float64}, θ2::Vector{Float64})
    @assert length(θ1) == length(θ2)
    s = 0.0
    @inbounds for i in eachindex(θ1)
        d = abs(mod(θ1[i] - θ2[i], pi))
        d = min(d, pi - d)
        s += d^2
    end
    return sqrt(s)
end

# ------------------------------------------------------------
# continuous energy / gradient
# compatible with your solver gradient
# ------------------------------------------------------------

function continuous_energy(W::SparseMatrixCSC{Float64,Int}, θ::Vector{Float64}, λ::Float64)
    N = length(θ)

    c2 = Vector{Float64}(undef, N)
    s2 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
        s2[i] = sin(2 * θ[i])
    end

    epair = 0.0
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                epair += W.nzval[idx] * c2[row] * c2[col]
            end
        end
    end

    efield = sum(s2)
    return λ * epair - (1 - λ) * efield
end

function continuous_gradient!(g::Vector{Float64}, θ::Vector{Float64}, λ::Float64, W::SparseMatrixCSC{Float64,Int})
    N = length(θ)

    c2 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
    end

    @inbounds for i in 1:N
        a = 0.0
        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            a += W.nzval[idx] * c2[j]
        end
        g[i] = -2 * λ * a * sin(2 * θ[i]) - 2 * (1 - λ) * cos(2 * θ[i])
    end
    return g
end

function continuous_grad_norm(θ::Vector{Float64}, λ::Float64, W::SparseMatrixCSC{Float64,Int})
    g = zeros(Float64, length(θ))
    continuous_gradient!(g, θ, λ, W)
    return norm(g)
end

# ------------------------------------------------------------
# band utilities
# ------------------------------------------------------------

function interpolate_band(θ1::Vector{Float64}, θ2::Vector{Float64}, n_images::Int)
    band = [zeros(Float64, length(θ1)) for _ in 1:n_images]
    for i in 1:n_images
        s = (i - 1) / (n_images - 1)
        @inbounds for j in eachindex(θ1)
            band[i][j] = (1 - s) * θ1[j] + s * θ2[j]
        end
        wrap_mod_pi!(band[i])
    end
    return band
end

function band_energies(band, W, λ)
    E = zeros(Float64, length(band))
    @inbounds for i in eachindex(band)
        E[i] = continuous_energy(W, band[i], λ)
    end
    return E
end

function band_grad_norms(band, W, λ)
    G = zeros(Float64, length(band))
    @inbounds for i in eachindex(band)
        G[i] = continuous_grad_norm(band[i], λ, W)
    end
    return G
end

function image_distances(band)
    d = zeros(Float64, length(band) - 1)
    @inbounds for i in 1:(length(band) - 1)
        d[i] = norm(band[i + 1] .- band[i])
    end
    return d
end

function tangent(prev::Vector{Float64}, next::Vector{Float64})
    t = next .- prev
    nt = norm(t)
    if nt < 1e-14
        t .= 0.0
        t[1] = 1.0
        return t
    end
    return t ./ nt
end

# ------------------------------------------------------------
# minimal NEB relaxation
# ------------------------------------------------------------

function neb_relax!(band, W, λ; spring_k=1.0, step_size=0.005, force_tol=1e-5, max_iters=4000, verbose_every=100)
    n_images = length(band)
    N = length(band[1])

    grads  = [zeros(Float64, N) for _ in 1:n_images]
    forces = [zeros(Float64, N) for _ in 1:n_images]

    force_history = Float64[]
    barrier_history = Float64[]

    for it in 1:max_iters
        E = band_energies(band, W, λ)
        emin = min(E[1], E[end])
        push!(barrier_history, maximum(E) - emin)

        max_force = 0.0

        for i in 2:(n_images - 1)
            prev = band[i - 1]
            curr = band[i]
            next = band[i + 1]

            continuous_gradient!(grads[i], curr, λ, W)
            t̂ = tangent(prev, next)

            gdot = dot(grads[i], t̂)

            # true force perpendicular to path
            @inbounds for j in 1:N
                forces[i][j] = -grads[i][j] + gdot * t̂[j]
            end

            # spring force parallel to path
            dplus = norm(next .- curr)
            dminus = norm(curr .- prev)
            fs = spring_k * (dplus - dminus)

            @inbounds for j in 1:N
                forces[i][j] += fs * t̂[j]
            end

            fi = norm(forces[i])
            max_force = max(max_force, fi)
        end

        push!(force_history, max_force)

        if verbose_every > 0 && (it == 1 || it % verbose_every == 0)
            @printf("iter %4d   max_force = %.6e   barrier = %.6e\n", it, max_force, barrier_history[end])
        end

        if max_force < force_tol
            println("NEB converged at iter = $it")
            return Dict(
                "converged" => true,
                "iterations" => it,
                "max_force" => max_force,
                "force_history" => force_history,
                "barrier_history" => barrier_history,
            )
        end

        for i in 2:(n_images - 1)
            @inbounds for j in 1:N
                band[i][j] += step_size * forces[i][j]
            end
            wrap_mod_pi!(band[i])
        end
    end

    println("NEB hit max_iters without satisfying force_tol.")
    return Dict(
        "converged" => false,
        "iterations" => max_iters,
        "max_force" => force_history[end],
        "force_history" => force_history,
        "barrier_history" => barrier_history,
    )
end

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

function main()
    println("==============================================================")
    println("  Minimal clean NEB test")
    println("==============================================================")

    # ---------------- parameters ----------------
    N = 50
    k = 3
    weighted = false
    graph_seed = 1
    λtarget = 0.3
    # which reduced minima to connect
min_idx1 = 2
min_idx2 = 3

    n_images = 21
    spring_k = 1.0
    step_size = 0.003
    force_tol = 1e-5
    max_iters = 4000

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
        "neb_relaxed_vs_straight_clean_N$(N)_k$(k)_seed$(graph_seed)_$(wtag)_lam$(λtag)_nimg$(n_images).png"
    )

    json_path = joinpath(
        NEB_DIR,
        "neb_relaxed_vs_straight_clean_N$(N)_k$(k)_seed$(graph_seed)_$(wtag)_lam$(λtag)_nimg$(n_images).json"
    )

    # ---------------- load minima ----------------
    @assert isfile(mins_path) "Minima-scan JSON not found: $mins_path"
    mins_data = JSON.parsefile(mins_path)
    entry = get_result_for_lambda(mins_data["results_per_lambda"], λtarget)
    red = entry["reduced_minima"]
    length(red) >= 2 || error("Need at least two reduced minima at λ = $λtarget")

    println("Number of reduced minima = ", length(red))

@assert min_idx1 ≤ length(red)
@assert min_idx2 ≤ length(red)

println("Using minima ", min_idx1, " and ", min_idx2)

θ1 = vector_float(red[min_idx1]["representative_theta"])
θ2 = vector_float(red[min_idx2]["representative_theta"])
println("rounded energies:")
println("min $min_idx1 → ", red[min_idx1]["rounded_energy"])
println("min $min_idx2 → ", red[min_idx2]["rounded_energy"])
    # ---------------- load graph ----------------
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)
    Wsp = (W isa SparseMatrixCSC) ? SparseMatrixCSC{Float64, Int}(W) : sparse(Float64.(W))

    println("Graph path: $gpath")
    println("Minima path: $mins_path")
    @printf("λ = %.4f\n", λtarget)
    @printf("distance(θ1, θ2) mod π = %.6f\n", theta_distance_modpi(θ1, θ2))

    # ---------------- straight band ----------------
    straight_band = interpolate_band(θ1, θ2, n_images)
    svals = collect(range(0.0, 1.0; length=n_images))

    E_straight = band_energies(straight_band, Wsp, λtarget)
    G_straight = band_grad_norms(straight_band, Wsp, λtarget)

    emin_ref = min(E_straight[1], E_straight[end])
    Erel_straight = E_straight .- emin_ref

    imax_straight = argmax(Erel_straight)
    barrier_straight = Erel_straight[imax_straight]
    s_peak_straight = svals[imax_straight]

    @printf("straight barrier = %.12f at s = %.4f\n", barrier_straight, s_peak_straight)

    # ---------------- relaxed band ----------------
    relaxed_band = [copy(img) for img in straight_band]

    neb_info = neb_relax!(
        relaxed_band, Wsp, λtarget;
        spring_k=spring_k,
        step_size=step_size,
        force_tol=force_tol,
        max_iters=max_iters,
        verbose_every=100,
    )

    E_relaxed = band_energies(relaxed_band, Wsp, λtarget)
    G_relaxed = band_grad_norms(relaxed_band, Wsp, λtarget)

    Erel_relaxed = E_relaxed .- emin_ref
    imax_relaxed = argmax(Erel_relaxed)
    barrier_relaxed = Erel_relaxed[imax_relaxed]
    s_peak_relaxed = svals[imax_relaxed]

    @printf("relaxed barrier  = %.12f at s = %.4f\n", barrier_relaxed, s_peak_relaxed)

    # ---------------- save data ----------------
    save_data = Dict(
        "experiment" => "neb_relaxed_vs_straight_clean",
        "graph_seed" => graph_seed,
        "graph_path" => gpath,
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "lambda" => λtarget,
        "mins_path" => mins_path,
        "theta_distance_modpi" => theta_distance_modpi(θ1, θ2),
        "neb_parameters" => Dict(
            "n_images" => n_images,
            "spring_k" => spring_k,
            "step_size" => step_size,
            "force_tol" => force_tol,
            "max_iters" => max_iters,
        ),
        "straight_path_summary" => Dict(
            "barrier_height" => barrier_straight,
            "peak_index" => imax_straight,
            "peak_s" => s_peak_straight,
        ),
        "relaxed_path_summary" => Dict(
            "barrier_height" => barrier_relaxed,
            "peak_index" => imax_relaxed,
            "peak_s" => s_peak_relaxed,
        ),
        "neb_convergence" => neb_info,
        "profiles" => Dict(
            "s" => svals,
            "continuous_energy_straight" => E_straight,
            "continuous_energy_straight_rel" => Erel_straight,
            "continuous_energy_relaxed" => E_relaxed,
            "continuous_energy_relaxed_rel" => Erel_relaxed,
            "grad_norm_straight" => G_straight,
            "grad_norm_relaxed" => G_relaxed,
            "image_distances_straight" => image_distances(straight_band),
            "image_distances_relaxed" => image_distances(relaxed_band),
        ),
    )

    atomic_json_write(json_path, save_data)

    # ---------------- plot ----------------
    fig = Figure(size = (900, 900))

    ax1 = Axis(fig[1, 1];
        xlabel = L"s",
        ylabel = L"E_{\mathrm{cont}} - E_{\min}",
        title = L"\lambda = 0.275",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
        titlesize = 18,
    )

    lines!(ax1, svals, Erel_straight; linewidth=3, label="straight")
    lines!(ax1, svals, Erel_relaxed; linewidth=3, linestyle=:dash, label="relaxed")
    scatter!(ax1, [s_peak_straight], [barrier_straight], markersize=12)
    scatter!(ax1, [s_peak_relaxed], [barrier_relaxed], markersize=12)
    axislegend(ax1, position=:rt, labelsize=14)

    ax2 = Axis(fig[2, 1];
        xlabel = L"s",
        ylabel = "grad norm",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    lines!(ax2, svals, G_straight; linewidth=3, label="straight")
    lines!(ax2, svals, G_relaxed; linewidth=3, linestyle=:dash, label="relaxed")
    axislegend(ax2, position=:rt, labelsize=14)

    ax3 = Axis(fig[3, 1];
        xlabel = "iteration",
        ylabel = "max force",
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 18,
        ylabelsize = 18,
    )

    lines!(ax3, 1:length(neb_info["force_history"]), neb_info["force_history"]; linewidth=3)

    Label(
        fig[0, 1],
        @sprintf("graph seed = %d   straight barrier = %.6f   relaxed barrier = %.6f",
                 graph_seed, barrier_straight, barrier_relaxed),
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