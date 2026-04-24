# scripts/plot_threshold_sign_bias_lambda.jl
#
# Plot:
#   B(λ) = (1/N) * Σ_i | < sign(θ_i - π/4) >_runs |
#
# Interpretation:
#   - B(λ) ≈ 0 : for each site, runs are roughly balanced on both sides of π/4
#   - B(λ) ≈ 1 : each site is consistently biased to one side of π/4
#
# This is designed to reveal structure for λ < 0.25 even when
# devTheta_abs_mean stays ~0.
#
# IMPORTANT:
#   - Reads θ from sol.metadata[:theta_converged]
#   - Uses fresh solves (does not rely on aggregated JSON)
#   - Saves under ROOT/results/plots

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using CairoMakie
using Statistics
using Printf

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

function sign_bias_metrics(
    W,
    N;
    λs::AbstractVector{<:Real},
    n_inits::Int,
    seed_base::Int,
    iterations::Int,
    inner_iterations::Int,
    tao::Real,
    angle_conv::Real,
    init_mode::Symbol,
    compute_hessian::Bool,
    hessian_tol::Real,
)
    λvals = Float64[]
    Bvals = Float64[]
    mean_abs_delta_vals = Float64[]

    for λ in λs
        site_sign_sum = zeros(Float64, N)
        sum_abs_delta = 0.0
        n_kept = 0

        for r in 1:n_inits
            run_seed = seed_base + r * 10_000 + Int(round(Float64(λ) * 1000))

            sol = QiIGS.solve!(
                W, N;
                solver = :grad,
                seed = run_seed,
                lambda = λ,
                iterations = iterations,
                inner_iterations = inner_iterations,
                tao = tao,
                angle_conv = angle_conv,
                init_mode = init_mode,
                save_params = true,
                progressbar = false,
                compute_hessian = compute_hessian,
                hessian_tol = hessian_tol,
            )

            θ = get(sol.metadata, :theta_converged, Float64[])
            isempty(θ) && error("Missing :theta_converged in solver metadata")

            # Optional: mimic your Hessian-filtered angle analysis
            if compute_hessian
                is_minimum = get(sol.metadata, :hess_is_minimum, false)
                is_minimum || continue
            end

            @inbounds for i in 1:N
                δ = θ[i] - (pi / 4)
                site_sign_sum[i] += sign(δ)
                sum_abs_delta += abs(δ)
            end
            n_kept += 1
        end

        if n_kept == 0
            push!(λvals, Float64(λ))
            push!(Bvals, NaN)
            push!(mean_abs_delta_vals, NaN)
        else
            site_sign_mean = site_sign_sum ./ n_kept
            B = mean(abs.(site_sign_mean))
            mean_abs_delta = sum_abs_delta / (N * n_kept)

            push!(λvals, Float64(λ))
            push!(Bvals, B)
            push!(mean_abs_delta_vals, mean_abs_delta)
        end

        @printf("λ=%.4f   B(λ)=%.6f   mean|θ-π/4|=%.6e   n_kept=%d\n",
            Float64(λ), Bvals[end], mean_abs_delta_vals[end], n_kept)
    end

    return λvals, Bvals, mean_abs_delta_vals
end

function main()
    N = 50
    k = 3
    weighted = false
    graph_seed = 1

    λs = collect(0.0:0.025:0.25)
    n_inits = 2000              # start smaller for a quick diagnostic; raise if needed

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-8
    init_mode = :uniform

    compute_hessian = true
    hessian_tol = 1e-8

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    seed_base = graph_seed * 1_000_000

    λvals, Bvals, mean_abs_delta_vals = sign_bias_metrics(
        W, N;
        λs = λs,
        n_inits = n_inits,
        seed_base = seed_base,
        iterations = iterations,
        inner_iterations = inner_iterations,
        tao = tao,
        angle_conv = angle_conv,
        init_mode = init_mode,
        compute_hessian = compute_hessian,
        hessian_tol = hessian_tol,
    )

    fig = Figure(size = (900, 360), figure_padding = (8, 8, 8, 8))

    xticks_to025 = (0.0:0.05:0.25, [@sprintf("%.2f", x) for x in 0.0:0.05:0.25])

    ax1 = Axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"B(\lambda)=\frac{1}{N}\sum_i \left|\langle \mathrm{sign}(\theta_i-\pi/4)\rangle\right|",
        xticks = xticks_to025,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 16,
        ylabelsize = 16,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    mask1 = isfinite.(λvals) .& isfinite.(Bvals)
    scatterlines!(ax1, λvals[mask1], Bvals[mask1];
        marker = :circle,
        markersize = 8
    )

    ax2 = Axis(fig[1, 2];
        xlabel = L"\lambda",
        ylabel = L"\left\langle |\theta-\pi/4| \right\rangle_{\mathrm{raw}}",
        xticks = xticks_to025,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 16,
        ylabelsize = 16,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    mask2 = isfinite.(λvals) .& isfinite.(mean_abs_delta_vals)
    scatterlines!(ax2, λvals[mask2], mean_abs_delta_vals[mask2];
        marker = :circle,
        markersize = 8
    )

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.01, 0.93), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.51, 0.93), space=:relative, fontsize=18)

    conv_tag = replace(@sprintf("%.0e", angle_conv), "." => "p")
    out_png = joinpath(
        PLOTS_DIR,
        "threshold_sign_bias_lambda_N$(N)_k$(k)_seed$(graph_seed)_ninit$(n_inits)_conv$(conv_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()