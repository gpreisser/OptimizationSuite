using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using CairoMakie
using LaTeXStrings
using Printf
using SparseArrays

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

function styled_axis(figpos; xlabel, ylabel, xticks=nothing)
    common = (;
        xlabel = xlabel,
        ylabel = ylabel,
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

    xticks === nothing ? Axis(figpos; common...) : Axis(figpos; xticks=xticks, common...)
end

function main()
    # ------------------------------------------------------------
    # User parameters
    # ------------------------------------------------------------
    N = 50
    k = 3
    weighted = false
    graph_seed = 1

    λs = collect(0.0:0.025:1.0)

    n_inits = 10_000
    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    init_mode = :uniform
    save_params = true

    angle_conv = 1e-8
    compute_hessian = true
    hessian_tol = 1e-8

    seed_salt = 0

    # ------------------------------------------------------------
    # Input paths
    # ------------------------------------------------------------
    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    spath = QiIGS.akmax_solution_path(N, k, graph_seed; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
    opt = QiIGS.load_optimal_cut(spath)
    opt === nothing && error("No optimal cut found at: $spath")

    # ------------------------------------------------------------
    # Sweep over lambda
    # ------------------------------------------------------------
    dom_frac = Float64[]
    dom_ratio = Float64[]

    for λ in λs
        counts = Dict{UInt64, Int}()
        rep_ratio = Dict{UInt64, Float64}()

        for r in 1:n_inits
            run_seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ * 1000)) + seed_salt * 1_000_000_000

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
                save_params = save_params,
                progressbar = false,
                compute_hessian = compute_hessian,
                hessian_tol = hessian_tol,
            )

            key = QiIGS.spin_config_key(sol.configuration)
            counts[key] = get(counts, key, 0) + 1

            if !haskey(rep_ratio, key)
                _, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
                rep_ratio[key] = Float64(ratio)
            end
        end

        pairs_sorted = sort(collect(counts); by = x -> x[2], rev = true)
        top_key, top_count = pairs_sorted[1]

        push!(dom_frac, top_count / n_inits)
        push!(dom_ratio, rep_ratio[top_key])

        @printf("λ=%.3f   p_dom=%.6f   r_dom=%.6f\n",
            λ, dom_frac[end], dom_ratio[end])
    end

    # ------------------------------------------------------------
    # Color scale: ignore λ = 0 and λ = 1 endpoints
    # ------------------------------------------------------------
    mask_color = (λs .!= 0.0) .& (λs .!= 1.0) .& isfinite.(dom_ratio)
    any(mask_color) || error("No valid interior values available for color scaling.")

    ratio_min = minimum(dom_ratio[mask_color])
    ratio_max = maximum(dom_ratio[mask_color])

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig = Figure(
        size = (700, 360),
        tellwidth = false,
        tellheight = false,
        figure_padding = (6, 6, 6, 6),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    ax = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"p_{\mathrm{dom}}",
        xticks = xticks_to1
    )

    hm = scatter!(ax,
        λs, dom_frac;
        color = dom_ratio,
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 10
    )

    Colorbar(fig[1, 2], hm;
        label = L"r_{\mathrm{dom}}",
        width = 14,
        ticklabelsize = 14,
        labelsize = 15
    )
    colsize!(fig.layout, 2, Auto(22))

    resize_to_layout!(fig)

    conv_tag = replace(@sprintf("%.0e", angle_conv), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "dominant_basin_vs_lambda_N$(N)_k$(k)_seed$(graph_seed)_ninit$(n_inits)_conv$(conv_tag).png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()