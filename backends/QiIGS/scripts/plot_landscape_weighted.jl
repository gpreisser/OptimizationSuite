# scripts/plot_landscape_weighted.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using CSV
using DataFrames
using Statistics
using Printf
using LaTeXStrings

try
    set_theme!(theme_latexfonts())
catch
end

# ------------------------------------------------------------------------------
# Stable paths
# ------------------------------------------------------------------------------

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

finite_mask(x) = .!isnan.(x) .&& .!isinf.(x)
finite_positive_mask(x) = finite_mask(x) .&& (x .> 0)

function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function log_normalise(x::AbstractVector)
    vals = Float64.(x)
    vals = vals[finite_positive_mask(vals)]
    isempty(vals) && return Float64[]
    lv = log.(vals)
    span = maximum(lv) - minimum(lv)
    iszero(span) && return zeros(length(vals))
    return (lv .- minimum(lv)) ./ span
end

function normalise_with_reference(values::AbstractVector, reference::AbstractVector)
    val = Float64.(values)
    ref = Float64.(reference)

    val = val[finite_positive_mask(val)]
    ref = ref[finite_positive_mask(ref)]

    isempty(val) && return Float64[]
    isempty(ref) && return log_normalise(val)

    lref = log.(ref)
    ref_min = minimum(lref)
    ref_span = maximum(lref) - ref_min

    iszero(ref_span) && return zeros(length(val))
    return (log.(val) .- ref_min) ./ ref_span
end

function denormalised_log_ticks(norm_ticks::AbstractVector, values::AbstractVector)
    vals = Float64.(values)
    vals = vals[finite_positive_mask(vals)]

    if isempty(vals)
        return ["" for _ in norm_ticks]
    end

    lv = log.(vals)
    lv_min = minimum(lv)
    lv_span = maximum(lv) - lv_min

    mapped = if iszero(lv_span)
        fill(exp(lv_min), length(norm_ticks))
    else
        exp.(lv_min .+ norm_ticks .* lv_span)
    end

    return [@sprintf("%d", round(Int, v)) for v in mapped]
end

function styled_axis(figpos; xlabel="", ylabel="", title="", yscale=identity)
    Axis(
        figpos;
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        xlabelsize = 14,
        ylabelsize = 14,
        titlesize = 14,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
        yscale = yscale,
    )
end

function newest_matching_json(dir::AbstractString; must_contain::Vector{String}=String[])
    isdir(dir) || error("Directory not found: $dir")
    files = filter(f -> endswith(f, ".json") && !occursin("checkpoint_", f), readdir(dir))
    for token in must_contain
        files = filter(f -> occursin(token, f), files)
    end
    isempty(files) && error("No matching JSON found in $dir with filters $(must_contain)")
    paths = [joinpath(dir, f) for f in files]
    sort!(paths, by = p -> stat(p).mtime)
    return paths[end]
end

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

function main()
    # --------------------------------------------------------------------------
    # User parameters
    # --------------------------------------------------------------------------
    N = 50
    k = 3
    weighted = true

    ngraphs = 100
    n_inits = 10000
    iterations = 1
    inner_iterations = 5000
    angle_conv = 1e-8
    angle_bin = 1e-2

    compute_hessian = true
    opt_tol = 1e-10

    max_condition_number = 100.0

    # Closest analogue to your old minima count
    minima_key = "unique_angle_count_reduced_mean"
    minima_err_key = "unique_angle_count_reduced_stderr"
    minima_label = L"\langle N_{\mathrm{min}} \rangle"

    wtag = weighted ? "weighted" : "unweighted"
    conv_tag = replace(@sprintf("%.3g", angle_conv), "." => "p")
    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    opttol_tag = replace(@sprintf("%.3g", opt_tol), "." => "p")

    save_dir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(ngraphs)_$(wtag)"
    )

    optmin_dir = joinpath(
        save_dir,
        "optimal_minima_hesscond_tol$(opttol_tag)"
    )

    json_path = newest_matching_json(
        save_dir;
        must_contain = [
            "conv$(conv_tag)",
            "abin$(abin_tag)",
            compute_hessian ? "hess_on" : "hess_off",
            "ngraphs$(ngraphs)",
            "ninit$(n_inits)",
            "outer$(iterations)",
            "inner$(inner_iterations)",
        ]
    )

    output_overview_pdf = joinpath(PLOTS_DIR, "qiigs_weighted_avg_cond_overview_N$(N)_k$(k).pdf")
    output_overview_png = joinpath(PLOTS_DIR, "qiigs_weighted_avg_cond_overview_N$(N)_k$(k).png")

    output_comparison_pdf = joinpath(PLOTS_DIR, "qiigs_weighted_avg_cond_comparison_N$(N)_k$(k).pdf")
    output_comparison_png = joinpath(PLOTS_DIR, "qiigs_weighted_avg_cond_comparison_N$(N)_k$(k).png")

    println("Using aggregated JSON: ", json_path)
    println("Using solved-case CSV dir: ", optmin_dir)

    # --------------------------------------------------------------------------
    # Load aggregated JSON
    # --------------------------------------------------------------------------
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    lambdas = [Float64(entry["λ"]) for entry in rpl]

    mean_minima = [getnum(entry, minima_key) for entry in rpl]
    mean_minima_err = [getnum(entry, minima_err_key) for entry in rpl]

    mean_cond_all = [getnum(entry, "hess_cond_mean_mean") for entry in rpl]
    mean_cond_all_err = [getnum(entry, "hess_cond_mean_stderr") for entry in rpl]

    # --------------------------------------------------------------------------
    # Load solved-case raw condition numbers from per-graph CSVs
    # --------------------------------------------------------------------------
    raw_cond_lambdas = Float64[]
    raw_cond_values = Float64[]

    isdir(optmin_dir) || error("Solved-case CSV directory not found: $optmin_dir")

    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(optmin_dir)))
    isempty(csv_files) && error("No CSV files found in $optmin_dir")

    for f in csv_files
        path = joinpath(optmin_dir, f)
        df = CSV.read(path, DataFrame)
        colnames = names(df)

        if !all(["lambda", "hess_cond"] .∈ Ref(colnames))
            @warn "Skipping $(path): missing expected columns"
            continue
        end

        λv = Float64.(df[!, "lambda"])
        κv = Float64.(df[!, "hess_cond"])

        mask = finite_positive_mask(κv)
        if any(mask)
            append!(raw_cond_lambdas, λv[mask])
            append!(raw_cond_values, κv[mask])
        end
    end

    if !isempty(raw_cond_values)
        keep = finite_mask(raw_cond_values) .&& (raw_cond_values .<= max_condition_number)
        dropped = length(raw_cond_values) - count(keep)
        raw_cond_lambdas = raw_cond_lambdas[keep]
        raw_cond_values = raw_cond_values[keep]
        println("Applied max_condition_number=$(max_condition_number): dropped $(dropped) solved-case condition numbers")
    end

    # Mean κ* per λ from the solved-case CSVs
    solved_cond_mean = fill(NaN, length(lambdas))
    solved_cond_err = fill(NaN, length(lambdas))

    for (i, λ) in enumerate(lambdas)
        vals = raw_cond_values[abs.(raw_cond_lambdas .- λ) .< 1e-12]
        vals = vals[finite_positive_mask(vals)]
        if isempty(vals)
            continue
        end
        solved_cond_mean[i] = mean(vals)
        solved_cond_err[i] = length(vals) > 1 ? std(vals; corrected=true) / sqrt(length(vals)) : 0.0
    end

    # --------------------------------------------------------------------------
    # Figure 1: overview
    # --------------------------------------------------------------------------
    fig1 = Figure(size = (900, 780))

    axa = styled_axis(
        fig1[1, 1];
        ylabel = "Number of minima",
        title = "QiIGS weighted landscape summary  (N = $(N), k = $(k), ngraphs = $(ngraphs), ninit = $(n_inits))",
        yscale = log10,
    )

    mN = finite_positive_mask(mean_minima)
    eN = copy(mean_minima_err)
    eN[.!finite_mask(eN)] .= 0.0
    mN .= mN .&& ((mean_minima .- eN) .> 0)

    if any(mN)
        errorbars!(axa, lambdas[mN], mean_minima[mN], eN[mN];
            whiskerwidth = 8,
            color = :black
        )
        lines!(axa, lambdas[mN], mean_minima[mN];
            color = :black,
            linewidth = 2.0
        )
        scatter!(axa, lambdas[mN], mean_minima[mN];
            color = :black,
            marker = :diamond,
            markersize = 8
        )
    end

    axb = styled_axis(
        fig1[2, 1];
        xlabel = L"\lambda",
        ylabel = "Hessian condition number",
        yscale = log10,
    )

    if !isempty(raw_cond_values)
        scatter!(axb,
            raw_cond_lambdas, raw_cond_values;
            color = (Makie.wong_colors()[1], 0.10),
            markersize = 4.5,
            label = L"\kappa^* \ \mathrm{(solved\ cases)}"
        )
    end

    m_all = finite_positive_mask(mean_cond_all)
    e_all = copy(mean_cond_all_err)
    e_all[.!finite_mask(e_all)] .= 0.0
    m_all .= m_all .&& ((mean_cond_all .- e_all) .> 0)

    if any(m_all)
        errorbars!(axb, lambdas[m_all], mean_cond_all[m_all], e_all[m_all];
            whiskerwidth = 8,
            color = :black
        )
        lines!(axb, lambdas[m_all], mean_cond_all[m_all];
            color = :black,
            linewidth = 2.0,
            label = "mean cond (all minima)"
        )
        scatter!(axb, lambdas[m_all], mean_cond_all[m_all];
            color = :black,
            markersize = 7
        )
    end

    m_sol = finite_positive_mask(solved_cond_mean)
    e_sol = copy(solved_cond_err)
    e_sol[.!finite_mask(e_sol)] .= 0.0
    m_sol .= m_sol .&& ((solved_cond_mean .- e_sol) .> 0)

    if any(m_sol)
        errorbars!(axb, lambdas[m_sol], solved_cond_mean[m_sol], e_sol[m_sol];
            whiskerwidth = 8,
            color = :purple
        )
        lines!(axb, lambdas[m_sol], solved_cond_mean[m_sol];
            color = :purple,
            linewidth = 2.0,
            linestyle = :dash,
            label = L"\mathrm{mean}\ \kappa^*"
        )
        scatter!(axb, lambdas[m_sol], solved_cond_mean[m_sol];
            color = :purple,
            marker = :rect,
            markersize = 7
        )
    end

    axislegend(axb; position = :rb, framevisible = false)

    for ax in (axa, axb)
        xlims!(ax, -0.05, 1.05)
        ax.xticks = 0.0:0.2:1.0
    end

    Label(fig1[1, 1, TopLeft()], "(a)"; fontsize = 20, padding = (0, 0, 8, 0))
    Label(fig1[2, 1, TopLeft()], "(b)"; fontsize = 20, padding = (0, 0, 8, 0))

    rowgap!(fig1.layout, 18)

    save(output_overview_pdf, fig1)
    save(output_overview_png, fig1)

    println("Saved PDF: ", output_overview_pdf)
    println("Saved PNG: ", output_overview_png)

    display(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: compact comparison
    # --------------------------------------------------------------------------
    fig2 = Figure(
        size = (420, 330),
        tellwidth = false,
        tellheight = false,
        figure_padding = (1e-10, 6, 4, 12),
    )

    ax = Axis(
        fig2[1, 1];
        xlabel = L"\lambda",
        ylabel = L"\kappa",
        xlabelsize = 14,
        ylabelsize = 14,
        xticklabelsize = 14,
        yticklabelsize = 14,
        xgridvisible = false,
        ygridvisible = false,
        xticksmirrored = true,
        yticksmirrored = true,
        xtickalign = 1,
        ytickalign = 1,
        xticks = 0.0:0.2:1.0,
    )

    p_cond = nothing
    if !isempty(raw_cond_values)
        raw_cond_norm = log_normalise(raw_cond_values)
        if !isempty(raw_cond_norm)
            p_cond = scatter!(ax,
                raw_cond_lambdas, raw_cond_norm;
                color = (Makie.wong_colors()[1], 0.10),
                markersize = 4.5,
            )
        end
    end

    p_meanN = nothing
    if any(mN)
        meanN_vals = mean_minima[mN]
        meanN_norm = log_normalise(meanN_vals)

        p_meanN = lines!(ax,
            lambdas[mN], meanN_norm;
            color = :black,
            linewidth = 1.5,
        )
        scatter!(ax,
            lambdas[mN], meanN_norm;
            color = :black,
            marker = :diamond,
            markersize = 8,
        )
    end

    p_meansol = nothing
    if any(m_sol)
        sol_vals = solved_cond_mean[m_sol]
        sol_norm = normalise_with_reference(sol_vals, raw_cond_values)

        p_meansol = lines!(ax,
            lambdas[m_sol], sol_norm;
            color = :purple,
            linewidth = 1.5,
            linestyle = :dash,
        )
        scatter!(ax,
            lambdas[m_sol], sol_norm;
            color = :purple,
            marker = :rect,
            markersize = 7,
        )
    end

    xlims!(ax, -0.05, 1.05)
    ylims!(ax, -0.05, 1.12)

    norm_ticks = collect(range(0.0, 1.0; length = 5))
    ax.yticks = (norm_ticks, denormalised_log_ticks(norm_ticks, raw_cond_values))
    ax.ylabelcolor = Makie.wong_colors()[1]
    ax.yticklabelcolor = Makie.wong_colors()[1]
    ax.ytickcolor = Makie.wong_colors()[1]
    ax.leftspinecolor = Makie.wong_colors()[1]

    axr = Axis(
        fig2[1, 1];
        yaxisposition = :right,
        xgridvisible = false,
        ygridvisible = false,
        xticklabelsvisible = false,
        xticksvisible = false,
        xminorgridvisible = false,
        ytickalign = 1,
        yticksmirrored = true,
        yticklabelsize = 14,
        ylabel = minima_label,
        ylabelsize = 14,
        xticks = 0.0:0.2:1.0,
    )
    hidespines!(axr, :l, :b, :t)
    hidexdecorations!(axr)

    ylims!(axr, -0.05, 1.12)
    axr.yticks = (norm_ticks, denormalised_log_ticks(norm_ticks, mean_minima[mN]))
    axr.ylabelcolor = :gray35
    axr.yticklabelcolor = :gray35
    axr.ytickcolor = :gray35
    axr.rightspinecolor = :gray35

    plots = Plot[]
    labels = Any[]

    if p_cond !== nothing
        push!(plots, scatter!(ax, [NaN], [NaN];
            color = Makie.wong_colors()[1],
            markersize = 7
        ))
        push!(labels, L"\kappa^*")
    end

    if p_meansol !== nothing
        p_leg_sol = scatterlines!(ax, [NaN], [NaN];
            color = :purple,
            marker = :rect,
            linewidth = 1.5,
            linestyle = :dash,
            markersize = 7,
        )
        push!(plots, p_leg_sol)
        push!(labels, L"\langle \kappa^* \rangle")
    end

    if p_meanN !== nothing
        p_leg_N = scatterlines!(ax, [NaN], [NaN];
            color = :black,
            marker = :diamond,
            linewidth = 1.5,
            markersize = 8,
        )
        push!(plots, p_leg_N)
        push!(labels, minima_label)
    end

    axislegend(
        ax, plots, labels;
        position = :lt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 14,
        rowgap = 2,
        padding = (2, 2, 2, 2),
    )

    save(output_comparison_pdf, fig2)
    save(output_comparison_png, fig2)

    println("Saved PDF: ", output_comparison_pdf)
    println("Saved PNG: ", output_comparison_png)

    display(fig2)
end

main()