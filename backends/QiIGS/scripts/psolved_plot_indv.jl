# scripts/plot_psolved_first10.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using LaTeXStrings
using Printf
using Statistics

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(ROOT, "results", "plots")
mkpath(PLOTS_DIR)

const MAIN_TICKSIZE   = 24
const MAIN_LABELSIZE  = 24
const PANEL_LABELSIZE = 24

const LAMBDA_TRANSITION = 0.25

# ------------------------------------------------------------
# basic helpers
# ------------------------------------------------------------
function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v === nothing || v == "none") && return NaN
    return Float64(v)
end

finite_mask(x) = .!isnan.(x) .&& .!isinf.(x)

function conv_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function tao_tag_to_float(tag::AbstractString)
    parse(Float64, replace(tag, "p" => "."))
end

function extract_conv_from_filename(path::AbstractString)
    b = basename(path)
    m = match(r"_conv([^_]+)_abin", b)
    m === nothing && return nothing
    try
        return conv_tag_to_float(m.captures[1])
    catch
        return nothing
    end
end

function extract_tao_from_filename(path::AbstractString)
    b = basename(path)
    m = match(r"_tao([^_]+)_conv", b)
    m === nothing && return nothing
    try
        return tao_tag_to_float(m.captures[1])
    catch
        return nothing
    end
end

function json_tao(path::AbstractString)
    data = JSON.parsefile(path)

    if haskey(data, "tao")
        v = data["tao"]
        if !(v === nothing || v == "none")
            return Float64(v)
        end
    end

    if haskey(data, "solver")
        solver = data["solver"]
        if solver isa AbstractDict && haskey(solver, "tao")
            v = solver["tao"]
            if !(v === nothing || v == "none")
                return Float64(v)
            end
        end
    end

    return nothing
end

function effective_tao(path::AbstractString)
    tf = extract_tao_from_filename(path)
    tf !== nothing && return tf
    return json_tao(path)
end

function json_is_new_like(path::AbstractString)
    data = JSON.parsefile(path)

    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false

    entry1 = rpl[1]

    return haskey(entry1, "success_rate_mean") &&
           haskey(entry1, "success_rate_stderr") &&
           haskey(entry1, "opt_min_probs_by_graph")
end

function list_matching_jsons(; N, k, ngraphs, n_inits, angle_bin, require_hessian=true)
    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")

    subdir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(ngraphs)_unweighted"
    )
    isdir(subdir) || error("Directory not found: $subdir")

    files = readdir(subdir; join=true)

    prefix = "qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d"
    abin_piece = "_abin$(abin_tag)_"
    ninit_piece = "_ngraphs$(ngraphs)_ninit$(n_inits)_outer1_inner5000_thr0p999.json"

    matches = filter(files) do f
        b = basename(f)
        ok = startswith(b, prefix) &&
             occursin(abin_piece, b) &&
             occursin(ninit_piece, b) &&
             !occursin("checkpoint", lowercase(b))
        if require_hessian
            ok &= occursin("_hess_on_", b)
        end
        ok
    end

    matches = filter(json_is_new_like, matches)

    isempty(matches) && error("No matching aggregated JSON found in:\n$subdir")
    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, tao, require_hessian=true)
    matches = list_matching_jsons(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        require_hessian = require_hessian,
    )

    tol = 1e-14

    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error("No matching aggregated JSON found for conv=$(angle_conv).")

    explicit_match = String[]
    legacy_unknown = String[]

    println("------------------------------------------------------------")
    println("Candidate files after conv filter:")
    for f in conv_matches
        tf = extract_tao_from_filename(f)
        tj = json_tao(f)
        te = effective_tao(f)
        println("file = ", basename(f))
        println("  tao from filename = ", tf)
        println("  tao from json     = ", tj)
        println("  effective tao     = ", te)

        if te === nothing
            push!(legacy_unknown, f)
        elseif isapprox(te, tao; atol=tol, rtol=0.0)
            push!(explicit_match, f)
        else
            println("  -> rejected (different tao)")
        end
    end
    println("------------------------------------------------------------")

    chosen_pool =
        if !isempty(explicit_match)
            println("Found file(s) with tao=$(tao).")
            explicit_match
        elseif !isempty(legacy_unknown)
            println("No file with explicit tao=$(tao) found.")
            println("Falling back only to legacy files with no tao in filename and no tao in JSON.")
            legacy_unknown
        else
            error("No matching aggregated JSON found for conv=$(angle_conv) and tao=$(tao).")
        end

    mtimes = [stat(f).mtime for f in chosen_pool]
    chosen = chosen_pool[argmax(mtimes)]

    println("Chosen file: ", chosen)
    println("Chosen effective tao: ", effective_tao(chosen))

    return chosen
end

# ------------------------------------------------------------
# per-graph psolved extraction
# ------------------------------------------------------------
function scalar_or_nan(x)
    if x === nothing || x == "none"
        return NaN
    end
    try
        return Float64(x)
    catch
        return NaN
    end
end

function numeric_vector(v)
    out = Float64[]
    v isa AbstractVector || return out

    for x in v
        y = scalar_or_nan(x)
        if isfinite(y)
            push!(out, y)
        end
    end

    return out
end

# Handles both:
#   - flat vector: [0.1, 0.0, 0.7, ...]
#   - vector of vectors: [[0.02, 0.03], [], [0.7], ...]
# In the second case, each graph gets the sum of its optimal-minimum probabilities.
function probs_by_graph_to_psolved(v)
    out = Float64[]
    v isa AbstractVector || return out

    for item in v
        if item === nothing || item == "none"
            push!(out, 0.0)
        elseif item isa AbstractVector
            vals = numeric_vector(item)
            push!(out, isempty(vals) ? 0.0 : sum(vals))
        else
            y = scalar_or_nan(item)
            push!(out, isfinite(y) ? y : 0.0)
        end
    end

    return out
end

function get_per_graph_psolved(entry)
    haskey(entry, "opt_min_probs_by_graph") || return Float64[]
    return probs_by_graph_to_psolved(entry["opt_min_probs_by_graph"])
end

function load_psolved_matrix(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    mean_psolved = Float64[]
    stderr_psolved = Float64[]
    per_λ_vectors = Vector{Vector{Float64}}()

    for (idx, entry) in enumerate(rpl)
        λ = getnum(entry, "λ")
        μ = getnum(entry, "success_rate_mean")
        σ = getnum(entry, "success_rate_stderr")
        vals = get_per_graph_psolved(entry)

        if !isfinite(λ)
            continue
        end

        isempty(vals) && continue

        push!(λs, λ)
        push!(mean_psolved, μ)
        push!(stderr_psolved, σ)
        push!(per_λ_vectors, vals)

        if idx == 1
            println("------------------------------------------------------------")
            println("First λ entry diagnostics:")
            println("λ = ", λ)
            println("number of per-graph points = ", length(vals))
            println("first few per-graph psolved values:")
            for i in 1:min(10, length(vals))
                @printf("  %d  %.8f\n", i, vals[i])
            end
            println("------------------------------------------------------------")
        end
    end

    isempty(λs) && error("No usable λ entries found.")
    ngraphs = length(per_λ_vectors[1])

    for (i, vals) in enumerate(per_λ_vectors)
        length(vals) == ngraphs || error("Inconsistent number of graphs at λ index $i: expected $ngraphs, got $(length(vals))")
    end

    p = sortperm(λs)
    λs = λs[p]
    mean_psolved = mean_psolved[p]
    stderr_psolved = stderr_psolved[p]
    per_λ_vectors = per_λ_vectors[p]

    nλ = length(λs)
    P = zeros(Float64, ngraphs, nλ)

    for iλ in 1:nλ
        @inbounds for g in 1:ngraphs
            P[g, iλ] = per_λ_vectors[iλ][g]
        end
    end

    return (
        λs = λs,
        mean_psolved = mean_psolved,
        stderr_psolved = stderr_psolved,
        P = P,
        ngraphs = ngraphs,
        nλ = nλ,
    )
end

# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
function styled_axis(figpos; xlabel, ylabel, xticks=nothing, kwargs...)
    common = (;
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = MAIN_TICKSIZE,
        yticklabelsize = MAIN_TICKSIZE,
        xlabelsize = MAIN_LABELSIZE,
        ylabelsize = MAIN_LABELSIZE,
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
        xticksmirrored = true,
        yticksmirrored = true,
    )

    if xticks === nothing
        return Axis(figpos; common..., kwargs...)
    else
        return Axis(figpos; xticks = xticks, common..., kwargs...)
    end
end

function plot_psolved_first_graphs(data;
    ngraphs_plot=10,
    N=50,
    k=3,
    ngraphs=100,
    n_inits=10000,
    angle_bin=0.01,
    angle_conv=1e-8)

    fig = Figure(
        size = (760, 540),
        tellwidth = false,
        tellheight = false,
        figure_padding = (8, 8, 8, 8),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    ax = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"P_{\mathrm{solved}}",
        xticks = xticks_to1,
        yticks = (0.0:0.2:1.0, [@sprintf("%.1f", y) for y in 0.0:0.2:1.0]),
    )

    ylims!(ax, -0.02, 1.02)

    colors = Makie.wong_colors()
    ng = min(ngraphs_plot, data.ngraphs)

    for g in 1:ng
        c = colors[mod1(g, length(colors))]
        lines!(ax,
            data.λs,
            view(data.P, g, :);
            color = c,
            linewidth = 2.2,
            label = "graph $g",
        )
    end

    # Optional mean overlay
    errorbars!(ax,
        data.λs,
        data.mean_psolved,
        data.stderr_psolved;
        color = (:black, 0.7),
        whiskerwidth = 6
    )

    lines!(ax,
        data.λs,
        data.mean_psolved;
        color = :black,
        linewidth = 3.0,
    )

    vlines!(ax, [LAMBDA_TRANSITION];
        color = :gray,
        linestyle = :dash,
        linewidth = 2
    )

    axislegend(ax;
        position = :rb,
        labelsize = 14,
        framevisible = false,
        nbanks = 2,
    )

    text!(fig.scene, "(c)"; position = (0.02, 0.97), space = :relative, fontsize = PANEL_LABELSIZE)

    resize_to_layout!(fig)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_tag = replace(@sprintf("%.0e", angle_conv), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "psolved_first$(ng)_instances_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_conv$(conv_tag).png"
    )

    out_pdf = joinpath(
        PLOTS_DIR,
        "psolved_first$(ng)_instances_unweighted.pdf"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    save(out_pdf, fig)
    @info "Saved $out_pdf"

    display(fig)
    return fig
end

function main(; N=50, k=3, ngraphs=100, n_inits=10000, angle_bin=0.01, angle_conv=1e-8, tao=0.1, ngraphs_plot=10)
    json_path = newest_matching_json(;
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        angle_conv = angle_conv,
        tao = tao,
        require_hessian = true,
    )

    println("============================================================")
    println("Loading psolved curves from:")
    println(json_path)
    println("============================================================")

    data = load_psolved_matrix(json_path)

    println("nλ               = ", data.nλ)
    println("ngraphs detected  = ", data.ngraphs)
    println("plotting first    = ", min(ngraphs_plot, data.ngraphs), " instances")

    return plot_psolved_first_graphs(data;
        ngraphs_plot = ngraphs_plot,
        N = N,
        k = k,
        ngraphs = ngraphs,
        n_inits = n_inits,
        angle_bin = angle_bin,
        angle_conv = angle_conv,
    )
end

# ------------------------------------------------------------
# run
# ------------------------------------------------------------
fig = main(;
    N = 50,
    k = 3,
    ngraphs = 100,
    n_inits = 10000,
    angle_bin = 0.01,
    angle_conv = 1e-8,
    tao = 0.1,
    ngraphs_plot = 10,
)