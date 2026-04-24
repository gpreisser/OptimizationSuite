using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using CairoMakie
using JSON
using LaTeXStrings
using Printf

try
    set_theme!(theme_latexfonts())
catch
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const PLOTS_DIR   = joinpath(RESULTS_DIR, "plots")
mkpath(PLOTS_DIR)

function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function json_is_v7_like(path::AbstractString)
    data = JSON.parsefile(path)

    haskey(data, "results_per_lambda") || return false
    rpl = data["results_per_lambda"]
    isempty(rpl) && return false

    entry1 = rpl[1]

    # Require the new per-λ diagnostics to exist
    haskey(entry1, "devTheta_abs_mean") || return false
    haskey(entry1, "devTheta_abs_stderr") || return false
    haskey(entry1, "devZ_abs_mean") || return false
    haskey(entry1, "devZ_abs_stderr") || return false

    # Also require the solver outputs metadata to mention them
    haskey(data, "solver") || return false
    solver = data["solver"]
    haskey(solver, "outputs") || return false
    outs = solver["outputs"]

    return ("devTheta_abs" in outs) && ("devZ_abs" in outs)
end

function load_curves(json_path::AbstractString)
    data = JSON.parsefile(json_path)
    rpl = data["results_per_lambda"]

    λs = Float64[]
    unique_angle_raw = Float64[]
    hess_mineig = Float64[]
    mean_ratio = Float64[]
    succ_rate = Float64[]
    devTheta_abs = Float64[]

    for entry in rpl
        λ   = getnum(entry, "λ")
        ua  = getnum(entry, "unique_angle_count_raw_mean")
        hm  = getnum(entry, "hess_mineig_mean_mean")
        mr  = getnum(entry, "mean_ratio_mean")
        sr  = getnum(entry, "success_rate_mean")
        dθ  = getnum(entry, "devTheta_abs_mean")

        if !isfinite(λ)
            continue
        end

        push!(λs, λ)
        push!(unique_angle_raw, ua)
        push!(hess_mineig, hm)
        push!(mean_ratio, mr)
        push!(succ_rate, sr)
        push!(devTheta_abs, dθ)
    end

    p = sortperm(λs)

    return (
        λs[p],
        unique_angle_raw[p],
        hess_mineig[p],
        mean_ratio[p],
        succ_rate[p],
        devTheta_abs[p],
    )
end

function load_success_curve(json_path::AbstractString)
    λs, _, _, _, succ_rate, _ = load_curves(json_path)
    mask = isfinite.(λs) .& isfinite.(succ_rate)
    return λs[mask], succ_rate[mask]
end

function styled_axis(figpos; xlabel, ylabel, xticks=nothing, kwargs...)
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

    if xticks === nothing
        return Axis(figpos; common..., kwargs...)
    else
        return Axis(figpos; xticks=xticks, common..., kwargs...)
    end
end

function conv_tag_to_float(tag::AbstractString)
    s = replace(tag, "p" => ".")
    return parse(Float64, s)
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
        return ok
    end

    # Force all selected files to be from the new data family
    matches = filter(json_is_v7_like, matches)

    isempty(matches) && error(
        "No matching v7-like aggregated JSON found in:\n$subdir\n" *
        "Check angle_bin and whether the updated sweep was rerun."
    )

    return matches
end

function newest_matching_json(; N, k, ngraphs, n_inits, angle_bin, angle_conv, require_hessian=true)
    matches = list_matching_jsons(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        require_hessian=require_hessian,
    )

    tol = 1e-14
    conv_matches = filter(matches) do f
        c = extract_conv_from_filename(f)
        c !== nothing && isapprox(c, angle_conv; atol=tol, rtol=0.0)
    end

    isempty(conv_matches) && error(
        "No matching v7-like aggregated JSON found for conv=$(angle_conv).\n" *
        "Available conv values: " *
        join(sort(unique(filter(!isnothing, extract_conv_from_filename.(matches)))), ", ")
    )

    mtimes = [stat(f).mtime for f in conv_matches]
    return conv_matches[argmax(mtimes)]
end

function conv_label(conv::Float64)
    if isapprox(conv, 1e-2; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-2}"
    elseif isapprox(conv, 1e-4; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-4}"
    elseif isapprox(conv, 1e-6; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-6}"
    elseif isapprox(conv, 1e-8; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-8}"
    elseif isapprox(conv, 1e-10; atol=0.0, rtol=0.0)
        return L"\mathrm{conv}=10^{-10}"
    else
        error("Unsupported conv = $conv")
    end
end

function main()
    N = 50
    k = 3
    ngraphs = 1
    n_inits = 10000
    angle_bin = 0.01

    angle_conv_main = 1e-8

    panelc_convs   = [1e-2, 1e-6, 1e-8]
    panelc_markers = [:dtriangle, :utriangle, :circle]
    panelc_colors  = ["#E69F00", "purple", "#0072B2"]

    path_main = newest_matching_json(;
        N=N,
        k=k,
        ngraphs=ngraphs,
        n_inits=n_inits,
        angle_bin=angle_bin,
        angle_conv=angle_conv_main,
        require_hessian=true,
    )

    @info "Using main path" path_main

    λs, uniques_angle, hess_mineig, mean_ratio,
    succ_rate_main, devTheta_abs = load_curves(path_main)

    isempty(λs) && error("No usable λ entries found in JSON.")

    mask_a = isfinite.(λs) .&
             isfinite.(uniques_angle) .&
             isfinite.(mean_ratio) .&
             (uniques_angle .> 0.0) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_b = isfinite.(λs) .&
             isfinite.(hess_mineig) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    mask_d = isfinite.(λs) .&
             isfinite.(devTheta_abs) .&
             (λs .>= 0.0) .&
             (λs .<= 1.0)

    any(mask_a) || error("No usable positive entries for panel (a).")
    any(mask_b) || error("No usable entries for panel (b).")
    any(mask_d) || error("No usable entries for panel (d).")

    mask_ratio = isfinite.(mean_ratio) .&
                 isfinite.(λs) .&
                 (λs .!= 0.0) .&
                 (λs .!= 1.0)

    any(mask_ratio) || error("No finite mean_ratio values available for color mapping.")
    ratio_min = minimum(mean_ratio[mask_ratio])
    ratio_max = maximum(mean_ratio[mask_ratio])

    fig = Figure(
        size = (1550, 320),
        tellwidth = false,
        tellheight = false,
        figure_padding = (6, 6, 6, 6),
    )

    xticks_to1 = (0.0:0.2:1.0, [@sprintf("%.1f", x) for x in 0.0:0.2:1.0])

    axa = styled_axis(fig[1, 1];
        xlabel = L"\lambda",
        ylabel = L"N_{\mathrm{minima}}",
        xticks = xticks_to1,
        yscale = log10,
    )

    hm = scatter!(axa,
        λs[mask_a], uniques_angle[mask_a];
        color = mean_ratio[mask_a],
        colormap = :viridis,
        colorrange = (ratio_min, ratio_max),
        markersize = 8
    )

    Colorbar(fig[1, 2], hm;
        label = L"\langle r\rangle",
        width = 14,
        ticklabelsize = 14,
        labelsize = 15
    )
    colsize!(fig.layout, 2, Auto(22))

    axc = styled_axis(fig[1, 3];
        xlabel = L"\lambda",
        ylabel = L"P(\mathrm{success})",
        xticks = xticks_to1
    )

    plots_c = Any[]
    labels_c = Any[]
    marksize1 = 9

    for (conv, marker, col) in zip(panelc_convs, panelc_markers, panelc_colors)
        path_c = newest_matching_json(;
            N=N,
            k=k,
            ngraphs=ngraphs,
            n_inits=n_inits,
            angle_bin=angle_bin,
            angle_conv=conv,
            require_hessian=true,
        )

        @info "Using panel (c) path" conv path_c

        λc, src = load_success_curve(path_c)
        isempty(λc) && error("No usable success-rate entries for conv=$(conv).")

        pc = scatterlines!(axc,
            λc, src;
            marker = marker,
            color = col,
            markersize = marksize1
        )

        push!(plots_c, pc)
        push!(labels_c, conv_label(conv))
    end

    axislegend(axc, plots_c, labels_c;
        position = :rt,
        framevisible = false,
        backgroundcolor = :transparent,
        labelsize = 12
    )

    axb = styled_axis(fig[1, 4];
        xlabel = L"\lambda",
        ylabel = L"\langle \lambda_{\min}(\nabla^2 f_\lambda) \rangle",
        xticks = xticks_to1
    )

    scatterlines!(axb,
        λs[mask_b], hess_mineig[mask_b];
        markersize = 8,
        marker = :circle
    )

    axd = styled_axis(fig[1, 5];
        xlabel = L"\lambda",
        ylabel = L"\left\langle \left|\theta_{\mathrm{eff}}-\frac{\pi}{4}\right| \right\rangle",
        xticks = xticks_to1
    )

    scatterlines!(axd,
        λs[mask_d], devTheta_abs[mask_d];
        markersize = 8,
        marker = :circle
    )

    resize_to_layout!(fig)

    text!(fig.scene, "(a)"; position=(0.00, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(b)"; position=(0.30, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(c)"; position=(0.535, 0.92), space=:relative, fontsize=18)
    text!(fig.scene, "(d)"; position=(0.77, 0.92), space=:relative, fontsize=18)

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    conv_main_tag = replace(@sprintf("%.0e", angle_conv_main), "." => "p")

    out_png = joinpath(
        PLOTS_DIR,
        "qiigs_acbd_hessianb_devtheta_unweighted_N$(N)_k$(k)_ngraphs$(ngraphs)_ninit$(n_inits)_abin$(abin_tag)_mainconv$(conv_main_tag)_panelc_multiconv.png"
    )

    save(out_png, fig)
    @info "Saved $out_png"

    display(fig)
end

main()