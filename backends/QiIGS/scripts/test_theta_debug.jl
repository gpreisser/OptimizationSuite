# scripts/test_devtheta_pipeline.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using QiIGS
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")

function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v == "none" || v === nothing) && return NaN
    return Float64(v)
end

function test_json_file(path::AbstractString)
    println("============================================================")
    println("TEST 1: aggregated JSON contents")
    println("============================================================")
    println("Path:")
    println(path)
    println()

    @assert isfile(path) "JSON file not found: $path"

    data = JSON.parsefile(path)
    @assert haskey(data, "results_per_lambda") "Missing key results_per_lambda"

    rpl = data["results_per_lambda"]
    @assert !isempty(rpl) "results_per_lambda is empty"

    first_entry = rpl[1]

    println("Top-level keys:")
    println(collect(keys(data)))
    println()

    println("First λ-entry keys:")
    println(collect(keys(first_entry)))
    println()

    has_devtheta = haskey(first_entry, "devTheta_abs_mean")
    has_devz     = haskey(first_entry, "devZ_abs_mean")

    println("Has devTheta_abs_mean? ", has_devtheta)
    println("Has devZ_abs_mean?     ", has_devz)
    println()

    λvals = Float64[]
    devtheta = Float64[]
    devz = Float64[]

    for entry in rpl
        push!(λvals, getnum(entry, "λ"))
        push!(devtheta, getnum(entry, "devTheta_abs_mean"))
        push!(devz, getnum(entry, "devZ_abs_mean"))
    end

    nθ_finite = count(isfinite, devtheta)
    nz_finite = count(isfinite, devz)

    println("Number of λ points: ", length(λvals))
    println("Finite devTheta_abs_mean values: ", nθ_finite)
    println("Finite devZ_abs_mean values:     ", nz_finite)
    println()

    println("First 10 (λ, devTheta_abs_mean, devZ_abs_mean):")
    for i in 1:min(10, length(λvals))
        @printf("  λ = %.3f   devTheta = %-12.6g   devZ = %-12.6g\n",
            λvals[i], devtheta[i], devz[i])
    end
    println()

    if nθ_finite == 0
        println("WARNING: No finite devTheta_abs_mean values found in JSON.")
    end
    if nz_finite == 0
        println("WARNING: No finite devZ_abs_mean values found in JSON.")
    end
    println()
end

function test_solver_metadata(; N=50, k=3, gs=1, λ=0.5)
    println("============================================================")
    println("TEST 2: direct solver metadata")
    println("============================================================")

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    gpath = QiIGS.graph_path(N, k, gs; weighted=false, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"

    W = QiIGS.load_weight_matrix(gpath)

    sol = QiIGS.solve!(
        W, N;
        solver = :grad,
        seed = 123456,
        lambda = λ,
        iterations = 1,
        inner_iterations = 200,
        tao = 0.1,
        angle_conv = 1e-8,
        init_mode = :uniform,
        save_params = true,
        progressbar = false,
        compute_hessian = true,
        hessian_tol = 1e-8,
    )

    md = sol.metadata

    println("Metadata keys:")
    println(collect(keys(md)))
    println()

    println("Has :devTheta_abs ? ", haskey(md, :devTheta_abs))
    println("Has :devZ_abs ?     ", haskey(md, :devZ_abs))
    println()

    println(":devTheta_abs = ", get(md, :devTheta_abs, "missing"))
    println(":devZ_abs     = ", get(md, :devZ_abs, "missing"))
    println(":gn_init      = ", get(md, :gn_init, "missing"))
    println(":gn_final     = ", get(md, :gn_final, "missing"))
    println(":hess_is_minimum = ", get(md, :hess_is_minimum, "missing"))
    println()

    θ = get(md, :theta_converged, Float64[])
    println("Length(theta_converged) = ", length(θ))
    if !isempty(θ)
        println("First 10 theta_converged entries:")
        println(θ[1:min(10, end)])
    end
    println()
end

function main()
    path = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N50_k3_graphs1_unweighted",
        "qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d0p025_conv1e-08_abin0p02_hess_on_htol1e-08_ngraphs1_ninit10000_outer1_inner5000_thr0p999.json"
    )

    test_json_file(path)
    test_solver_metadata()
end

main()