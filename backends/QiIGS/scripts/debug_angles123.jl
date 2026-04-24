using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf

function main()

N = 50
k = 3
weighted = false
graph_seed = 1

λ = 1.0
n_inits = 10000

iterations = 1
inner_iterations = 5000
tao = 0.1
angle_conv = 0.01
init_mode = :uniform
save_params = true

angle_bin = 1e-2

ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

@inline canonical_angle(x) = mod(x, pi)
@inline quantize_angle(x, δ) = Int32(round(canonical_angle(x)/δ))
angle_key(θ; δ=1e-2) = Tuple(quantize_angle(x,δ) for x in θ)

gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
W = QiIGS.load_weight_matrix(gpath)

seen_angle = Dict()
seen_round = Dict()

na = 0
nr = 0

for r in 1:n_inits

    run_seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ*1000))

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
    )

    θc = sol.metadata[:theta_converged]

    ak = angle_key(θc; δ=angle_bin)
    rk = QiIGS.spin_config_key(sol.configuration)

    new_angle = !haskey(seen_angle, ak)
    new_round = !haskey(seen_round, rk)

    if new_angle
        seen_angle[ak] = (r, run_seed, copy(θc), copy(sol.configuration))
        na += 1
    end

    if new_round
        seen_round[rk] = (r, run_seed, copy(θc), copy(sol.configuration))
        nr += 1
    end

    if new_angle != new_round
        println("\n============================")
        println("Mismatch at run = $r")
        println("seed = $run_seed")
        println("unique_angle = $na")
        println("unique_round = $nr")
        println("============================")

        if new_round && !new_angle
            println("Type: NEW rounded, OLD angle")
        else
            println("Type: NEW angle, OLD rounded")
        end
    end

end

println("\nFinal counts")
println("unique_angle = $na")
println("unique_round = $nr")

end

main()