using JSON

path = "results/N30/k3/seed1/weighted/runs/lambda1_sw80_pct0.3_maxd1_modelocal/run_s0201.json"
d = JSON.parsefile(path)

h = Float64.(d["best_history"])
println("T = ", length(h))
println("unique best_history values = ", length(unique(h)))
println("first 10 = ", h[1:min(10,end)])
println("last  10 = ", h[max(end-9,1):end])