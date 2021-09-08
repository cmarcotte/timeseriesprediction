using DifferentialEquations, Plots, CellMLToolkit, Sundials, TimeseriesPrediction, DynamicalSystems

model_root = joinpath(splitdir(pathof(CellMLToolkit))[1], "..", "models")
ml = CellModel(joinpath(model_root, "ohara_rudy_cipa_v1_2017.cellml.xml"));

tspan = (0, 15000.0)
prob = ODEProblem(ml, tspan);

p = prob.p; p[201] = 175.0; prob = remake(prob, p=p)
tt = collect(tspan[begin]:1.0:tspan[end]) 

sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), dtmax=0.5, saveat=tt)
plot(sol, vars=49, linewidth=3)

data = sol[49,:]

train_inds = 1:10000
testy_inds = 10001:length(tt)

trainData = data[train_inds] .+ 1e-0 .* randn(Float64, length(train_inds))

#theiler = estimate_delay(trainData, "mi_min") # estimate a Theiler window
#Tmax = 200 # maximum possible delay
#Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(trainData; τs = 0:1:Tmax , w = theiler, econ = true)

τ_vals = 0:1:10
d = length(τ_vals)
ntype = FixedMassNeighborhood(3)
pred = localmodel_tsp(trainData, d, τ_vals, length(testy_inds); ntype=ntype)

plot!(tt[train_inds], trainData, linewidth=2, label = "training (trunc.)")
plot!(tt[testy_inds], data[testy_inds], linewidth=2, label = "actual signal")
plot!(tt[testy_inds], pred[2:end], color=:red, linestyle=:dash, label="predicted")
plot!(title = "Training points: $(length(train_inds)), predicted points: $(length(testy_inds))", xlabel="\$ t \$", ylabel = "\$ V(t) \$")
plot!(xlims=[tt[train_inds[end-1000]], tt[end]])
savefig("./testing.pdf")

