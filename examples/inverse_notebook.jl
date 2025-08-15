### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6c4083e2-1f27-11f0-1077-e9a0ae678f2d
begin 
	import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	Pkg.resolve()
	
	using TCMGreensFunctions, Plots, 
	Distributions, Interpolations, 
	PlutoUI, Random
	using Optim, BlackBoxOptim
	using JuMP
	using Ipopt
	# Random.seed!(1234)
end

# ╔═╡ a24a0d7b-7bfa-41e0-8e6f-c98a67125e8b
using LinearAlgebra

# ╔═╡ e947418c-fe9c-4a2d-94a5-6f6e2e72b8a3
begin
	import Optim: minimizer
	minimizer(x::BlackBoxOptim.OptimizationResults) = best_candidate(x)
end

# ╔═╡ a9445454-daf5-4183-8aba-cb61e8595d06
begin
	# This will let us adjust the base parameters
	Γ_0_slider = @bind Γ_0 Slider(20:1:50, default=25, show_value=true)
	Δ_0_slider = @bind Δ_0 Slider(10:1:20, default=15.0, show_value=true)

	md"""
	## Inverse Gaussian Parameters 

	Given the mean age ($\Gamma$) and width ($\Delta$), the inverse Guassian TTD is 
	
	$$\mathcal{G}(τ) = \sqrt{\frac{\Gamma^3}{4 \pi \Delta^2 \tau^3}} \exp\left(-\frac{\Gamma(\tau - \Gamma)^2}{4 \Delta^2 \tau}\right)$$

	Where, 

	$$\Gamma = \int_0^\infty \mathcal{G}(\tau)\tau d\tau$$

	$$2 \Delta^2 = \int_0^\infty \mathcal{G}(\tau)\tau^2 d\tau - \Gamma^2.$$

	Mean Age ($\Gamma_0$): $Γ_0_slider [years]
	
	Width ($\Delta_0$): $Δ_0_slider

	"""
end


# ╔═╡ d899eb4e-bc38-4107-a59b-a2bd4e9e47e2
begin 
	τs = collect(0.1:0.1:100) 
	#now create a timeseries based on green's function & forcing functipon
	t = collect(1.:5:250.) #define time of integration
	@. f_src(x) = (x >= 0.) ? sqrt(x) : 0.0 # forcing function is linear
	τm = 250_000.
	# nτ = Int(τ_max * 5)
end

# ╔═╡ ac9927b8-4f97-45bf-a0c4-b19aecb339d5
begin
	break_points = [
	    1.2 * (t[end] - t[1]),   # first break after twice the data span
	    25_000.0               # second break at fixed value, ~ 1000 years
	]
	
	nodes_points = [
	    10.0  * break_points[1],                # dense in first panel
	    0.03 * (break_points[2] - break_points[1]), # moderate in middle
	    0.01 * (τm - break_points[2])               # lighter in last
	]
	
	nodes_points = Int.(round.(nodes_points))
	panels = make_integration_panels(0., τm, break_points, nodes_points)
	gauss_integrator = make_integrator(:gausslegendre, panels)
	uniform_integrator = make_integrator(:uniform, panels)
end

# ╔═╡ 8aa20b7b-768e-4d6b-aa47-4eb5bd9bc9a5
uniform_integrator.weights

# ╔═╡ b9d5dccc-16ed-42f5-a1b4-d9600670219f
begin 

	# f(x) = sqrt(x)
	#Setup Inverse Gaussian
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function
	Gp0 = x -> Gp(x, Γ_0, Δ_0)

    p = plot(layout=@layout [a b{0.5w}])
	plot!(p[1], τs, Gp0.(τs), lw=4, 
	xlabel = "τ [yrs]", 
    ylabel = "Density [1/yrs]", xlims = (0, 100), 
	title = "Green's Function", label = nothing, color = "black")
	
	BP = BoundaryPropagator(Function[Gp0], Function[f_src], t; 
								τ_max = τm, integr = gauss_integrator) 

	pred_values = zeros(length(Function[Gp0]), length(t))
	pred_values[1, :] .= convolve_at(BP, 1, t)
	
	plot!(p[2], t, pred_values[:], lw=4, 
	xlabel = "time [yrs]", 
    ylabel = "Tracer Value [unitless]", xlims = (0, 50), 
	label="Interior Value Source", color = "black")
	plot!(p[2], t,  f_src.(t), lw=4, label="Atmospheric Source", 
	color = "green", linestyle=:dash)
    p
end

# ╔═╡ d93d36a6-a0ab-424a-9100-88cb4c5990af
begin
	nobs = 15;
	σ =  0.5; 
	# obs_error = rand(Normal(0, σ), nobs)
	y_obs = zeros(nobs) .- 1
	t_obs = zeros(nobs)

	for i_n in 1:nobs
		while y_obs[i_n] < 0.0
			obs_idx = rand(collect(1:length(t))); #get random index samples 
			obs_error = rand(Normal(0, σ))
			if t[obs_idx] ∉ t_obs
				t_obs[i_n] = t[obs_idx]; 
				y_obs[i_n] = pred_values[obs_idx] .+ obs_error; 
			else
				continue
			end
		end
	end
	# v = 3
	# t_obs = t_obs[y_obs .> 0];  #remove erroneous values that cannot be fit
	# y_obs = y_obs[y_obs .> 0]; 
end

# ╔═╡ 6cc5ed3f-ed91-469e-bd24-794b67bc2d25
1 ∉ t_obs

# ╔═╡ 0d6a522a-1f12-41f0-8c3e-e3e68ac7d329
begin
	
	results, Γ_opt, Δ_opt, opt_IG_Dist, IG_BP_Estimate =IG_BP_Inversion(Function[f_src], 
															t_obs, y_obs, σ, 
													τm; integr = gauss_integrator)
	opt_params = [Γ_opt, Δ_opt]
end

# ╔═╡ ba34c832-0bac-4c48-9dde-5dbaed65da20
results

# ╔═╡ 8e1f8c5a-8e2f-495d-b66f-79b73d7579b8
begin
	
	results2, Γ_opt2, Δ_opt2, opt_IG_Dist2, IG_BP_Estimate2 =IG_BP_Inversion_EqualVars(Function[f_src], 
															t_obs, y_obs, σ, 
													τm; integr = gauss_integrator)
	opt_params2 = [1 * Γ_opt2[1], 1 * Δ_opt2[1]]
end

# ╔═╡ 565a07f3-8f9d-4b60-9ca7-a062ae99a0a8
begin 
	# using JuMP
	TruncatedNormal(μ, σ) = truncated(Normal(μ, σ); lower = 1e-12)
	
	nensemble = 1000; 
	param_mean = Γ_opt2[1]
	param_uncertainty =  Γ_opt2[1]* 0.303
	TCM_Γ_dist = TruncatedNormal(param_mean, param_uncertainty)

	prior_ensemble = zeros(nensemble, length(gauss_integrator.nodes))
	for i in 1:nensemble
		prior_ensemble[i, :] .= opt_IG_Dist2(gauss_integrator.nodes, [rand(TCM_Γ_dist)])
	end
end

# = opt_IG_Dist2(gauss_integrator.nodes, opt_params2)

# ╔═╡ e0033af0-b74d-486c-ac9c-4d1e1d670633
Γ_opt2

# ╔═╡ 78bd6b12-3042-46ca-86b3-0a523ccc8389
begin
	matquantile(A, p; dims, kwargs...) = mapslices(x->quantile(x, p; kwargs...), A; dims)

	pp = plot()
	ql = matquantile(prior_ensemble, 0.025, dims = 1)[:]
	qu = matquantile(prior_ensemble, 1. - 0.025, dims = 1)[:]
	qmean = mean(prior_ensemble, dims = 1)[:]
	qm = matquantile(prior_ensemble, 0.5, dims = 1)[:]
	# for i in 1:nensemble
	# 	plot!(prior_ensemble[i, :])
	# end

	plot!(pp, gauss_integrator.nodes, ql; fillrange = qu, fillalpha = 0.35, label = "Confidence band", alpha = 0.0)
	plot!(pp, gauss_integrator.nodes, qm; label = "Median", xlims = (0, 100), alpha = 1.0, c=:black)
	pp
end

# ╔═╡ 8167794d-7735-4114-aa52-8bdb61e94274
plot(mean(prior_ensemble .- qmean', dims = 1)[:])

# ╔═╡ 5e7c8b87-0ab6-41ad-af59-d6f67882cb04
# cov(big.(prior_ensemble))[1, 1]

# ╔═╡ 2e8aad58-31d2-4c8c-8049-e87b4e7dc822
LinearAlgebra.dot(prior_ensemble[:, 1] .- qmean[1], prior_ensemble[:, 1] .- qmean[1]) / nensemble

# ╔═╡ f418858d-b806-40cf-acda-6b427700a049
begin
	normalized_prior_ensemble = prior_ensemble ./ sum(prior_ensemble, dims = 2)
	norm_ensemble_cov = cov(normalized_prior_ensemble; dims=1) + ( I*1e-6) #large number allows for large variations
	norm_ensemble_mean = mean(normalized_prior_ensemble, dims = 1)[:]
	nts = length(norm_ensemble_mean)
	fac_cov = cholesky(Symmetric(norm_ensemble_cov))
	inv_ensemble_cov = Symmetric(fac_cov \ I(nts))
end

# ╔═╡ fa8939bb-c9dd-417d-b666-c8dedd9ae559
# x = Nonconvex.@load Ipopt

# ╔═╡ 501b2bd8-9037-4add-9288-d090f6dd9f39
	# result = optimize(model, alg, x0, options = options)


# ╔═╡ ed60cb7a-f7ea-4e56-a8ae-1b85af73ded6
begin 
	#jump is super fast! 8x faster than NonConvex.jl
	jmodel = JuMP.Model(Ipopt.Optimizer)
	set_time_limit_sec(jmodel, 60.0)
	@variable(jmodel, 0 <= x[i = 1:N], start = 1/N)
	@constraint(jmodel, sum(x) == 1)
	
	@objective(jmodel, Min, cost_function(x))
end

# ╔═╡ e288c46b-585f-4d19-9cf7-109adf41be28
@time JuMP.optimize!(jmodel)

# ╔═╡ 5f4bda6a-88d7-4822-8a70-1a4de24fd73c
begin
	 normgp0 = Gp0.(gauss_integrator.nodes)
	 normgp0 = normgp0 / sum(normgp0)
	 plot(gauss_integrator.nodes, normgp0, label = "True dist", lw = 2)
	 plot!(gauss_integrator.nodes, value(x), label = "TCM solution", lw = 2)
	 plot!(gauss_integrator.nodes, norm_ensemble_mean, xlims = (0, 150), label = "IG Prior", lw = 2)
end

# ╔═╡ eb0d99a8-6933-4273-b356-3e56405123bd
begin
	scatter(t_obs, convolveG(value(x)))
	scatter!(t_obs, y_obs)
end

# ╔═╡ 27d59a43-493c-4245-96c1-c9aa5f748d05
sum( .<= 0)

# ╔═╡ 1aea28bd-a365-4fb0-bb91-9c88ce96abd1
convolve_discrete(Gnew0, 10)

# ╔═╡ 36b321fe-e524-446c-8ac9-2f880eefe125
Gnew0

# ╔═╡ fbd4fb1d-109b-45fc-b727-0e5875bcbb52
norm_ensemble_mean

# ╔═╡ c0e047f0-d442-48c5-8efc-2bc04dee27f4
#  * pinv(cov(prior_ensemble; dims=1))

# ╔═╡ e2da6ee7-131d-4546-b102-e5b0608acc00
# (prior_ensemble .- qmean')' * (prior_ensemble .- qmean')

# ╔═╡ 07717215-c39d-4a28-8d0b-80b2806c8763
begin
	plot(q1[:])
	plot!(q2[:])
end

# ╔═╡ 3761e8d4-2ccb-4a2a-8108-185b062f36a1
quantile(rand(TCM_Γ_dist, 100_000), 0.975)

# ╔═╡ a41059e1-96fd-4bc7-8abf-368f15c83540
TruncatedNormal(μ, σ)

# ╔═╡ 7ae547b5-92fa-43b7-98bc-82f09253b251
BoundaryPropagator(Function[Gp0], Function[f_src], t; 
								τ_max = τm, integr = gauss_integrator) 

# ╔═╡ 4fdec2ac-af36-4248-bc57-b98ce6842026
function plot_minimize_results(p1::Plots.Plot, 
							   opt_params::AbstractArray, 
							   GFunc::Function,
							   EstimateFunc::Function)
		
		Γ_opt, Δ_opt = round.(opt_params)
	
		plot!(p1[1], τs, Gp0.(τs), lw=4, 
		xlabel = "τ [yrs]", 
	    ylabel = "Density [1/yrs]", xlims = (0, 100), 
		title = "Green's Function", 
		label = "True Distriubtion\n(Γ=$Γ_0 yrs, Δ=$Δ_0 yrs)", 
		color = "red")
		plot!(p1[1], τs, GFunc(τs, opt_params), lw=4, 
			  label = "Estimated Distriubtion\n(Γ=$Γ_opt yrs, Δ=$Δ_opt yrs)", 
		color = "blue")
	
		scatter!(p1[2], t_obs, y_obs, color = "red")
		plot!(p1[2], t, EstimateFunc([Γ_0, Δ_0], t), lw=4, color = "red")
		plot!(p1[2], t, EstimateFunc(opt_params, t), lw=4, color = "blue")
end

# ╔═╡ 88b18409-ef68-4196-8ba4-56cda076d0bc
begin
		p1 = plot(layout=@layout [a b{0.5w}])
		plot_minimize_results(p1, opt_params, opt_IG_Dist, IG_BP_Estimate)	
	
end

# ╔═╡ e3cb541b-2aef-4a04-bc92-20262ed3cc84
begin
		p2 = plot(layout=@layout [a b{0.5w}])
		plot_minimize_results(p2, opt_params2, opt_IG_Dist2, IG_BP_Estimate2)	
	
end

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╠═e947418c-fe9c-4a2d-94a5-6f6e2e72b8a3
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═d899eb4e-bc38-4107-a59b-a2bd4e9e47e2
# ╠═ac9927b8-4f97-45bf-a0c4-b19aecb339d5
# ╠═8aa20b7b-768e-4d6b-aa47-4eb5bd9bc9a5
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═6cc5ed3f-ed91-469e-bd24-794b67bc2d25
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═ba34c832-0bac-4c48-9dde-5dbaed65da20
# ╠═0d6a522a-1f12-41f0-8c3e-e3e68ac7d329
# ╠═88b18409-ef68-4196-8ba4-56cda076d0bc
# ╠═8e1f8c5a-8e2f-495d-b66f-79b73d7579b8
# ╠═e3cb541b-2aef-4a04-bc92-20262ed3cc84
# ╠═565a07f3-8f9d-4b60-9ca7-a062ae99a0a8
# ╠═e0033af0-b74d-486c-ac9c-4d1e1d670633
# ╠═78bd6b12-3042-46ca-86b3-0a523ccc8389
# ╠═8167794d-7735-4114-aa52-8bdb61e94274
# ╠═5e7c8b87-0ab6-41ad-af59-d6f67882cb04
# ╠═2e8aad58-31d2-4c8c-8049-e87b4e7dc822
# ╠═a24a0d7b-7bfa-41e0-8e6f-c98a67125e8b
# ╠═f418858d-b806-40cf-acda-6b427700a049
# ╠═fa8939bb-c9dd-417d-b666-c8dedd9ae559
# ╠═501b2bd8-9037-4add-9288-d090f6dd9f39
# ╠═ed60cb7a-f7ea-4e56-a8ae-1b85af73ded6
# ╠═e288c46b-585f-4d19-9cf7-109adf41be28
# ╠═5f4bda6a-88d7-4822-8a70-1a4de24fd73c
# ╠═eb0d99a8-6933-4273-b356-3e56405123bd
# ╠═27d59a43-493c-4245-96c1-c9aa5f748d05
# ╠═1aea28bd-a365-4fb0-bb91-9c88ce96abd1
# ╠═36b321fe-e524-446c-8ac9-2f880eefe125
# ╠═fbd4fb1d-109b-45fc-b727-0e5875bcbb52
# ╠═c0e047f0-d442-48c5-8efc-2bc04dee27f4
# ╠═e2da6ee7-131d-4546-b102-e5b0608acc00
# ╠═07717215-c39d-4a28-8d0b-80b2806c8763
# ╠═3761e8d4-2ccb-4a2a-8108-185b062f36a1
# ╠═a41059e1-96fd-4bc7-8abf-368f15c83540
# ╠═7ae547b5-92fa-43b7-98bc-82f09253b251
# ╠═4fdec2ac-af36-4248-bc57-b98ce6842026
