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
	using TCMGreensFunctions, Plots, 
	Distributions, Interpolations, 
	PlutoUI, Random, QuadGK
	# using Optim, BlackBoxOptim
	# Random.seed!(1234)
	# using NLsolve
	# using DoubleExponentialFormulas
	# using NonlinearSolve
	using Optim
	using BenchmarkTools
	using ForwardDiff
	using LinearAlgebra
end

# ╔═╡ 5f5eb464-b9ab-4f50-89e0-332a32cab719
minimizer(x::AbstractVector) = x

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


# ╔═╡ ef1d746d-97bc-490f-ade0-5b2971416876


# ╔═╡ ada4f055-8da0-4abf-bae0-e38789276d7a
begin 
	τs = collect(0.1:0.1:100) 
	#now create a timeseries based on green's function & forcing functipon
	t = collect(1.:5:250.) #define time of integration
	@. f_src(x) = (x >= 0.) ? sqrt(x) : 0.0 # forcing function is linear
	τm = 250_000.
	# nτ = Int(τ_max * 5)
end

# ╔═╡ fb9d560d-118d-4d8f-b02f-50f4ffab182d
begin
	break_points = [
	    1.1 * (t[end] - t[1]),   # first break after twice the data span
	    25_000.0               # second break at fixed value, ~ 1000 years
	]
	
	nodes_points = [
	    30  * break_points[1],                # dense in first panel
	    0.4 * (break_points[2] - break_points[1]), # moderate in middle
	    0.03 * (τm - break_points[2])               # lighter in last
	]
	
	nodes_points = Int.(round.(nodes_points))
	panels = make_integration_panels(0., τm, break_points, nodes_points)
	gauss_integrator = make_integrator(:gausslegendre, panels)
	nnodes = length(gauss_integrator.weights)
	println("Using $nnodes Nodes")
end

# ╔═╡ e9dff805-81f6-45eb-98bb-225bc40a1963
break_points

# ╔═╡ b9d5dccc-16ed-42f5-a1b4-d9600670219f
begin 

	# f(x) = sqrt(x)
	#Setup Inverse Gaussian
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function
	Gp1 = x -> Gp(x, Γ_0, Δ_0) #+ (0.001 * rand())
	Gp2 = x -> Gp(x, Γ_0 * 2, Δ_0) #+ (0.001 * rand())
	Gp0 = x -> (Gp1(x) + Gp2(x)) / 2
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
	nobs = 3;
	σ = .2; 
	# obs_error = rand(Normal(0, σ), nobs)
	y_obs = zeros(nobs) .- 1
	t_obs = zeros(nobs)

	for i_n in 1:nobs
		while y_obs[i_n] < 0.0
			obs_idx = rand(collect(1:length(t))); #get random index samples 
			obs_error = rand(Normal(0, σ))
			t_obs[i_n] = t[obs_idx]; 
			y_obs[i_n] = pred_values[obs_idx] .+ obs_error; 
		end
	end
	# t_obs = sort(t_obs)
	# y_obs = sort(y_obs)
	# t_obs = t_obs[y_obs .> 0];  #remove erroneous values that cannot be fit
	# y_obs = y_obs[y_obs .> 0]; 
end

# ╔═╡ 8c58158d-1a7f-4165-b72f-2d50b17a5bdb
t_obs

# ╔═╡ c16bd826-6abd-4a75-acd3-d92177f7eddb
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
		plot!(p1[1], τs, GFunc.(τs), lw=4, 
			  label = "Estimated Distriubtion\n(Γ=$Γ_opt yrs, Δ=$Δ_opt yrs)", 
		color = "blue")
	
		scatter!(p1[2], t_obs, y_obs, color = "red")
		# plot!(p1[2], t, EstimateFunc([Γ_0, Δ_0], t), lw=4, color = "red")
		plot!(p1[2], t, EstimateFunc.(t), lw=4, color = "blue")
end

# ╔═╡ 6b02e3c5-797d-4acc-87c0-306475ff2abc
begin 
	results, Γ_opt, Δ_opt, opt_IG_Dist, IG_BP_Estimate =IG_BP_Inversion(Function[f_src], 
															t_obs, y_obs, σ, 
													τm; integr = gauss_integrator)
	opt_params = [Γ_opt, Δ_opt]	
	println("most optimal params: ", opt_params)
	# τ_max = t[end] - t[1]
	weightsum =  sum(gauss_integrator.weights .* gauss_integrator.nodes)
	# μ(τ) = 1 / weightsum

	# Inverse-Gaussian kernel (expects TracerInverseGaussian(Γ, Δ) to exist)
    IG_Dist(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)
    IG_Dist(τ, λ::Vector{<:Real}) = IG_Dist(τ, λ[1], λ[2])
	
	μ_opt(τ) = IG_Dist(τ, opt_params)
	μ_opt_max = maximum(μ_opt.(gauss_integrator.nodes))
	μ_Z = sum(μ_opt.(gauss_integrator.nodes))
	μ(τ) = μ_opt(τ) / μ_Z
end

# ╔═╡ 9e54d189-d52f-44e6-afa3-d33cf76e6d6c
μ_Z

# ╔═╡ bc87c57b-60a6-4c36-8566-a4e723f76e63
begin 
	C0 = 0.0
	λ = -25 .* (rand(length(y_obs)))
	# ME_numerator_logged(τ) = MaxEntFuncNumeratorPreLogged(τ, λ, t_obs, f_src, μ)
	# logZ = MaxEntFuncDenominatorPreLogged(λ, t_obs, f_src, μ, gauss_integrator)
	ME_dist_prelogged = MaxEntDist(λ, t_obs, f_src, μ, gauss_integrator, implementation=:stable)
	# @time println(integrate(ME_dist_prelogged,0.0, gauss_integrator, 0.0))
end

# ╔═╡ cf24ecc4-b846-4ca1-a995-1267f63c1fa7
@btime MaxEntDist(λ, t_obs, f_src, μ, gauss_integrator, implementation=:stable)

# ╔═╡ e34093c5-b70d-4417-9f28-7ce4c8244fc7
plot(ME_dist_prelogged.(gauss_integrator.nodes))

# ╔═╡ 60f2bad0-b8d1-41cd-9551-b3d252c4c9db


# ╔═╡ a5a6cb36-bef3-4502-9639-77549e5f3c84
maximum(ME_dist_prelogged.(gauss_integrator.nodes))

# ╔═╡ 1fe0423b-7a56-4b0e-bc16-a068a1bf3a5e
begin

    function BP_Estimate(λ, t_obs, t; τm, integr, C0)
		ME_dist = MaxEntDist(λ, t_obs, f_src, μ, integr, implementation=:stable)
		ME_dist_func(τ) = ME_dist(τ)
        return convolve_at(ME_dist_func, Function[f_src], t; τ_max=τm, integr=gauss_integrator, C0=C0)
    end
	
	function J2_NLS(λ, p)
		Ĉ = BP_Estimate(λ, t_obs, t_obs; τm=τm, integr=gauss_integrator, C0=C0)
		return @. ((Ĉ) - (y_obs))
	end
	function J2_NLS!(J, λ, p)
		Ĉ = BP_Estimate(λ, t_obs, t_obs; τm=τm, integr=gauss_integrator, C0=C0)
		@. J = ((Ĉ) - (y_obs))
	end
	
	function J2_jac_NLS(λ, p)
		nodes   = gauss_integrator.nodes
		weights = gauss_integrator.weights
	
		m = length(t_obs)   # number of residuals (observations)
		n = length(λ)       # number of parameters
	
		# Max-Ent distribution evaluated on quadrature nodes
		ME_dist   = MaxEntDist(λ, t_obs, f_src, μ, gauss_integrator; implementation = :stable)
		ME_at_lags = ME_dist.(nodes)
	
		J = zeros(m, n)
	
		# J[i, k] = ∑_j w_j * f(t_i - τ_j) * ME(τ_j) * ( Ĉ_k - f(t_k - τ_j) )
		#verified correct via autodifferentiation 
		@inbounds for i in 1:m               # rows: observations
			ti = t_obs[i]
			for k in 1:n                     # cols: surface sources
				tk = t_obs[k]
	
				lagged_f_ti = f_src.(ti .- nodes)
				lagged_f_tk = f_src.(tk .- nodes)
	
				Ĉ = dot(weights, lagged_f_tk .* ME_at_lags)

				#potential cancellation issues
				# integrand = lagged_f_ti .* ME_at_lags .* (Ĉ .- lagged_f_tk)
				# J[i, k] = dot(weights, integrand)

				A_i = dot(weights, lagged_f_ti .* ME_at_lags)
				B_ik = dot(weights, (lagged_f_ti .* lagged_f_tk) .* ME_at_lags)
				J[i,k] = Ĉ * A_i - B_ik
			end
		end
	
		return J .+ Diagonal(1e-12 .* ones(n))
	end

end

# ╔═╡ a8edba89-5feb-4848-b859-caf28412f3f5
BP_Estimate(λ, t_obs, t; τm=τm, integr=gauss_integrator, C0=C0)

# ╔═╡ b391f1ce-77e2-4825-b573-f8c0e8a6e7a5
begin
	
	function J(λ)
		ME_dist   = MaxEntDist(λ, t_obs, f_src, μ, gauss_integrator; implementation = :stable)
		Zlog = ME_dist.Zlog
		Σ = I(nobs) .* (σ^2)
		
		return dot(λ, y_obs) + 0.5 * (λ' * Σ * λ) + Zlog 
		
	end
	
	initial_x = rand(nobs)
	resultsz1 = optimize(J, initial_x, SimulatedAnnealing(), 
						Optim.Options(iterations=50); 
						autodiff = :forward)
	results21 = Optim.minimizer(resultsz1)

	resultsz = optimize(J, results21, LBFGS(), 
						Optim.Options(iterations=50); 
						autodiff = :forward)
	
	# results2opt = Optim.minimizer(resultsz)

	# fn = NonlinearFunction(r!, jac=jac_r!)
	# prob_0 = NonlinearProblem(fn, initial_x, nothing)
	# # resultsz  = solve(prob, LevenbergMarquardt(); maxiters=200, store_trace = Val(true))

	# resultsz_0  = solve(prob_0, LevenbergMarquardt(); maxiters=250, store_trace = Val(true))

	# prob = NonlinearProblem(fn, resultsz_0[:], nothing)
	# resultsz  = solve(prob, TrustRegion(); maxiters=250, store_trace = Val(true))

	# results2 = resultsz[:]
	results2 = Optim.minimizer(resultsz)
	resultsz
end

# ╔═╡ 0ff62d38-1e72-4062-9722-bded50bb4453
begin 
	λ_opt = results2

	
	ME_dist_obj = MaxEntDist(λ_opt, t_obs, f_src, μ, gauss_integrator, implementation=:stable)
	ME_dist2(τ) = ME_dist_obj(τ)
	BP_Estimate_opt(t::Real) = BP_Estimate(λ_opt, t_obs, t; τm = τm, integr = gauss_integrator, C0 = 0.0)
	
	p2 = plot(layout=@layout [a b{0.5w}])
	plot_minimize_results(p2, λ_opt, ME_dist2, BP_Estimate_opt)
end

# ╔═╡ 95a0ac57-d95b-4535-9d23-3ec3ce705e88
begin 
	τs2 = gauss_integrator.nodes
	plot(τs2,  gauss_integrator.weights .* Gp0.(τs2), label = "Source", lw = 8)
	plot!(τs2, gauss_integrator.weights .* ME_dist2.(τs2), label = "Posterior", lw = 10)
	plot!(τs2, gauss_integrator.weights .* μ.(τs2), label = "Prior", lw = 3, color = :black)
	xlims!((-1,  250))
end

# ╔═╡ 803f5be7-52ff-48e9-aa83-844bbb8cfbd7
maximum(ME_dist2.(τs2) .- μ_opt.(τs2))

# ╔═╡ b1a49a60-8220-4133-a9f7-43396d900884


# ╔═╡ 0ff20c4f-63a2-4227-801f-1b693e0ca3c3
λ_opt

# ╔═╡ 68502a9f-c732-4661-afd1-b310edf7926b
ME_dist2.(τs)

# ╔═╡ 73cef489-db16-4880-90de-74330ab21186
ME_dist2.(τs) .- μ.(τs)

# ╔═╡ 1dafa554-a81a-456b-9efb-63c5ea2067b3
BP_Estimate(λ.* 0 .+  0.003, t_obs, t_obs; τm=τm, integr=gauss_integrator, C0=C0)

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╠═5f5eb464-b9ab-4f50-89e0-332a32cab719
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═ef1d746d-97bc-490f-ade0-5b2971416876
# ╠═ada4f055-8da0-4abf-bae0-e38789276d7a
# ╠═fb9d560d-118d-4d8f-b02f-50f4ffab182d
# ╠═e9dff805-81f6-45eb-98bb-225bc40a1963
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═8c58158d-1a7f-4165-b72f-2d50b17a5bdb
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═c16bd826-6abd-4a75-acd3-d92177f7eddb
# ╠═6b02e3c5-797d-4acc-87c0-306475ff2abc
# ╠═9e54d189-d52f-44e6-afa3-d33cf76e6d6c
# ╠═bc87c57b-60a6-4c36-8566-a4e723f76e63
# ╠═cf24ecc4-b846-4ca1-a995-1267f63c1fa7
# ╠═e34093c5-b70d-4417-9f28-7ce4c8244fc7
# ╠═60f2bad0-b8d1-41cd-9551-b3d252c4c9db
# ╠═a5a6cb36-bef3-4502-9639-77549e5f3c84
# ╠═a8edba89-5feb-4848-b859-caf28412f3f5
# ╠═1fe0423b-7a56-4b0e-bc16-a068a1bf3a5e
# ╠═b391f1ce-77e2-4825-b573-f8c0e8a6e7a5
# ╠═0ff62d38-1e72-4062-9722-bded50bb4453
# ╠═95a0ac57-d95b-4535-9d23-3ec3ce705e88
# ╠═803f5be7-52ff-48e9-aa83-844bbb8cfbd7
# ╠═b1a49a60-8220-4133-a9f7-43396d900884
# ╠═0ff20c4f-63a2-4227-801f-1b693e0ca3c3
# ╠═68502a9f-c732-4661-afd1-b310edf7926b
# ╠═73cef489-db16-4880-90de-74330ab21186
# ╠═1dafa554-a81a-456b-9efb-63c5ea2067b3
