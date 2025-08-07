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
	PlutoUI, Random
	using Optim, BlackBoxOptim
	# Random.seed!(1234)
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


# ╔═╡ b9d5dccc-16ed-42f5-a1b4-d9600670219f
begin 
    x_pdf = collect(0.1:0.1:100) 

	#Setup Inverse Gaussian
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function
	Gp0 = x -> Gp(x, Γ_0, Δ_0)

    p = plot(layout=@layout [a b{0.5w}])
	plot!(p[1], x_pdf, Gp0.(x_pdf), lw=4, 
	xlabel = "τ [yrs]", 
    ylabel = "Density [1/yrs]", xlims = (0, 100), 
	title = "Green's Function", label = nothing, color = "black")
	
	#now create a timeseries based on green's function & forcing functipon
	t = collect(0:5:500) #define time of integration
	# @. f(x) = sqrt(x)  # forcing function is linear
	@. f(t::Real) = (t < 0) ? 0.0 : sqrt(t)

	BP = BoundaryPropagator(Gp0, f, t) #setup Boundary Propagator problem
	pred_values = boundary_propagator_timeseries(BP)[1, 1, :]; 
	
	plot!(p[2], t, pred_values, lw=4, 
	xlabel = "time [yrs]", 
    ylabel = "Tracer Value [unitless]", xlims = (0, 50), 
	label="Interior Value Source", color = "black")
	plot!(p[2], t,  f.(t), lw=4, label="Atmospheric Source", 
	color = "green", linestyle=:dash)
    p
end

# ╔═╡ d93d36a6-a0ab-424a-9100-88cb4c5990af
begin
	nobs = 1000;
	obs_idx = sort(rand(collect(1:length(t)), nobs)); #get random index samples 
	
	t_obs = t[obs_idx]; 
	σ = 0.5; obs_error = rand(Normal(0, σ), nobs)
	y_obs = pred_values[obs_idx] .+ obs_error; 
	
	t_obs = t_obs[y_obs .> 0] #remove erroneous values that cannot be fit
	y_obs = y_obs[y_obs .> 0] 
end

# ╔═╡ 433b827e-e2c4-4e26-91a3-9292936678dc
begin 
	GPDist(t, p) = Gp(t, p[1], p[2])
	function BP_Estimate(p::AbstractVector{T}, t) where T
	  # build your propagator and surface time series
	  propagator = BoundaryPropagator(x -> GPDist(x, p), f, t)
	
	  # extract the time series
	  ts_iter = boundary_propagator_timeseries(propagator; 
											   integration_method = :simpsons)
	
	  # collect it into a Vector{T}
	  return collect(ts_iter[:])
	end
	
	#objective
	J(u) = sum(((y_obs .- BP_Estimate(u, t_obs)) ./ σ).^2)
	
	# initial guess and simple lower/upper bounds
	u0 = [10.0, 10.0]
	lower = [0.1, 0.1]
	upper = [Inf, Inf]

	results = optimize(J, lower, upper, u0,
					   Fminbox(LBFGS()))

end

# ╔═╡ f29eb59f-3ff3-4bb7-80b3-ccd4aaadeecd
begin
	# p[1] = α (shape), p[2] = β (rate)
    # Distributions.Gamma(k, θ) uses θ=scale=1/β
	function gamma_from_IG(μ, λ)
	  α = λ/μ             # shape
	  β = λ/(μ^2)           # rate
	  return Gamma(α, 1/β)  # Distributions.Gamma(shape, scale)
	end
		
	InvGammaDist(t, p) = pdf(gamma_from_IG(p[1], p[2]), t) 
  # === Tracer prediction using a Gamma TTD ===
  function BP_Estimate3(p::AbstractVector{T}, t) where T

    propagator = BoundaryPropagator(x -> InvGammaDist(x, p), f, t)
    ts_iter    = boundary_propagator_timeseries(propagator; 
												integration_method = :simpsons)
    return collect(ts_iter[:])
  end

  # === Objective: just the squared‐misfit ===
  function J3(p)
    y_model = BP_Estimate3(p, t_obs)
    return sum(((y_obs .- y_model) ./ σ).^2)
  end

  # === Initial guess and bounds ===
  u03    = [5., 100.]           # α≈2, β≈0.1
  lower3 = [0.2, 0.2]         # enforce α>0, β>0
  upper3 = [ Inf,  Inf ]

  # === Run the fit ===
  results3 = optimize(J3, lower3, upper3, u03,
					  Fminbox(LBFGS()))
	# results3 = bboptimize(
	#   J3, 
	#   # u03;
	#   SearchRange  = [(0.1, 1e7),  # α ∈ [0.1, ∞)
	# 				  (0.1, 1e7)], # β ∈ [0.1, ∞)
	#   NumDimensions = 2,
	#   MaxSteps      = 100_000,        # or whatever budget you like)
	# )
	
end

# ╔═╡ 2133071c-60cb-45be-9500-da2ce603b2b4
begin
	import Optim: minimizer
	minimizer(x::BlackBoxOptim.OptimizationResults) = best_candidate(x)
end

# ╔═╡ 4fdec2ac-af36-4248-bc57-b98ce6842026
function plot_minimize_results(p1, results, GFunc::Function, EstimateFunc::Function)
		Γ_opt, Δ_opt = round.(minimizer(results))
		params = round.(minimizer(results))
		# p1 = plot(layout=@layout [a b{0.5w}])
	
		plot!(p1[1], x_pdf, Gp0.(x_pdf), lw=4, 
		xlabel = "τ [yrs]", 
	    ylabel = "Density [1/yrs]", xlims = (0, 50), 
		title = "Green's Function", 
		label = "True Distriubtion\n(Γ=$Γ_0 yrs, Δ=$Δ_0 yrs)", 
		color = "red")
		plot!(p1[1], x_pdf, GFunc(x_pdf, params), lw=4, 
			  # label = "Estimated Distriubtion\n(Γ=$Γ_opt yrs, Δ=$Δ_opt yrs)", 
		color = "blue")
	
		scatter!(p1[2], t_obs, y_obs, color = "red")
		plot!(p1[2], t, BP_Estimate([Γ_0, Δ_0], t), lw=4, color = "red")
		plot!(p1[2], t, EstimateFunc(params, t), lw=4, color = "blue")
end

# ╔═╡ 88b18409-ef68-4196-8ba4-56cda076d0bc
begin
		# Γ_opt, Δ_opt = round.(Optim.minimizer(results))
		p1 = plot(layout=@layout [a b{0.5w}])
		plot_minimize_results(p1, results, GPDist, BP_Estimate)	
	
end

# ╔═╡ 015f9e47-3000-4203-aad4-c0e1c8f8f7fa
begin 
	p3 = plot(layout=@layout [a b{0.5w}])
	plot_minimize_results(p3, results3, InvGammaDist, BP_Estimate3)	
end

# ╔═╡ 421759e3-1e0c-4f67-91dd-20acfb955c8b
plot( 0.1:50, pdf(gamma_from_IG(20., 46.), 0.1:50))

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═433b827e-e2c4-4e26-91a3-9292936678dc
# ╠═88b18409-ef68-4196-8ba4-56cda076d0bc
# ╠═4fdec2ac-af36-4248-bc57-b98ce6842026
# ╠═f29eb59f-3ff3-4bb7-80b3-ccd4aaadeecd
# ╠═2133071c-60cb-45be-9500-da2ce603b2b4
# ╠═015f9e47-3000-4203-aad4-c0e1c8f8f7fa
# ╠═421759e3-1e0c-4f67-91dd-20acfb955c8b
