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
	using NLsolve

end

# ╔═╡ 5f5eb464-b9ab-4f50-89e0-332a32cab719
minimizer(x::AbstractVector) = x

# ╔═╡ f8b14034-845d-46e2-8420-f6cbd20857bb
begin

	function f!(F, x)
	    F[1] = (x[1]+3)*(x[2]^3-7)+18
	    F[2] = sin(x[2]*exp(x[1])-1)
	end
	
	function j!(J, x)
	    J[1, 1] = x[2]^3-7
	    J[1, 2] = 3*x[2]^2*(x[1]+3)
	    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
	    J[2, 1] = x[2]*u
	    J[2, 2] = u
	end
	
	nlsolve(f!, j!, [ 0.1; 1.2]).zero
end

# ╔═╡ 76b7f535-e142-417e-8a2c-6800539e0faa
# begin 
	
# 	# Define the nonlinear function using QuadGK for integration
# 	function F(x, p)
# 	    # Integrate sin(x[1] * t) from t=0 to t=1
# 	    integral, _ = quadgk(t -> sin(x[1] * t), 0, 1)
# 	    return [integral - 0.5]
# 	end
	
# 	# Initial guess
# 	x0 = [1.0]
	
# 	# Solve the nonlinear equation
# 	probs = NonlinearProblem(F, x0, [0.0])
# 	solss = solve(probs)[:]
# 	println("Root found: x = ", solss)
# end

# ╔═╡ 56754bf2-79a2-474e-8cf9-0d48b90d3477
# begin
# 	import Optim: minimizer
# 	minimizer(x::BlackBoxOptim.OptimizationResults) = best_candidate(x)
# end

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
	Δt = 5
	t = Float64.(collect(1:Δt:500)) #define time of integration
	@. f(t::Real) = (t < 0) ? 0.0 : sqrt(t)
	f_atm = f
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
	nobs = 10;
	obs_idx = sort(rand(collect(1:length(t)), nobs)); #get random index samples 
	
	t_obs = t[obs_idx]; 
	σ = 0.5; obs_error = rand(Normal(0, σ), nobs)
	y_obs = pred_values[obs_idx] .+ obs_error; 
	
	t_obs = t_obs[y_obs .> 0] #remove erroneous values that cannot be fit
	y_obs = y_obs[y_obs .> 0] 
end

# ╔═╡ 8c58158d-1a7f-4165-b72f-2d50b17a5bdb
t_obs

# ╔═╡ c16bd826-6abd-4a75-acd3-d92177f7eddb
function plot_minimize_results(p1, results, GFunc::Function, EstimateFunc::Function)
		# Γ_opt, Δ_opt = round.(minimizer(results))
		params = round.(minimizer(results))
		# p1 = plot(layout=@layout [a b{0.5w}])
	
		plot!(p1[1], x_pdf, Gp0.(x_pdf), lw=4, 
		xlabel = "τ [yrs]", 
	    ylabel = "Density [1/yrs]", xlims = (0, 50), 
		title = "Green's Function", 
		label = "True Distriubtion\n(Γ=$Γ_0 yrs, Δ=$Δ_0 yrs)", 
		color = "red")
		GFuncParamed(x) =  GFunc(x, params)
		plot!(p1[1], x_pdf, GFuncParamed.(x_pdf), lw=4, 
			  # label = "Estimated Distriubtion\n(Γ=$Γ_opt yrs, Δ=$Δ_opt yrs)", 
		color = "blue")
	
		plot!(p1[2], t, pred_values, lw=4, color = "red")
		scatter!(p1[2], t_obs, y_obs, color = "purple")
		plot!(p1[2], t, EstimateFunc(params, t), lw=4, color = "blue")
end

# ╔═╡ 1fe0423b-7a56-4b0e-bc16-a068a1bf3a5e
begin
	τ_max = t[end] - t[1]
	μ = Δt / τ_max
	# === MaxEnt function ===
	function MaxEntFuncNumerator(τ::Real, λ::AbstractVector{T}, t_obs; a=0, b=10000, N=200) where T
		C_s_vec = f_atm.(t_obs .- τ)
		return μ * exp(-sum(λ .* C_s_vec))
	end
	# === Normalization constant ===
	function MaxEntFuncDenominator(λ::AbstractVector{T}; a=0, b=10000, N=200)  where T
		
	    return quadgk(τ -> MaxEntFuncNumerator(τ, λ, t_obs), 0, Inf)[1]
	end

	# === Distribution (cached Z per call) ===
	function MaxEntDist(τ::Real, λ::AbstractVector{T}; a=0, b=10000, N=200) where T
		
		Z = MaxEntFuncDenominator(λ; a=a, b=b, N=N) 

	    return MaxEntFuncNumerator(τ, λ, t_obs; a=a, b=b, N=N) / Z
	end
	
	# === Entropy computation ===
	function EntMeasure(λ::Real; a=0, b=10000, N=200)
	    h = (b - a) / N
	    τs = range(a, stop=b, length=N+1)
		entropy(τ) =  MaxEntDist(τ, λ) * log(MaxEntDist(τ, λ) / μ)
	    return -quadgk(entropy, 0, Inf)[1]
	end

	# === Tracer prediction ===
	function BP_Estimate2(λ::AbstractVector{G}, t::AbstractVector{T}; a=0, b=10000, N=200) where {G, T}
	    dist_fn(τ) = MaxEntDist(τ, λ; a=a, b=b, N=N)
	    propagator = BoundaryPropagator(dist_fn, f, t)
	    ts_iter = boundary_propagator_timeseries(propagator)
	    return collect(ts_iter[:])
	end

	
	# # === Objective function ===
	# function J2(λ; a=0, b=10000, N=100)
	# 	Ĉi = BP_Estimate2(λ, t_obs; a=a, b=b, N=N)
	# 	C_obs_i = y_obs

		
	#     misfit = sum(λ .* (C_obs_i .- Ĉi))
	#     entropy = EntMeasure(λ; a=a, b=b, N=N)
	#     return misfit 
	# end

	# function J2(λ; a=0, b=10000, N=100)
	# 	Ĉi = BP_Estimate2(λ, t_obs; a=a, b=b, N=N)
	# 	C_obs_i = y_obs
		
		
	# 	MaxEntDist(τ::Real, λ::AbstractVector{T}
		
	#     return λ .* (Ĉi .-  C_obs_i)
	# end

	# function J2_jac(λ; a=0, b=10000, N=100)
	# 	Ĉi = BP_Estimate2(λ, t_obs; a=a, b=b, N=N)
	# 	nC = length(Ĉi)
	# 	C_s_vec = [τ ->  f_atm.(t_o .- τ) for t_o in t_obs]

	# 	MED(τ) = MaxEntDist(τ, λ; a=a, b=b, N=N)
		
	# 	jac_vec = [quadgk(τ -> C_s_vec(τ) * MED(τ) * (Ĉi - C_s_vec(τ)), 0, Inf)[1] for i in 1:nC]
	#     return jac_vec
	# end

	function J2!(J2, λ; a=0, b=10000, N=100)
		Ĉi = BP_Estimate2(λ, t_obs; a=a, b=b, N=N)
		C_obs_i = y_obs
			
	    J2 .= (Ĉi .-  C_obs_i)
	end

	function J2_jac!(J2_jac, λ; a=0, b=10000, N=100)
		Ĉi = BP_Estimate2(λ, t_obs; a=a, b=b, N=N)
		nC = length(Ĉi)
		C_s_vec = [τ ->  f_atm.(t_o .- τ) for t_o in t_obs]

		MED(τ) = MaxEntDist(τ, λ; a=a, b=b, N=N)
		
		jac_vec = [quadgk(τ -> C_s_vec[i](τ) * MED(τ) * (Ĉi[i] - C_s_vec[i](τ)), 0, Inf)[1] for i in 1:nC]
	    J2_jac .= jac_vec
	end


	
	# # === Optimization ===
	# u02 = [1.0]

	# # results2 = optimize(J2, lower2, upper2, u02,					  Fminbox(LBFGS()),  # box‐constrained L-BFGS
	# # 				  Optim.Options(g_tol = 1e-16, f_tol = 1e-16))
	# results2 = bboptimize(
	#   J2, 
	#   u02;
	#   SearchRange  = (1e-16, 10),
	#   NumDimensions = length(u02),
	#   MaxSteps      = 10,        # or whatever budget you like)
	# )
		# fs(u, p) = u .* u .- p
	# u0 = ones(length(t_obs))
	# # ps = 2.0
	# prob = NonlinearProblem((λ, p) -> J2(λ), u0, NaN)
	# nlsolve(f!, j!, [ 0.1; 1.2])
	
	results2 = 	nlsolve(J2!, J2_jac!, 1 .+ zeros(nobs)).zero

end

# ╔═╡ 0ff62d38-1e72-4062-9722-bded50bb4453
begin 
	p2 = plot(layout=@layout [a b{0.5w}])
	plot_minimize_results(p2, results2, MaxEntDist, BP_Estimate2)	
end

# ╔═╡ dcb63d4a-5081-4e8f-a59c-51164d460326
MaxEntDist

# ╔═╡ 49308a16-d257-4213-a0a8-140b0d0bc82e
boundary_propagator_timeseries(propagator)

# ╔═╡ 6dbe6bad-75be-4cd9-b932-dd9ca645a803
MaxEntFuncNumerator(x_pdf, [2], t_obs)

# ╔═╡ 543236f2-2252-477b-a953-b5f1fd7421a5
MaxEntFuncDenominator(1e-2)

# ╔═╡ 5330ccef-8043-4530-8dec-87cfc206fdba
MaxEntFuncDenominator

# ╔═╡ 7e7648cb-33e9-465e-b7a1-f49b8b18d357
begin
	MaxEntDistVec(x) = MaxEntDist(x, [-10, 10, -10, -10])
	plot(MaxEntDistVec.(x_pdf))
end


# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╠═5f5eb464-b9ab-4f50-89e0-332a32cab719
# ╠═f8b14034-845d-46e2-8420-f6cbd20857bb
# ╠═76b7f535-e142-417e-8a2c-6800539e0faa
# ╠═56754bf2-79a2-474e-8cf9-0d48b90d3477
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═8c58158d-1a7f-4165-b72f-2d50b17a5bdb
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═c16bd826-6abd-4a75-acd3-d92177f7eddb
# ╠═1fe0423b-7a56-4b0e-bc16-a068a1bf3a5e
# ╠═0ff62d38-1e72-4062-9722-bded50bb4453
# ╠═dcb63d4a-5081-4e8f-a59c-51164d460326
# ╠═49308a16-d257-4213-a0a8-140b0d0bc82e
# ╠═6dbe6bad-75be-4cd9-b932-dd9ca645a803
# ╠═543236f2-2252-477b-a953-b5f1fd7421a5
# ╠═5330ccef-8043-4530-8dec-87cfc206fdba
# ╠═7e7648cb-33e9-465e-b7a1-f49b8b18d357
