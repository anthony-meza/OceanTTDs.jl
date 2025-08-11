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
	# Random.seed!(1234)
end

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
	    0.05 * (break_points[2] - break_points[1]), # moderate in middle
	    0.01 * (τm - break_points[2])               # lighter in last
	]
	
	nodes_points = Int.(round.(nodes_points))
	panels = make_integration_panels(0., τm, break_points, nodes_points)
	gauss_integrator = make_integrator(:gausslegendre, panels)
end

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
	nobs = 2;
	σ =  0.5; 
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
	# t_obs = t_obs[y_obs .> 0];  #remove erroneous values that cannot be fit
	# y_obs = y_obs[y_obs .> 0]; 
end

# ╔═╡ 0d6a522a-1f12-41f0-8c3e-e3e68ac7d329
begin
	
	results, Γ_opt, Δ_opt, opt_IG_Dist, IG_BP_Estimate =IG_BP_Inversion_NLLS(Function[f_src], 
															t_obs, y_obs, σ, 
													τm; integr = gauss_integrator)
	opt_params = [Γ_opt, Δ_opt]
end

# ╔═╡ ba34c832-0bac-4c48-9dde-5dbaed65da20
results

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

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╠═e947418c-fe9c-4a2d-94a5-6f6e2e72b8a3
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═d899eb4e-bc38-4107-a59b-a2bd4e9e47e2
# ╠═ac9927b8-4f97-45bf-a0c4-b19aecb339d5
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═ba34c832-0bac-4c48-9dde-5dbaed65da20
# ╠═0d6a522a-1f12-41f0-8c3e-e3e68ac7d329
# ╠═88b18409-ef68-4196-8ba4-56cda076d0bc
# ╠═4fdec2ac-af36-4248-bc57-b98ce6842026
