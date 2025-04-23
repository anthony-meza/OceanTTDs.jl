### A Pluto.jl notebook ###
# v0.20.6

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
	using Optim
	Random.seed!(1234)
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
    x_pdf = collect(0:0.05:100) 

	#Setup Inverse Gaussian
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function
	Gp0 = x -> Gp(x, Γ_0, Δ_0)

    p = plot(layout=@layout [a b{0.5w}])
	plot!(p[1], x_pdf, Gp0.(x_pdf), lw=4, 
	xlabel = "τ [yrs]", 
    ylabel = "Density [1/yrs]", xlims = (0, 100), 
	title = "Green's Function", label = nothing, color = "black")
	
	#now create a timeseries based on green's function & forcing functipon
	t = collect(1:500) #define time of integration
	@. f(x) = sqrt(x)  # forcing function is linear
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
	nobs = 100;
	obs_idx = sort(rand(collect(1:length(t)), nobs)); #get random index samples 
	
	t_obs = t[obs_idx]; 
	σ = 2; obs_error = rand(Normal(0, σ), nobs)
	y_obs = pred_values[obs_idx] .+ obs_error; 
	
	t_obs = t_obs[y_obs .> 0] #remove erroneous values that cannot be fit
	y_obs = y_obs[y_obs .> 0] 
end

# ╔═╡ 433b827e-e2c4-4e26-91a3-9292936678dc
begin 

	BP_Estimate(p, t) = boundary_propagator_timeseries(BoundaryPropagator(x -> Gp(x, p[1], p[2]), f, t))[:]
	
	J(u) = sum(((y_obs .- BP_Estimate(u, t_obs)) / σ).^2) #define a simple objective
	u0 = [5.0, 5.0]
	results = optimize(J, [1.0, 1.0], [Inf, Inf], u0, IPNewton())

end

# ╔═╡ 88b18409-ef68-4196-8ba4-56cda076d0bc
begin
		Γ_opt, Δ_opt = round.(Optim.minimizer(results))
		p1 = plot(layout=@layout [a b{0.5w}])
	
		plot!(p1[1], x_pdf, Gp0.(x_pdf), lw=4, 
		xlabel = "τ [yrs]", 
	    ylabel = "Density [1/yrs]", xlims = (0, 50), 
		title = "Green's Function", 
		label = "True Distriubtion\n(Γ=$Γ_0 yrs, Δ=$Δ_0 yrs)", 
		color = "red")
		plot!(p1[1], x_pdf, Gp.(x_pdf, Γ_opt, Δ_opt), lw=4, 
			  label = "Estimated Distriubtion\n(Γ=$Γ_opt yrs, Δ=$Δ_opt yrs)", 
		color = "blue")
	
		scatter!(p1[2], t_obs, y_obs, color = "red")
		plot!(p1[2], t, BP_Estimate([Γ_0, Δ_0], t), lw=4, color = "red")
		plot!(p1[2], t, BP_Estimate([Γ_opt, Δ_opt], t), lw=4, color = "blue")
	
end

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═433b827e-e2c4-4e26-91a3-9292936678dc
# ╠═88b18409-ef68-4196-8ba4-56cda076d0bc
