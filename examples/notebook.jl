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
	Pkg.activate("../")
	using TCMGreensFunctions, Plots, 
	Distributions, QuadGK, Interpolations, 
	PlutoUI
end

# ╔═╡ a9445454-daf5-4183-8aba-cb61e8595d06
begin
	# This will let us adjust the base parameters
	μ_0_slider = @bind μ_0 Slider(0.5:0.5:30, default=15, show_value=true)
	λ_0_slider = @bind λ_0 Slider(0.5:0.5:20, default=10.0, show_value=true)
	scaling_slider = @bind age_scaling Slider(10:5:40, default=10.0, show_value=true)

	md"""
	## Base Parameters
	
	Mean (μ₀): $μ_0_slider
	
	Width (λ₀): $λ_0_slider

	Depth Scaling: $scaling_slider
	"""
end


# ╔═╡ d93d36a6-a0ab-424a-9100-88cb4c5990af
begin
	Gp(x, μ, λ) = pdf(InverseGaussian(μ, λ), x)  # Inv. Gauss. Boundary propagator function

	depths = [0, 5, 20, 30] #depths of boxes we are modeling 
	depth_colors = cgrad(:greens, length(depths), categorical=true)  # Create a categorical gradient with fixed colors
	
	μs = exp.(depths ./ age_scaling) .* μ_0 #mean age increases with depth
	λs = exp.(depths ./ age_scaling) .* λ_0 #width increases with depth

	Gps = [x -> Gp(x, μ, λ) for (μ, λ) in zip(μs, λs)]

end

# ╔═╡ b9d5dccc-16ed-42f5-a1b4-d9600670219f
begin 
	x_pdf = collect(0:0.05:50) 
	p = plot()
	for i in 1:length(depths)
	    plot!(p, x_pdf, Gps[i].(x_pdf), 
	          label="GP $i", 
	          color = depth_colors[i], 
	          lw = 4)
	end
	p
end

# ╔═╡ 83c0c2e8-8044-4ca7-95b2-a716afa5ea65
begin
	#define time of integration
	t = 1:40
	# Surface flux rate is defined quadratic 
	@. f(x) = x^2 / 2  

	#setup Boundary Propagator problem
	BP = BoundaryPropagator(Gps, f, t) 

	#solve for convolution of surface and boundary props
	pred_values = boundary_propagator_timeseries(BP); nothing
	
	p1 = plot(t, f.(t), 
	         label="Atmospheric Source", 
	         color = :black, 
	         linewidth = 3)
	for i in 1:length(depths)
	    plot!(p1, t, pred_values[i, 1, :], 
	    label="Ocean Values $i", 
	    color = depth_colors[i], 
	    lw = 4)
	end
	p1
end

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╟─b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═83c0c2e8-8044-4ca7-95b2-a716afa5ea65
