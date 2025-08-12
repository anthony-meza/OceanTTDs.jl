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
	using Revise
	Pkg.activate(".")
	Pkg.instantiate()
	using TCMGreensFunctions, Plots, 
	Distributions, Interpolations, 
	PlutoUI
	
end

# ╔═╡ a9445454-daf5-4183-8aba-cb61e8595d06
begin
	# This will let us adjust the base parameters
	Γ_0_slider = @bind Γ_0 Slider(0.5:0.5:30, default=15, show_value=true)
	Δ_0_slider = @bind Δ_0 Slider(0.5:0.5:20, default=10.0, show_value=true)
	scaling_slider = @bind λ Slider(200:5:400, default=10.0, show_value=true)

	md"""
	## Inverse Gaussian Parameters 

	Given the mean age ($\Gamma$) and width ($\Delta$), the inverse Guassian TTD is 
	
	$$\mathcal{G}(τ) = \sqrt{\frac{\Gamma^3}{4 \pi \Delta^2 \tau^3}} \exp\left(-\frac{\Gamma(\tau - \Gamma)^2}{4 \Delta^2 \tau}\right)$$

	Where, 

	$$\Gamma = \int_0^\infty \mathcal{G}(\tau)\tau d\tau$$

	$$2 \Delta^2 = \int_0^\infty \mathcal{G}(\tau)\tau^2 d\tau - \Gamma^2.$$

	Here, we also introduce an additional parameter, $\lambda$, which will 
	dictate how $\Gamma$ and $\Delta$ change as a function of height. I.e., $\mathbf{\Gamma}(z) = \Gamma_0 \exp(z / \lambda)$ and $\mathbf{\Delta}(z) = \Delta_0 \exp(z / \lambda)$. Using these defitions of $\mathbf{\Gamma}$ and \mathbf{\Delta}, we can explore how different distributions can effect how tracers can propagate into the interior. 

	Mean Age ($\Gamma_0$): $Γ_0_slider [years]
	
	Width ($\Delta_0$): $Δ_0_slider

	Depth Scaling ($\lambda$): $scaling_slider
	"""
end


# ╔═╡ d93d36a6-a0ab-424a-9100-88cb4c5990af
begin
	# Inv. Gauss. Boundary propagator function
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  
	depths = [0, 5, 100, 500] #depths of boxes we are modeling 
	depth_colors = cgrad(:greens, length(depths), categorical=true)  # Create a categorical gradient with fixed colors
	
	Γs = exp.(depths ./ λ) .* Γ_0 # mean age increases with depth
	Δs = exp.(depths ./ λ) .*Δ_0 # width increases with depth

	Gps = Function[x -> Gp(x, μ, λ) for (μ, λ) in zip(Γs, Δs)]  # Vector{Function}

end

# ╔═╡ 2971aa56-fb85-40bd-b19c-884d89a57af8
begin
	#define time of integration
	t = collect(0.1:1.:250.)
	#its a function of mean age mostly, if the mean agge is small then the inverse 
	#gaussian is not very smooth
	τm = 250_000.
	nτ = 100_000
	# Surface flux rate is defined quadratic 
	@. f_src(x) = (x >= 0) ? x^2 / 2 : 0.0
	# @. f_src(x) = (x >= 0) ? exp(-x/.1) / 2 : 0.0

	# gauss_integrator = make_integrator(:gausslegendre, nτ, 0., τm)

	break_points = [
	    2 * (t[end] - t[1]),   # first break after twice the data span
	    25_000.0               # second break at fixed value
	]
	
	nodes_points = [
	    10.0  * break_points[1],                # dense in first panel
	    0.05 * (break_points[2] - break_points[1]), # moderate in middle
	    0.01 * (τm - break_points[2])               # lighter in last
	]
	
	nodes_points = Int.(round.(nodes_points))
	panels = make_integration_panels(0., τm, break_points, nodes_points)
	gauss_integrator = make_integrator(:gausslegendre, panels)
	#setup Boundary Propagator problem
	BP = BoundaryPropagator(Gps, Function[f_src], t; 
							τ_max = τm, integr = gauss_integrator) 

	#solve for convolution of surface and boundary props
	pred_values = zeros(length(Gps), length(t))
	for i in 1:length(Gps)
		
		pred_values[i, :] .= convolve_at(BP, i, t)
	end
	p1 = plot(t, f_src.(t), 
	         label="Atmospheric Source", 
	         color = :black, 
	         linewidth = 3)
	for i in 1:length(depths)
		depth = depths[i]
		gamma = round(Γs[i])
	    plot!(p1, t, pred_values[i, :], 
	    label="Box $i\n z=$depth meters, Γ=$gamma yrs", 
	    color = depth_colors[i], 
	    lw = 4)
	end
	p1
end

# ╔═╡ 9b87a9eb-c2d4-4a52-a529-6d0c6abfee28
	println("Length of weights: ", length(gauss_integrator.weights))


# ╔═╡ e95c8ef3-a123-465d-9ea4-876ebc4d6593
minimum(pred_values)

# ╔═╡ b9d5dccc-16ed-42f5-a1b4-d9600670219f
begin 
    x_pdf = collect(0:0.05:50) 
    p = plot(layout=@layout [a b{0.3w}])
	nz = length(depths)
	
    for i in 1:nz
		curr_depth = depths[i]
        plot!(p[1], x_pdf, Gps[i].(x_pdf), 
			  label="Box $i\n z=$curr_depth meters", 
			  color=depth_colors[i], lw=4, 
			 xlabel = "τ [yrs]", ylabel = "Density [1/yrs]")
        scatter!(p[2], [i], [curr_depth], 
				 label="", 
				 color=depth_colors[i], 
				 marker=:circle, 
				 xlabel = "Box #", ylabel = "Box \"Depth\"")
		if i < nz
	        plot!(p[2], [i, i+1], [depths[i], depths[i+1]], 
				  color=depth_colors[i], 
				  lw=2, label="")
		end
    end
    
    xlims!(p[1], (0, 50))

	yflip!(p[2], true)
    xlims!(p[2], (0.5, length(depths)+0.5))
    p
end

# ╔═╡ d86dbef3-b697-4012-9ef2-a3a2a5a30cc2
Gps

# ╔═╡ 83c0c2e8-8044-4ca7-95b2-a716afa5ea65
x = TracerInverseGaussian(4.0, 4.0)

# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╠═9b87a9eb-c2d4-4a52-a529-6d0c6abfee28
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╠═2971aa56-fb85-40bd-b19c-884d89a57af8
# ╠═e95c8ef3-a123-465d-9ea4-876ebc4d6593
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═d86dbef3-b697-4012-9ef2-a3a2a5a30cc2
# ╠═83c0c2e8-8044-4ca7-95b2-a716afa5ea65
