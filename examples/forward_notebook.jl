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
	simpsons_slider = @bind N_simp Slider(2:4:100, default=40.0, show_value=true)

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

	Simpsons Integration Bins ($N_{simp}$): $simpsons_slider
	"""
end


# ╔═╡ d93d36a6-a0ab-424a-9100-88cb4c5990af
begin
	Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function

	depths = [0, 5, 100, 500] #depths of boxes we are modeling 
	depth_colors = cgrad(:greens, length(depths), categorical=true)  # Create a categorical gradient with fixed colors
	
	Γs = exp.(depths ./ λ) .* Γ_0 # mean age increases with depth
	Δs = exp.(depths ./ λ) .*Δ_0 # width increases with depth

	Gps = [x -> Gp(x, μ, λ) for (μ, λ) in zip(Γs, Δs)]

end

# ╔═╡ 2971aa56-fb85-40bd-b19c-884d89a57af8
begin
	#define time of integration
	t = 0:40
	# Surface flux rate is defined quadratic 
	@. f(x) = x^2 / 2  

	#setup Boundary Propagator problem
	BP = BoundaryPropagator(Gps, f, t) 

	#solve for convolution of surface and boundary props
	pred_values = boundary_propagator_timeseries(BP, N_simpson = N_simp); nothing
	
	p1 = plot(t, f.(t), 
	         label="Atmospheric Source", 
	         color = :black, 
	         linewidth = 3)
	for i in 1:length(depths)
		depth = depths[i]
		gamma = round(Γs[i])
	    plot!(p1, t, pred_values[i, 1, :], 
	    label="Box $i\n z=$depth meters, Γ=$gamma yrs", 
	    color = depth_colors[i], 
	    lw = 4)
	end
	p1
end

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

# ╔═╡ 83c0c2e8-8044-4ca7-95b2-a716afa5ea65


# ╔═╡ Cell order:
# ╠═6c4083e2-1f27-11f0-1077-e9a0ae678f2d
# ╟─a9445454-daf5-4183-8aba-cb61e8595d06
# ╟─2971aa56-fb85-40bd-b19c-884d89a57af8
# ╠═b9d5dccc-16ed-42f5-a1b4-d9600670219f
# ╠═d93d36a6-a0ab-424a-9100-88cb4c5990af
# ╠═83c0c2e8-8044-4ca7-95b2-a716afa5ea65
