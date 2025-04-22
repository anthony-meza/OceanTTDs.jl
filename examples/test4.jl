# Setup environment and imports
import Pkg
Pkg.activate(".")
using TCMGreensFunctions, Plots, Distributions, QuadGK, Interpolations

# Model parameters
μ_0 = 1.5    # Mean (transport time)
λ_0 = 10.0     # Width (dispersion parameter)

# Define time domain and functions
x_pdf = collect(0:0.05:6)
Gp(x, μ, λ) = pdf(InverseGaussian(μ, λ), x)  # Boundary propagator

depth = [0, 5, 20, 30]
decay_scaling  = 25
μs = exp.(depth ./ decay_scaling) .* μ_0
λs = exp.(depth ./ decay_scaling) .* λ_0

Gps = [x -> Gp(x, μ, λ) for (μ, λ) in zip(μs, λs)]

depth_colors = cgrad(:greens, length(depth), categorical=true)  # Create a categorical gradient with fixed colors

p = plot()
for i in 1:length(depth)
    plot!(p, x_pdf, Gps[i].(x_pdf), 
          label="GP $i", 
          color = depth_colors[i], 
          lw = 4)
end
p

f = x_pdf.^2 ./ 2                      # Surface source
f_interp = linear_interpolation(x_pdf, f)

BP = BoundaryPropagator(Gps, f_interp, x_pdf)

ocean_values = boundary_propagator_timeseries(BP)

p = plot(x_pdf, f_interp.(x_pdf), 
         label="Atmospheric Source", 
         color = :black, 
         linewidth = 3)
for i in 1:length(depth)
    plot!(p, x_pdf, ocean_values[i, 1, :], 
    label="Ocean Values $i", 
    color = depth_colors[i], 
    lw = 4)
end
p