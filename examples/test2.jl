#!/usr/bin/env julia
"""
Model of boundary propagation with inverse Gaussian distribution.
"""

# Setup environment and imports
import Pkg
Pkg.activate(".")
using TCMGreensFunctions, Plots, Distributions, QuadGK, Interpolations

"""
    boundary_propagator_timeseries(t_vec)
    
Calculate convolution of boundary propagator with surface source
for a vector of time points.

Parameters:
    t_vec: Vector of time points
    
Returns:
    Vector of convolution results
"""
function boundary_propagator_timeseries(Gp::Function, 
    f_atm::Union{<:Function, <:AbstractInterpolation}, t_vec::Vector)
    result = zeros(length(t_vec))
    
    for (i, t) in enumerate(t_vec)
        # Directly compute convolution integral for each time point
        integral, _ = quadgk(tp -> Gp(t - tp) * f_atm(tp), 0, t)
        result[i] = integral
    end
    
    return result
end

# Model parameters
μ = 10.0    # Mean (transport time)
λ = 3.0     # Width (dispersion parameter)

# Define time domain and functions
x_pdf = collect(0:0.05:6)
Gp(x) = pdf(InverseGaussian(μ, λ), x)  # Boundary propagator
f = x_pdf.^2 ./ 2                      # Surface source
f_interp = linear_interpolation(x_pdf, f)
# Calculate and plot results

ocean_values = boundary_propagator_timeseries(Gp, f_interp, x_pdf)
plot(x_pdf, ocean_values, label="Ocean Values")
plot!(x_pdf, f_interp.(x_pdf), label="Atmospheric Source")