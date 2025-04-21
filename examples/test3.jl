"""
Model of boundary propagation with inverse Gaussian distribution.
"""

# Setup environment and imports
import Pkg
Pkg.activate(".")
using TCMGreensFunctions, Plots, Distributions, QuadGK, Interpolations

"""
    boundary_propagator_timeseries(Gp_vec, f_atm, t_vec)
    
Calculate convolution of boundary propagators with atmospheric source
for a vector of time points.

Parameters:
    Gp_vec: Vector of boundary propagator functions
    f_atm: Atmospheric source function (interpolated)
    t_vec: Vector of time points
    
Returns:
    Matrix where each row corresponds to a propagator function and
    each column corresponds to a time point
"""
function boundary_propagator_timeseries(Gp_vec::Vector{<:Function}, 
                                        f_atm::Union{<:Function, <:AbstractInterpolation}, 
                                          t_vec::AbstractVector{<:Real})
    # Initialize result matrix
    result = zeros(length(Gp_vec), length(t_vec))
    
    # Iterate through each propagator function
    for (j, Gp) in enumerate(Gp_vec)
        result[j, :] .= boundary_propagator_timeseries(Gp, f_atm, t_vec)
        # Iterate through each time point
        # for (i, t) in enumerate(t_vec)
        #     # Compute convolution integral
        #     integral, _ = quadgk(tp -> Gp(t - tp) * f_atm(tp), 0, t)
        #     result[j, i] = integral
        # end
    end
    
    return result
end

# Model parameters
μ_0 = 10.0    # Mean (transport time)
λ_0 = 3.0     # Width (dispersion parameter)

# Define time domain and functions
x_pdf = collect(0:0.05:6)
Gp(x, μ, λ) = pdf(InverseGaussian(μ, λ), x)  # Boundary propagator
Gps = [x -> Gp(x, μ, λ), x -> Gp(x, 2 * μ, 2 * λ), x -> Gp(x, 3 * μ, 4 * λ)]
f = x_pdf.^2 ./ 2                      # Surface source
f_interp = linear_interpolation(x_pdf, f)

# Calculate and plot results
ocean_values = boundary_propagator_timeseries(Gps, f_interp, x_pdf)

plot(x_pdf, f_interp.(x_pdf), label="Atmospheric Source")
plot!(x_pdf, ocean_values[1, :], label="Ocean Values 1")
plot!(x_pdf, ocean_values[2, :], label="Ocean Values 2")
plot!(x_pdf, ocean_values[3, :], label="Ocean Values 3")
