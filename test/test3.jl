"""
Model of boundary propagation with inverse Gaussian distribution.
"""

# Setup environment and imports
import Pkg
Pkg.activate(".")
using TCMGreensFunctions, Plots, Distributions, QuadGK, Interpolations
"""
BoundaryPropagator

A class for modeling boundary propagation with propagator functions
and atmospheric source terms.

Fields:
    Gp_arr::AbstractMatrix{<:Function} - Matrix of boundary propagator functions
    f_atm::AbstractVector{Union{<:Function, <:AbstractInterpolation}} - Vector of atmospheric source functions
    t_vec::AbstractVector{<:Real} - Vector of time points 
    results::Union{Nothing, AbstractArray{<:Real}} - Cached results
"""
struct BoundaryPropagator
    Gp_arr::AbstractMatrix{<:Function}
    f_atm::AbstractVector{Union{<:Function, <:AbstractInterpolation}}
    t_vec::AbstractVector{<:Real}
    results::Union{Nothing, AbstractArray{<:Real}}
    
    # Flexible constructor that handles different input types
    function BoundaryPropagator(
        Gp_input::Union{<:Function, <:AbstractVector{<:Function}, <:AbstractMatrix{<:Function}}, 
        f_atm_input::Union{<:Function, <:AbstractInterpolation, 
                          <:AbstractVector{Union{<:Function, <:AbstractInterpolation}}}, 
        t_vec::AbstractVector{<:Real}
    )
        # Convert single function to 1×1 matrix
        if isa(Gp_input, Function)
            Gp_arr = fill(Gp_input, 1, 1)
        # Convert vector to n×1 matrix
        elseif isa(Gp_input, AbstractVector)
            Gp_arr = reshape(Gp_input, length(Gp_input), 1)
        else
            Gp_arr = Gp_input
        end
        
        # Convert single atmospheric function to 1-element vector
        if isa(f_atm_input, Union{<:Function, <:AbstractInterpolation})
            f_atm = [f_atm_input]
        else
            f_atm = f_atm_input
        end
        
        # Check that dimensions match
        n_rows, n_cols = size(Gp_arr)
        if n_cols != length(f_atm)
            error("Dimension mismatch: Gp_arr has $(n_cols) columns but f_atm has $(length(f_atm)) elements")
        end
        
        return new(Gp_arr, f_atm, t_vec, nothing)
    end
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

BP = BoundaryPropagator(Gps, f_interp, x_pdf)

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
function boundary_propagator_timeseries(Gp_vec::AbstractMatrix{<:Function}, 
                                        f_atm::AbstractVector{Union{<:Function, <:AbstractInterpolation}}, 
                                          t_vec::AbstractVector{<:Real})
    # Initialize result matrix
    N, Nₛ = size(Gp_vec)
    Nt =  length(t_vec)
    
    integration_results = zeros(N, Nt)
    
    # Iterate through each propagator function
    for (j, Gp) in enumerate(Gp_vec)
        result[j, :] .= boundary_propagator_timeseries(Gp, f_atm, t_vec)
    end
    
    return result
end


# Calculate and plot results
ocean_values = boundary_propagator_timeseries(Gps, f_interp, x_pdf)

plot(x_pdf, f_interp.(x_pdf), label="Atmospheric Source")
plot!(x_pdf, ocean_values[1, :], label="Ocean Values 1")
plot!(x_pdf, ocean_values[2, :], label="Ocean Values 2")
plot!(x_pdf, ocean_values[3, :], label="Ocean Values 3")
