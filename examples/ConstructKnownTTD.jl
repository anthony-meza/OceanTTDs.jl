"""
ConstructKnownTTD.jl

Utility functions for creating known TTDs and synthetic observations for examples.
This module provides standardized setups that can be reused across different examples.
"""

using OceanTTDs
using Statistics, Distributions, Random
using StatsBase: sample

"""
    setup_inverse_gaussian_TTD(Γ, Δ, source_func; times=nothing, τ_max=250_000.0)

Create an Inverse Gaussian TTD setup with specified parameters.

# Arguments
- `Γ::Real`: Mean transit time
- `Δ::Real`: Width parameter
- `source_func::Function`: Source function for boundary conditions
- `times=nothing`: Time points for observations (defaults to 1:5:250)
- `τ_max::Real=250_000.0`: Maximum transit time to consider

# Returns
- Named tuple containing TTD, source function, times, integrator, and parameters
"""
function setup_inverse_gaussian_TTD(Γ, Δ, source_func; times=nothing, τ_max=250_000.0)
    # Default time points if not provided
    if times === nothing
        times = collect(1.0:5.0:250.0)
    end
    
    # Create TTD
    gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ, Δ))
    
    # Create integrator with standard break points
    break_points = [1.2 * (times[end] - times[1]), 25_000.0]
    nodes_points = Int.(round.([
        5.0 * break_points[1],                      # Dense in recent times
        0.1 * (break_points[2] - break_points[1]),  # Moderate in middle range
        0.01 * (τ_max - break_points[2])            # Sparse in ancient times
    ]))
    
    panels = make_integration_panels(0., τ_max, break_points, nodes_points)
    integrator = make_integrator(:gausslegendre, panels)
    
    return (
        ttd = gaussian_ttd,
        source = source_func,
        times = times,
        integrator = integrator,
        τ_nodes = integrator.nodes,
        τ_max = τ_max,
        Γ_true = Γ,
        Δ_true = Δ,
        parameters = (Γ, Δ)
    )
end

"""
    create_synthetic_observations(ttd_setup; n_obs=nothing, noise_level=1.0, seed=nothing)

Create synthetic tracer observations from a known TTD setup.

# Arguments
- `ttd_setup`: Output from `setup_inverse_gaussian_TTD()`
- `n_obs=nothing`: Number of observation points to sample (if nothing, uses all time points)
- `noise_level::Union{Real,Vector}=1.0`: Standard deviation of observation noise. Can be:
  - Scalar: same noise level for all observations
  - Vector of length `n_obs`: individual noise level for each observation
- `seed=nothing`: Random seed for reproducibility

# Returns
- `TracerObservation` with synthetic data and added noise
"""
function create_synthetic_observations(ttd_setup; n_obs=nothing, noise_level=1.0, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Determine observation times
    if n_obs === nothing
        tobs = ttd_setup.times
        n_actual = length(tobs)
    else
        tobs = sort(sample(ttd_setup.times, n_obs, replace=false))
        n_actual = n_obs
    end
    
    # Generate true observations
    true_observations = convolve(ttd_setup.ttd, ttd_setup.source, tobs; 
                                τ_max=ttd_setup.τ_max, 
                                integrator=ttd_setup.integrator)
    
    # Handle noise level (scalar or vector)
    if isa(noise_level, AbstractVector)
        length(noise_level) == n_actual || throw(ArgumentError("noise_level vector must have length $n_actual"))
        σ_obs = collect(noise_level)
        noise_vec = σ_obs .* randn(n_actual)
    else
        σ_obs = fill(noise_level, n_actual)
        noise_vec = noise_level .* randn(n_actual)
    end
    
    # Add noise
    contaminated_observations = true_observations .+ noise_vec
    
    # Create observation object
    tracer_obs = TracerObservation(tobs, contaminated_observations; 
                                  σ_obs = σ_obs, 
                                  f_src = ttd_setup.source)
    
    return tracer_obs
end

# Export the main functions
export setup_inverse_gaussian_TTD, create_synthetic_observations