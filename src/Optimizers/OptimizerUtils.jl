d"""
OptimizerUtils.jl

Essential utilities shared across optimization methods.
"""

using ..TracerObservations
using ..InversionResults

export prepare_observations, handle_C0_parameter, tracer_eltype

"""
    tracer_eltype(obs::TracerObservation) -> Type

Returns the element type of the tracer observation data.
"""
tracer_eltype(obs::TracerObservation) = eltype(obs.y_obs)

"""
    prepare_observations(observations) -> (Vector{TracerObservation}, Vector{TracerEstimate})

Standardizes observation input handling across all optimizers.

# Arguments
- `observations`: Either a single TracerObservation or Vector of TracerObservations

# Returns
- Tuple of (observations_vector, estimates_vector)

# Throws
- `ArgumentError`: If any observation lacks a source function (f_src)
"""
function prepare_observations(observations::Union{G, AbstractVector{G}}) where {G<:TracerObservation}
    observations_vec = observations isa AbstractVector ? observations : [observations]
    
    # Validate all observations have source functions
    any(obs -> obs.f_src === nothing, observations_vec) &&
        throw(ArgumentError("All observations must have f_src defined for inversion."))
    
    # Create estimates from observations
    estimates = [TracerEstimate(obs) for obs in observations_vec]
    
    return observations_vec, estimates
end

"""
    handle_C0_parameter(C0, observations_vec) -> Vector

Handles C0 (additive constant) parameter broadcasting.

# Arguments
- `C0`: C0 value (nothing, scalar, or vector)
- `observations_vec`: Vector of TracerObservation objects

# Returns
- Vector of C0 values matching observation count
"""
function handle_C0_parameter(C0, observations_vec)
    T = tracer_eltype(observations_vec[1])
    n_obs = length(observations_vec)
    
    if C0 === nothing
        return zeros(T, n_obs)
    elseif C0 isa AbstractVector
        length(C0) == n_obs || 
            throw(ArgumentError("C0 vector length $(length(C0)) ≠ #observations $n_obs"))
        return Vector{T}(C0)
    else
        return fill(T(C0), n_obs)
    end
end