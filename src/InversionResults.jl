module InversionResults

using ..TracerObservations
using Distributions

export InversionResult

"""
    InversionResult{T}

Flexible container for inversion/optimization results that can handle different types of distributions
and integration contexts.

# Fields
- `parameters::AbstractVector`: Estimated parameters. Interpretation depends on method:
  - `:inverse_gaussian`: [Γ, Δ] 
  - `:inverse_gaussian_equalvars`: [Γ]
  - `:max_entropy`: [λ₁, λ₂, ..., λₙ] (Lagrange multipliers)
- `obs_estimates::Vector{TracerEstimate{T}}`: Updated TracerEstimate objects with predictions
- `distribution::Union{Distribution, AbstractVector{<:Distribution}, Function, Nothing}`: Fitted distribution(s) or template function
- `integrator::Union{Nothing, Any}`: Integration method/context used
- `support::Union{Nothing, AbstractVector}`: Discrete support points if applicable
- `method::Symbol`: Method used (e.g., :inverse_gaussian, :max_entropy, :inverse_gaussian_equalvars)
- `optimizer_output`: Raw optimization results from the solver
"""
struct InversionResult{T}
    parameters::Union{AbstractVector, Nothing}
    obs_estimates::Vector{TracerEstimate{T}}
    distribution::Union{Distribution, AbstractVector, Function, Nothing}
    integrator::Union{Nothing, Any}
    support::Union{Nothing, AbstractVector}
    method::Symbol
    optimizer_output
end

# Keyword constructor for clarity and flexibility
function InversionResult(;
    parameters::Union{AbstractVector, Nothing} = nothing,
    obs_estimates::Vector{TracerEstimate{T}},
    distribution::Union{Distribution, AbstractVector, Function, Nothing} = nothing,
    integrator = nothing,
    support::Union{Nothing, AbstractVector} = nothing,
    method::Symbol,
    optimizer_output = nothing
) where {T}
    InversionResult{T}(parameters, obs_estimates, distribution, integrator, support, method, optimizer_output)
end

end