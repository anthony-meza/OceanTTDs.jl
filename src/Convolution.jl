import Distributions: convolve   # avoid importing `support` to prevent name clashes

export convolve

# Helper functions for distribution type checking  
_is_discrete(d)   = d isa DiscreteUnivariateDistribution
_is_continuous(d) = d isa ContinuousUnivariateDistribution

"""
    convolve(TTD, f_src, times; kwargs...)

Convolve a Transit Time Distribution with a source function at observation times.

This is the main convolution interface used by all optimization methods.
Automatically dispatches to the appropriate method based on TTD type:

- **Vector TTD**: Direct discrete convolution with τ_support
- **Discrete Distribution**: Uses pdf() values with τ_support  
- **Continuous Distribution**: Numerical integration with integrator

# Arguments
- `TTD`: Transit time distribution (Vector, DiscreteUnivariateDistribution, or ContinuousUnivariateDistribution)
- `f_src`: Source function f(t) 
- `times`: Vector of observation times
- Method-specific keywords (see individual methods)

# Examples
```julia
# Discrete vector convolution (MaxEnt, TCM)
τs = collect(0:100:10000)
probs = [0.1, 0.3, 0.4, 0.2]  # Probability weights
result = convolve(probs, f_src, times; τ_support=τs)

# Continuous distribution convolution (Inverse Gaussian)  
dist = InverseGaussian(μ, λ)
result = convolve(dist, f_src, times; τ_max=10000.0, integrator=integrator)
```
"""
function convolve(
    TTD::Distribution,
    f_src::Function,
    times::Vector{T};
    kwargs...
) where {T<:Real}
    if _is_discrete(TTD)
        return convolve(TTD::DiscreteUnivariateDistribution, f_src, times; kwargs...)
    elseif _is_continuous(TTD)
        return convolve(TTD::ContinuousUnivariateDistribution, f_src, times; kwargs...)
    else
        throw(ArgumentError("TTD must be a DiscreteUnivariateDistribution or ContinuousUnivariateDistribution"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════════
# DISCRETE CONVOLUTION METHODS
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    convolve(TTD::Vector, f_src::Function, t_obs::Vector; 
             τ_support::Vector, C0::Real = 0.0)

Convolve discrete TTD probability weights with a source function.

Computes: ŷ(t) = ∑ⱼ TTD[j] * f_src(t - τⱼ) + C0

Used by MaxEnt and TCM optimization methods.

# Arguments
- `TTD`: Vector of probability weights/probabilities  
- `f_src`: Source function f(t)
- `t_obs`: Observation times
- `τ_support`: Transit time support points corresponding to TTD entries
- `C0`: Initial condition - tracer concentration at t=0 (default: 0.0)

# Returns
Vector of convolution results at each observation time.

# Example
```julia
τs = collect(1:100) 
TTD_weights = [0.1, 0.3, 0.4, 0.2]  # Discrete probabilities
f_src = t -> (t >= 0) ? 1.0 : 0.0    # Unit step function
t_obs = [10.0, 20.0, 30.0]

result = convolve(TTD_weights, f_src, t_obs; τ_support=τs)
```
"""
function convolve(TTD::Vector{G}, f_src::Function, t_obs::Vector{T}; 
                  τ_support::Vector, C0::Real = 0.0) where {G, T}
    out_type = promote_type(G, T) #promotion required to work with JuMP
    out = Vector{out_type}(undef, length(t_obs))
    for (i, t) in pairs(t_obs)
        out[i] = _convolve_vector(TTD, f_src, t; τ_support=τ_support, C0=C0)
    end
    return out
end

"""
    _convolve_vector(TTD, f_src, t_obs; τ_support, C0=0.0)

Core discrete convolution computation for a single observation time.

Computes: ŷ(t) = ∑ⱼ TTD[j] * f_src(t - τⱼ) + C0

# Arguments
- `TTD`: Transit time distribution weights
- `f_src`: Source function f(t) 
- `t_obs`: Single observation time
- `τ_support`: Transit time support points (must be sorted)
- `C0`: Initial condition - background tracer concentration (default: 0.0)

Note: Assumes τ_support is sorted in increasing order for causality.
"""
function _convolve_vector(TTD::Vector{G}, f_src::Function, t_obs::T;
                          τ_support::AbstractVector, C0::Real = 0.0) where {T, G}
    # Initialize accumulator with proper type promotion
    acc = zero(promote_type(G, T))
    for (j, τ) in enumerate(τ_support)
        if t_obs >= τ  # Causality constraint
            acc += TTD[j] * f_src(t_obs - τ)
        end
    end
    return acc + C0
end

"""
    convolve(TTD::DiscreteUnivariateDistribution, f_src::Function, times::Vector;
             τ_support::Vector, C0 = 0.0)

Convolve discrete distribution with source function using pdf values.

# Arguments  
- `TTD`: Discrete univariate distribution
- `f_src`: Source function f(t)
- `times`: Vector of observation times
- `τ_support`: Sorted, non-negative support points for evaluation
- `C0`: Initial condition - tracer concentration at t=0 (default: 0.0)

# Returns
Vector with same length as `times` containing convolution results.
"""
function convolve(
    TTD::DiscreteUnivariateDistribution,
    f_src::Function,
    times::Vector{T};
    τ_support::Vector{T},
    C0::T = zero(T),
) where {T<:Real}
    isempty(τ_support) && throw(ArgumentError("τ_support must be non-empty for a discrete TTD"))
    !issorted(τ_support) && throw(ArgumentError("τ_support must be sorted ascending"))
    any(<(0), τ_support) && throw(ArgumentError("τ_support must be non-negative"))

    out = similar(times)
    for (i, t_obs) in pairs(times)
        out[i] = _convolve_discrete_sfun(τ_support, TTD, f_src, t_obs) + C0
    end
    return out
end

"""
    _convolve_discrete_sfun(τ_support, TTD, f_src, t_obs)

Compute ∑_{τ∈τ_support} pdf(TTD, τ) * f_src(t_obs - τ).
"""
function _convolve_discrete_sfun(
    τ_support::AbstractVector{T},
    TTD::DiscreteUnivariateDistribution,
    f_src::Function,
    t_obs::T
) where {T<:Real}
    acc = zero(T)
    @inbounds for τ in τ_support
        pτ = pdf(TTD, τ)
        if pτ != 0
            acc += pτ * f_src(t_obs - τ)
        end
    end
    return acc
end

# ═══════════════════════════════════════════════════════════════════════════════════
# CONTINUOUS CONVOLUTION METHODS
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    convolve(TTD::ContinuousUnivariateDistribution, f_src::Function, times::Vector;
             τ_max, integrator, C0 = 0.0)

Convolve continuous distribution with source function using numerical integration.

Used by Inverse Gaussian optimization method.

# Arguments
- `TTD`: Continuous univariate distribution
- `f_src`: Source function f(t)
- `times`: Vector of observation times
- `τ_max`: Maximum transit time for integration domain [0, τ_max]
- `integrator`: Numerical integrator with `.nodes` and `.weights` fields
- `C0`: Initial condition - tracer concentration at t=0 (default: 0.0)

# Returns
Vector with same length as `times` containing convolution results.

# Example
```julia
using Distributions
dist = InverseGaussian(300.0, 600.0)  # Mean=300, shape=600
integrator = make_gausslegendre_integrator(100, 0.0, 10000.0)
result = convolve(dist, f_src, times; τ_max=10000.0, integrator=integrator)
```
"""
function convolve(
    TTD::ContinuousUnivariateDistribution,
    f_src::Function,
    times::Vector{T};
    τ_max::T,
    integrator,
    C0::T = zero(T),
) where {T<:Real}
    τ_max > zero(T) || throw(ArgumentError("τ_max must be positive for a continuous TTD"))
    out = similar(times)
    for (i, t_obs) in pairs(times)
        out[i] = _convolve_continuous_sfun(TTD, f_src, t_obs, integrator) + C0
    end
    return out
end

"""
    _convolve_continuous_sfun(TTD, f_src, t_obs, integrator)

Compute ∫₀^{τ_max} pdf(TTD, τ) * f_src(t_obs - τ) dτ using numerical integration.

The full convolution is: ŷ(t) = ∫₀^{τ_max} TTD(τ) * f_src(t - τ) dτ + C0
"""
function _convolve_continuous_sfun(
    TTD::ContinuousUnivariateDistribution,
    f_src::Function,
    t_obs::T,
    integrator
) where {T<:Real}
    acc = zero(T)
    @inbounds for k in eachindex(integrator.nodes)
        τk = integrator.nodes[k]
        wk = integrator.weights[k]
        pτ = pdf(TTD, τk)
        if pτ != 0
            acc += pτ * f_src(t_obs - τk) * wk
        end
    end
    return acc
end