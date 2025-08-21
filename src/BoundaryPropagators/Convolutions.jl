
import Distributions: convolve   # avoid importing `support` to prevent name clashes

#############################
# Evaluation APIs for BoundaryPropagatorTTD
#############################

export convolve_at, convolve_tracer, convolve_all, convolve

### Common implementations

"""
    convolve_at(bp, p, t_obs)

Convolve the shared TTD with tracer `p`'s surface history at a single time `t_obs`.
- Discrete TTD:  finite sum over `bp.τ_support`
- Continuous TTD: quadrature over [0, bp.τ_max] using `bp.integrator`
Adds `bp.C0[p]` at the end.
"""
function convolve_at(bp::BoundaryPropagatorTTD{D,I,T}, p::Integer, t_obs::T) where {D,I,T<:Real}
    (1 ≤ p ≤ length(bp.tracers)) || throw(ArgumentError("tracer index p=$(p) out of bounds"))
    tracer = bp.tracers[p]
    acc = if _is_discrete(bp.TTD)
        _convolve_discrete_sfun(bp.τ_support, bp.TTD, tracer.f_src, t_obs)
    elseif _is_continuous(bp.TTD)
        _convolve_continuous_sfun(bp.TTD, tracer.f_src, t_obs, bp.integrator)
    else
        throw(ArgumentError("TTD must be DiscreteUnivariateDistribution or ContinuousUnivariateDistribution"))
    end
    return acc + bp.C0[p]
end

"""
    convolve_tracer(bp, p)

Evaluate the convolution for tracer `p` at all of its observation times.
Returns a vector with the same length as `bp.tracers[p].t_obs`.
"""
function convolve_tracer(bp::BoundaryPropagatorTTD{D,I,T}, p::Integer) where {D,I,T<:Real}
    times = bp.tracers[p].t_obs
    out   = similar(times)
    for (i, t_obs) in pairs(times)
        out[i] = convolve_at(bp, p, t_obs)
    end
    return out
end

"""
    convolve_all(bp)

Evaluate all the convolution across all surface sources 
at the the observation time)  
`[convolve_tracer(bp, 1), convolve_tracer(bp, 2), …]`.
"""
function convolve_all(bp::BoundaryPropagatorTTD{D,I,T}) where {D,I,T<:Real}
    results = Vector{Vector{T}}(undef, length(bp.tracers))
    for p in eachindex(bp.tracers)
        results[p] = convolve_tracer(bp, p)
    end
    return results
end

#############################
# Standalone "convolve" helpers (single f_src and time list)
#############################



"""
    convolve(TTD::Distribution, f_src::Function, times::Vector{T}; kwargs...) -> Vector{T}

Convenience wrapper that dispatches to the discrete or continuous method.
- If `TTD` is **discrete**, pass `τ_support::Vector{T}`.
- If `TTD` is **continuous**, pass `τ_max::T` and `integrator`.
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

### Discrete Implementations ####
"""
    _convolve_discrete_sfun(τ_support, TTD, f_src, t_obs)

Compute ∑_{τ∈τ_support} pdf(TTD, τ) * f_src(t_obs - τ).
Assumes τ_support ⊆ [0, τ_max] and is sorted.
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

"""
    convolve(TTD::DiscreteUnivariateDistribution,
             f_src::Function,
             times::Vector{T};
             τ_support::Vector{T},
             C0::T = zero(T)) -> Vector{T}

Convolve a **discrete** TTD with a surface history `f_src` at a list of `times`.
- `τ_support` must be a sorted, non-negative vector in the domain you care about.
- Returns a vector with the same length as `times`.
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

### continous implementations
"""
    _convolve_continuous_sfun(TTD, f_src, t_obs, integrator)

Compute ∫₀^{τ_max} pdf(TTD, τ) * f_src(t_obs - τ) dτ
using `integrator.nodes` and `integrator.weights` over [0, τ_max].
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

"""
    convolve(TTD::ContinuousUnivariateDistribution,
             f_src::Function,
             times::Vector{T};
             τ_max::T,
             integrator,
             C0::T = zero(T)) -> Vector{T}

Convolve a **continuous** TTD with a surface history `f_src` at a list of `times`.
Integrates over `[0, τ_max]` using the provided `integrator` (with `.nodes`/`.weights`).
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

