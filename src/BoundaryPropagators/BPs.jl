# include("Integrator.jl")
# include("Tracers.jl")


#############################
# Internal utilities
#############################
_is_discrete(d)   = d isa DiscreteUnivariateDistribution
_is_continuous(d) = d isa ContinuousUnivariateDistribution

# Reasonable default for τ_max from tracer timing span
_default_τmax(trs) = maximum(getfield.(trs, :times)) - minimum(getfield.(trs, :times))

# Validate a discrete support vector
function _validate_discrete_support(τ_support::Vector{T}, τ_max::T) where {T<:Real}
    isempty(τ_support) && throw(ArgumentError("τ_support must be non-empty for a discrete TTD"))
    any(<(0), τ_support) && throw(ArgumentError("τ_support must be non-negative"))
    any(>(τ_max), τ_support) && throw(ArgumentError("τ_support must lie within [0, τ_max]"))
    issorted(τ_support) || throw(ArgumentError("τ_support must be sorted ascending"))
    return nothing
end

#############################
# BoundaryPropagatorTTD
#############################
"""
    BoundaryPropagatorTTD

Shared TTD (continuous or discrete) for all tracers.

Fields
- `TTD`        :: D (any Distribution; support assumed τ ≥ 0)
- `tracers`    :: Vector{TracerObservation{T}}  (each has its own `times` and `f_src`)
- `C0`         :: Vector{T}  (additive constant per tracer)
- `integrator` :: I          (only used if TTD is continuous)
- `τ_max`      :: T          (upper limit for τ; lower limit is 0)
- `τ_support`  :: Vector{T}  (required if TTD is discrete; empty for continuous)
"""
struct BoundaryPropagatorTTD{D<:Distribution,I,T<:Real}
    TTD        :: D
    tracers    :: Vector{TracerObservation{T}}  # accepts mixed f_src function types
    C0         :: Vector{T}
    integrator :: I
    τ_max      :: T
    τ_support  :: Vector{T}
end

# --- DISCRETE constructor (τ_support is required; integrator ignored) ---
function BoundaryPropagatorTTD(
    TTD::D,
    tracers::Vector{TracerObservation{T}},
    τ_support::Vector{T},
    integrator::I;                         # pass `nothing` for clarity
    C0    = nothing,
    τ_max::T = _default_τmax(tracers)
) where {D<:DiscreteUnivariateDistribution, I, T<:Real}
    _validate_discrete_support(τ_support, τ_max)
    M = length(tracers)
    C0_vec = C0 === nothing ? zeros(T, M) : (C0 isa Number ? fill(C0, M) : C0)
    length(C0_vec) == M || throw(ArgumentError("C0 length $(length(C0_vec)) ≠ #tracers $M"))
    return BoundaryPropagatorTTD{D,typeof(integrator),T}(
        TTD, Vector{TracerObservation{T}}(tracers), C0_vec, integrator, τ_max, τ_support
    )
end

# Convenience: discrete TTD w/out integrator argument
function BoundaryPropagatorTTD(
    TTD::D,
    tracers::Vector{TracerObservation{T}},
    τ_support::Vector{T};
    C0    = nothing,
    τ_max::T = _default_τmax(tracers)
) where {D<:DiscreteUnivariateDistribution, T<:Real}
    return BoundaryPropagatorTTD(TTD, tracers, τ_support, nothing; C0=C0, τ_max=τ_max)
end

# --- CONTINUOUS constructor (integrator is required; τ_support unused) ---
function BoundaryPropagatorTTD(
    TTD::D,
    tracers::Vector{TracerObservation{T}},
    integrator::I;
    C0    = nothing,
    τ_max::T = _default_τmax(tracers)
) where {D<:ContinuousUnivariateDistribution, I, T<:Real}
    M = length(tracers)
    C0_vec = C0 === nothing ? zeros(T, M) : (C0 isa Number ? fill(C0, M) : C0)
    length(C0_vec) == M || throw(ArgumentError("C0 length $(length(C0_vec)) ≠ #tracers $M"))
    return BoundaryPropagatorTTD{D,typeof(integrator),T}(
        TTD, Vector{TracerObservation{T}}(tracers), C0_vec, integrator, τ_max, T[]
    )
end


# using LinearAlgebra
# τ_support = collect(0.0:0.5:10.0)
# pmf       = normalize(exp.(-τ_support ./ 3), 1)
# d_disc    = Categorical(pmf)

# t1   = collect(0.0:0.25:20.0); sfc1 = Δt -> exp(-((Δt - 5.0)/1.3)^2)
# t2   = collect(0.0:0.5:30.0);  sfc2 = Δt -> exp(-0.1*max(Δt,0)) * sin(0.6*Δt)
# TracerObservationTimes(t, f_src) = TracerObservation(t, NaN .+ zero(t), f_src = f_src)
# trs  = [TracerObservationTimes(t1, sfc1), TracerObservationTimes(t2, sfc2)]


# bp_d = BoundaryPropagatorTTD(d_disc, trs, τ_support; τ_max=maximum(τ_support), C0=0.0)
# y1d  = convolve_tracer(bp_d, 1)
# y2d  = convolve_tracer(bp_d, 2)


# t1

# convolve_all(bp_d)