#############################
# MaxEnt TTD (Tracer-based) — DiscreteUnivariateDistribution
# Optional PDF cache (default: true)
#############################

export LambdaIndexMap, MaxEntTTD, MaxEntTTDStable

# -------------------------------
# LambdaIndexMap (bookkeeping for flat λ)
# -------------------------------

"""
    LambdaIndexMap(tracers)

Map each tracer to the slice of the flat λ that corresponds to its times.
`ranges[k]` is the unit range of indices in λ for tracer k.
"""
struct LambdaIndexMap{I<:AbstractVector{UnitRange{Int}}}
    ranges::I
end

function LambdaIndexMap(tracers::AbstractVector{<:TracerObservation})
    offs = Int[1]
    for tr in tracers
        push!(offs, last(offs) + length(tr.t_obs))
    end
    ranges = [ offs[k] : offs[k+1]-1 for k in 1:length(tracers) ]
    return LambdaIndexMap(ranges)
end

# -------------------------------
# MaxEnt log/numerators (per τ)
# -------------------------------

"""
    maxent_lognumerator(τ, λ, tracers, λmap, prior; debug_checks=false)

Compute:
    log μ(τ) - ∑_k ∑_j λ[idx(k,j)] * F_k( t_{k,j} - τ )
"""
function maxent_lognumerator(
    τ,
    λ::AbstractVector,
    tracers::AbstractVector{<:TracerObservation},
    λmap::LambdaIndexMap,
    prior;
    debug_checks::Bool=false
)
    length(λmap.ranges) == length(tracers) ||
        throw(ArgumentError("LambdaIndexMap length must match number of tracers"))
    expected_len = sum(length(tr.t_obs) for tr in tracers)
    length(λ) == expected_len ||
        throw(ArgumentError("length(λ) must equal total number of observation times across tracers"))

    Tτ = eltype(tracers[1].t_obs)
    s  = zero(Tτ)

    @inbounds for k in eachindex(tracers)
        tr    = tracers[k]
        times = tr.t_obs
        r     = λmap.ranges[k]
        (length(r) == length(times)) || throw(ArgumentError("λ slice and times mismatch for tracer $k"))

        f = tr.f_src
        @inbounds for (j, t) in enumerate(times)
            Δt = t - τ
            v  = f(Δt)
            if debug_checks
                @assert _is_finite_real(v) "f_src produced non-finite/non-Real at Δt=$Δt"
            end
            s += λ[r[j]] * v
        end
    end

    return log(prior(τ)) - s
end

maxent_numerator(τ, λ, tracers, λmap, prior; debug_checks=false) =
    exp(maxent_lognumerator(τ, λ, tracers, λmap, prior; debug_checks=debug_checks))

# -------------------------------
# Distribution types (optional pdf cache)
# -------------------------------

struct MaxEntTTDStable{T,Fprior,LType,TrType,MapType} <: DiscreteUnivariateDistribution
    λ::LType
    tracers::TrType
    prior::Fprior
    support::Vector{T}
    weights::Union{Nothing,Vector{T}}
    λmap::MapType
    Zlog::T
    indexmap::Dict{T,Int}
    probs::Union{Nothing,Vector{T}}     # cached pdf over support (optional)
end

# -------------------------------
# Unified constructor (cache_pdf default true)
# -------------------------------

"""
    MaxEntTTD(λ, tracers, prior, support;
              weights=nothing, implementation=:stable,
              cache_pdf=true, debug_checks=false)

Build a discrete MaxEnt TTD over `support` with one flat λ (one per observation time).
If `cache_pdf==true`, precomputes and stores `probs` (length == length(support)).
"""
function MaxEntTTD(
    λ::AbstractVector,
    tracers::AbstractVector{<:TracerObservation},
    prior,
    support::AbstractVector{T};
    weights::Union{Nothing,AbstractVector{T}} = nothing,
    cache_pdf::Bool = true,
    debug_checks::Bool = false,
) where {T<:Real}

    λmap    = LambdaIndexMap(tracers)
    supp    = collect(support)
    index   = Dict(τ => i for (i, τ) in enumerate(supp))
    w       = weights
    logw    = w === nothing ? zeros(T, length(supp)) : log.(w)

    LogNum = τ -> maxent_lognumerator(τ, λ, tracers, λmap, prior; debug_checks=debug_checks)
    a      = LogNum.(supp)
    Zlog   = logsumexp_with_logw(a, logw)
    probs  = cache_pdf ? (@. exp(a - Zlog)) : nothing
    return MaxEntTTDStable{T,typeof(prior),typeof(λ),typeof(tracers),typeof(λmap)}(
        λ, tracers, prior, supp, w, λmap, Zlog, index, probs)
end

# -------------------------------
# Distributions.jl interface
# -------------------------------

# Support & membership
support(d::Union{MaxEntTTDStable}) = d.support
insupport(d::Union{MaxEntTTDStable}, x::Real) = haskey(d.indexmap, x)

# PDF / LOGPDF — use cache if present, else compute on the fly
function pdf(d::MaxEntTTDStable, τ::Real)
    i = get(d.indexmap, τ, 0)
    if i == 0
        return zero(eltype(d.support))
    end
    if d.probs !== nothing
        return d.probs[i]
    else
        ℓ = maxent_lognumerator(τ, d.λ, d.tracers, d.λmap, d.prior)
        return exp(ℓ - d.Zlog)
    end
end

function logpdf(d::MaxEntTTDStable, τ::Real)
    i = get(d.indexmap, τ, 0)
    if i == 0
        return -Inf
    end
    if d.probs !== nothing
        return log(d.probs[i])
    else
        ℓ = maxent_lognumerator(τ, d.λ, d.tracers, d.λmap, d.prior)
        return ℓ - d.Zlog
    end
end



# # -------------------------------
# # Example usage (single f_src per tracer)
# # -------------------------------

# # Surface concentration histories (defined on ℝ)
# f1 = Δt -> exp(-abs(Δt)/2)
# fA = Δt -> exp(-abs(Δt))
# fB = Δt -> 1/(1 + Δt^2)
# fAB = Δt -> fA(Δt) + fB(Δt)   # combine if you want multiple components

# tr1 = TracerObservation([2.0, 3.0], NaN .* [2.0, 3.0], f_src = f1)
# tr2 = TracerObservation([7.5], NaN .* [7.5], f_src = fAB)
# tracers = [tr1, tr2]

# λ      = 0. .* [0.10, -0.05, 0.02]              # one per observation across all tracers
# prior  = τ -> 1.              # base measure μ(τ)
# supp   = collect(0.0:0.0001:50.0)          # discrete support for the TTD

# d = MaxEntTTD(λ, tracers, prior, supp; cache_pdf=true)
# sum(pdf(d))
# # Evaluate
# pdf.(d, supp);
# cdf(d, 50.0)
# mean(d), var(d)

