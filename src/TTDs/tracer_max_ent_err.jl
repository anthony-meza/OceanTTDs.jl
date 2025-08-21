#############################
# MaxEntTTDErr — DiscreteUnivariateDistribution
# Optional PDF cache (default: true)
#############################

export MaxEntTTDErr, MaxEntTTDErrStable

# -------------------------------
# MaxEntTTDErr log/numerator (per τ)
# -------------------------------

"""
    maxent_err_lognumerator(τ, λ, prior; debug_checks=false)

Compute:
    log μ(τ) - λ * τ
where μ is the base measure (`prior`).
"""
function maxent_err_lognumerator(
    τ,
    λ::Real,
    prior;
    debug_checks::Bool=false
)
    vμ = prior(τ)
    if debug_checks
        @assert _is_finite_real(vμ) && vμ > 0 "prior(τ) must be positive, finite"
    end
    return log(vμ) - λ*τ
end

maxent_err_numerator(τ, λ, prior; debug_checks=false) =
    exp(maxent_err_lognumerator(τ, λ, prior; debug_checks=debug_checks))

# -------------------------------
# Distribution types (optional pdf cache)
# -------------------------------

struct MaxEntTTDErrStable{T,Fprior,LType} <: DiscreteUnivariateDistribution
    λ::LType
    prior::Fprior
    support::Vector{T}
    weights::Union{Nothing,Vector{T}}
    Zlog::T
    indexmap::Dict{T,Int}
    probs::Union{Nothing,Vector{T}}
end

# -------------------------------
# Unified constructor (cache_pdf default true)
# -------------------------------

"""
    MaxEntTTDErr(λ, prior, support;
              weights=nothing,
              cache_pdf=true, debug_checks=false)

Build a discrete MaxEnt **error** distribution over `support` for a single λ.
If `cache_pdf==true`, precomputes and stores `probs` (length == length(support)).
"""
function MaxEntTTDErr(
    λ::Real,
    prior,
    support::AbstractVector{T};
    weights::Union{Nothing,AbstractVector{T}} = nothing,
    cache_pdf::Bool = true,
    debug_checks::Bool = false,
) where {T<:Real}

    @assert !isempty(support) "support cannot be empty"
    @assert issorted(support) "support must be sorted ascending"
    supp  = collect(support)
    index = Dict(τ => i for (i, τ) in enumerate(supp))

    # validate / prepare weights
    w    = weights
    logw = w === nothing ? zeros(T, length(supp)) : log.(w)
    if w !== nothing
        @assert length(w) == length(supp) "weights must match support length"
        @assert all(isfinite, w) && all(>=(zero(T)), w) "weights must be finite and ≥ 0"
        @assert any(>(zero(T)), w) "at least one weight must be > 0"
    end

    
    #calculating the numerator and denominator of the error distribution in 
    #log space for numerical stability. 
    LogNum = τ -> maxent_err_lognumerator(τ, λ, prior; debug_checks=debug_checks)
    a      = LogNum.(supp)
    Zlog   = logsumexp_with_logw(a, logw)

    probs = if cache_pdf
        if w === nothing
            @. exp(a - Zlog)
        else
            @. exp((a + logw) - Zlog)
        end
    else
        nothing
    end

    return MaxEntTTDErrStable{T,typeof(prior),typeof(λ)}(
        λ, prior, supp, w, Zlog, index, probs)


end

# -------------------------------
# Distributions.jl interface
# -------------------------------

# Support & membership
support(d::Union{MaxEntTTDErrStable}) = d.support
insupport(d::Union{MaxEntTTDErrStable}, x::Real) = haskey(d.indexmap, x)

# PDF / LOGPDF — use cache if present, else compute on the fly (include weights!)
function pdf(d::MaxEntTTDErrStable, τ::Real)
    i = get(d.indexmap, τ, 0)
    if i == 0
        return zero(eltype(d.support))
    end
    if d.probs !== nothing
        return d.probs[i]
    else
        Le = maxent_err_lognumerator(τ, d.λ, d.prior)
        lw = d.weights === nothing ? zero(Le) : log(d.weights[i])
        return exp(Le + lw - d.Zlog)
    end
end



# CDF — sums cached probs if present; otherwise calls pdf on the fly
function cdf(d::Union{MaxEntTTDErrStable}, τ::Real)
    k = searchsortedlast(d.support, τ)
    if k <= 0
        return zero(eltype(d.support))
    elseif k >= length(d.support)
        return one(eltype(d.support))
    end
    if d.probs !== nothing
        return sum(@view d.probs[1:k])
    else
        acc = zero(eltype(d.support))
        @inbounds for i in 1:k
            acc += pdf(d, d.support[i])
        end
        return acc
    end
end

# Moments
function mean(d::Union{MaxEntTTDErrStable})
    if d.probs === nothing
        p = pdf.(Ref(d), d.support)
        return sum(@. d.support * p)
    else
        return sum(@. d.support * d.probs)
    end
end

function var(d::Union{MaxEntTTDErrStable})
    μ = mean(d)
    if d.probs === nothing
        p = pdf.(Ref(d), d.support)
        return sum(@. (d.support - μ)^2 * p)
    else
        return sum(@. (d.support - μ)^2 * d.probs)
    end
end

# function logpdf(d::MaxEntTTDErrStable, τ::Real)
#     i = get(d.indexmap, τ, 0)
#     if i == 0
#         return -Inf
#     end
#     if d.probs !== nothing
#         return log(d.probs[i])
#     else
#         Le = maxent_err_lognumerator(τ, d.λ, d.prior)
#         lw = d.weights === nothing ? zero(Le) : log(d.weights[i])
#         return Le + lw - d.Zlog
#     end
# end

# -------------------------------
# Example usage
# -------------------------------

# # base measure (μ): positive on support
# prior = τ -> 1              # e.g., Laplace-like base measure
# supp  = collect(0.0:0.001:10.0)
# λ     = 0.0
# # optional quadrature weights (e.g., uniform grid here)
# # w   = fill(step(supp), length(supp))
# w = nothing

# d = MaxEntTTDErr(λ, prior, supp; implementation=:stable, cache_pdf=true)


# Evaluate
# pdf.(d, supp);
# cdf(d, 5.0)
# mean(d), var(d)
# @assert isapprox(sum(pdf.(d, supp)), 1.0; rtol=1e-10, atol=1e-12)
