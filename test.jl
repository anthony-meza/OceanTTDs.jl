"""
    struct CumTrapWeights{T}

Holds the precomputed trapezoidal‐rule segment weights
for an (irregular) grid `t` of length `n`.
- w0[j] is the weight multiplying f[j] on [t[j],t[j+1]]
- w1[j] is the weight multiplying f[j+1] on [t[j],t[j+1]]
"""
struct CumTrapWeights{T}
    w0::Vector{T}
    w1::Vector{T}
end

"""
    precompute_weights(t::Vector{<:Real}) -> CumTrapWeights

Given a strictly increasing grid `t` of length `n`, build the
trapezoidal weights for each segment [t[j],t[j+1]]:
  dt = t[j+1] - t[j]
  w0[j] =  dt/2   # multiplies f[j]
  w1[j] =  dt/2   # multiplies f[j+1]
Returns a CumTrapWeights holding two length-(n-1) vectors.
"""
function precompute_weights(t::Vector{<:Real})
    n = length(t)
    @assert n ≥ 2 "need at least two points"
    dt = diff(t)
    half = 0.5 .* dt
    return CumTrapWeights(half, half)
end

"""
    cumtrap(weights::CumTrapWeights{T}, f::Vector{T}) where T

Apply the precomputed trapezoidal segment weights to your data `f`
to get the cumulative integral vector `F` of length `n`:

  F[1] = 0
  for j=1:n-1
    F[j+1] = F[j] + w0[j]*f[j] + w1[j]*f[j+1]
  end

so that F[k] ≈ ∫ₜ₁^{t[k]} f(t) dt in O(n) time.
"""
function cumtrap(weights::CumTrapWeights{T}, f::Vector{T}) where T
    n = length(f)
    @assert length(weights.w0) == n-1 "grid and data length mismatch"
    F = zeros(T, n)
    @inbounds for j in 1:n-1
        F[j+1] = F[j] + weights.w0[j]*f[j] + weights.w1[j]*f[j+1]
    end
    return F
end

# ——— Usage example ———

t = collect(LinRange(0, 10, 10_000))    # any irregular or regular grid
weights = precompute_weights(t)          # do this once

f1 = cos.(t)
F1 = cumtrap(weights, f1)                # fast cumulative ∫₀ᵗ cos → sin
sin.(t) .≈ F1
f2 = exp.(-0.1 .* t)
F2 = cumtrap(weights, f2)                # reuse the same weights on new data


using BenchmarkTools
using LinearAlgebra
N = 5_000_000; w = rand(N); h = rand(N);

@btime dot(w,h)

@btime sum(w .* h)