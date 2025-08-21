@inline _is_finite_real(y) = (y isa Real) && isfinite(float(y))

function logsumexp_with_logw(a::AbstractVector, logw::AbstractVector)
    @assert length(a) == length(logw)
    m = -Inf
    @inbounds for i in eachindex(a)
        m = max(m, a[i] + logw[i])
    end
    if !isfinite(m)           # all terms -Inf (e.g., all weights zero)
        return m
    end
    s = zero(eltype(a))
    @inbounds for i in eachindex(a)
        s += exp((a[i] + logw[i]) - m)
    end
    return m + log(s)
end
