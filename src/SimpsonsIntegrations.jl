using Integrals

"""
    simpsons_integration(u, y; method=SimpsonEven())

Composite Simpson’s rule on the finite grid `u` with sampled values `y`.
"""
function simpsons_integration(u::AbstractVector, y::AbstractVector; )
    method=SimpsonsRule()
    problem = SampledIntegralProblem(y, u)
    sol     = solve(problem, method)
    return sol.u   # the integral approximation
end

"""
    simpsons_integration(f, a, b; method=SimpsonEven(), N=1_000_001)

Approximate ∫ₐᵇ f(u) du, allowing `a` or `b` to be ±Inf by the standard 1- or 2-sided transforms.
Automatically switches to SimpsonEven for any infinite endpoint.
"""
function simpsons_integration(f::Function, a::Real, b::Real;
                             N::Int = 1_000_001)
    if isfinite(a) && isfinite(b)
        # plain finite [a,b]
        pts  = LinRange(a, b, N)
        vals = f.(pts)

    elseif isfinite(a) && b === Inf
        # ∫ₐ^∞  u = a + t/(1-t),  t ∈ [0,1)
        dt   = 1/N
        t    = LinRange(0.0, 1.0-dt, N)
        u    = a .+ t./(1 .- t)
        jac  = 1.0 ./(1 .- t).^2
        pts, vals = t, f.(u).*jac

    elseif a === -Inf && isfinite(b)
        # ∫_{-∞}^b  u = b - t/(1-t),  t ∈ [0,1)
        dt   = 1/N
        t    = LinRange(0.0, 1.0-dt, N)
        u    = b .- t./(1 .- t)
        jac  = 1.0 ./(1 .- t).^2
        pts, vals = t, f.(u).*jac

    else
        # ∫_{-∞}^∞  u = t/(1-t^2),  t ∈ (-1,1)
        dt   = 2/N
        t    = LinRange(-1+dt, 1-dt, N)
        u    = t./(1 .- t.^2)
        jac  = (1 .+ t.^2)./(1 .- t.^2).^2
        pts, vals = t, f.(u).*jac
    end

    return simpsons_integration(pts, vals;)
end


