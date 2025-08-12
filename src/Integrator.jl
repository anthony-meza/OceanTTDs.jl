# Integrator.jl
# TrapezoidalIntegrator: Precompute nodes and weights
struct TrapezoidalIntegrator{T}
    nodes::Vector{T}
    weights::Vector{T}
end

"""
    make_trapezoidal_integrator(a, b, N)
Create a TrapezoidalIntegrator on [a,b] with N subintervals.
"""
function make_trapezoidal_integrator(a::Float64, b::Float64, N::Integer)
    h = (b - a) / N
    nodes = range(a, stop=b, length=N+1)
    weights = fill(h, N+1)
    weights[1] /= 2
    weights[end] /= 2
    return TrapezoidalIntegrator{Float64}(collect(nodes), collect(weights))
end

"""
    integrate(f, integrator)
Compute ∫ f(x) dx using trapezoidal rule; specialized on closure type.
"""
@inline function integrate(f::F, integrator::TrapezoidalIntegrator{T}) where {F<:Function, T}
    nodes, weights = integrator.nodes, integrator.weights
    acc = zero(T)
    @inbounds @simd for i in eachindex(nodes)
        acc += f(nodes[i]) * weights[i]
    end
    return acc
end
