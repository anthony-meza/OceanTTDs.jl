using FastGaussQuadrature

export make_integrator, make_integration_panels

function make_integrator(type::Symbol, panels::Vector{<:Tuple{T,T,<:Integer}}) where T
    @assert type in [:gausslegendre, :uniform] "Piecewise panels are only supported for :gausslegendre and :uniform"

    if !isempty(panels)
        Ts = map(p -> promote_type(typeof(p[1]), typeof(p[2])), panels)
        Tprom = foldl(promote_type, Ts; init=T)
    else
        Tprom = T
    end

    canon = [(Tprom(a), Tprom(b), Int(N)) for (a,b,N) in panels]
    if type == :uniform
        return make_uniform_integrator(canon)
    elseif type == :gausslegendre
        return make_gausslegendre_integrator(canon)
    else 
        error("Unsupported integrator type: $type")
    end
end

function make_integrator(type::Symbol, N::Integer, a::Real, b::Real)
    @assert type in [:trapezoid, :gausslegendre] "Unknown integrator type"
    if type == :trapezoid
        return make_trapezoidal_integrator(a, b, N)
    elseif type == :gausslegendre
        return make_gausslegendre_integrator(N, a, b)
    elseif type == :uniform
        return make_uniform_integrator(a, b, N)
    else
        error("Unsupported integrator type: $type")
    end
end

# ──────────────── Integrator Types ────────────────

"""
Fixed-grid trapezoidal integrator on [τ₁, τ₂] with N subintervals.
"""
struct TrapezoidalIntegrator{T<:Real}
    nodes   :: Vector{T}
    weights :: Vector{T}
end

function make_trapezoidal_integrator(τ₁::T, τ₂::T, N::Integer) where T<:Real
    @assert N > 0 "N must be positive"
    h = (τ₂ - τ₁) / N
    nodes   = collect(range(τ₁, τ₂, length = N + 1))
    weights = fill(h, N + 1)
    weights[1]   *= 0.5
    weights[end] *= 0.5
    TrapezoidalIntegrator{T}(nodes, weights)
end

"""
Gaussian–Legendre integrator on [a, b] with N points.
"""
struct GaussLegendreIntegrator{T<:Real}
    nodes   :: Vector{T}
    weights :: Vector{T}
end

function make_gausslegendre_integrator(N::Integer, a::T, b::T) where T<:Real
    @assert N > 0 "N must be positive"
    x, w = gausslegendre(N)          # nodes/weights on [-1, 1]
    nodes   = 0.5 * (b - a) .* x .+ 0.5 * (b + a)
    weights = 0.5 * (b - a) .* w
    GaussLegendreIntegrator{T}(nodes, weights)
end

# --- Overload: piecewise Gauss–Legendre with per-panel N ---
function make_gausslegendre_integrator(
    panels::Vector{Tuple{T,T,Int}}
) where {T<:Real}
    @assert !isempty(panels) "panels must be non-empty"

    # cache canonical [-1,1] nodes/weights by N
    cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

    total = 0
    @inbounds for (_,_,N) in panels
        @assert N > 0 "N must be positive"
        total += N
    end

    nodes   = Vector{T}(undef, total)
    weights = Vector{T}(undef, total)

    k = 1
    @inbounds for (a,b,N) in panels
        xw = get!(cache, N) do
            gausslegendre(N)  # Float64 on [-1,1]
        end
        x, w = xw
        scale = (b - a) / 2
        shift = (a + b) / 2
        for i in 1:N
            nodes[k]   = T(scale * x[i] + shift)
            weights[k] = T(scale * w[i])
            k += 1
        end
    end

    GaussLegendreIntegrator{T}(nodes, weights)
end

struct UniformIntegrator{T<:Real}
    nodes   :: Vector{T}
    weights :: Vector{T}
end

function make_uniform_integrator(N::Integer, a::T, b::T) where T<:Real
    @assert N > 0 "N must be positive"
    x, w = gausslegendre(N)          # nodes/weights on [-1, 1]
    nodes   = 0.5 * (b - a) .* x .+ 0.5 * (b + a)
    weights = zero(w) .+ one(T)
    UniformIntegrator{T}(nodes, weights)
end

# --- Overload: piecewise uniform with per-panel N ---
function make_uniform_integrator(
    panels::Vector{Tuple{T,T,Int}}
) where {T<:Real}
    @assert !isempty(panels) "panels must be non-empty"

    # cache canonical [-1,1] nodes/weights by N
    cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

    total = 0
    @inbounds for (_,_,N) in panels
        @assert N > 0 "N must be positive"
        total += N
    end

    nodes   = Vector{T}(undef, total)
    weights = Vector{T}(undef, total)

    k = 1
    @inbounds for (a,b,N) in panels
        xw = get!(cache, N) do
            gausslegendre(N)  # Float64 on [-1,1]
        end
        x, w = xw
        scale = (b - a) / 2
        shift = (a + b) / 2
        for i in 1:N
            nodes[k]   = T(scale * x[i] + shift)
            weights[k] = one(T)
            k += 1
        end
    end

    UniformIntegrator{T}(nodes, weights)
end

"""
    make_integration_panels(a, b, internal_breaks, Ns) -> Vector{Tuple{T,T,Int}}

Create a vector of integration panels `(a, b, N)` for piecewise quadrature.

Arguments:
- `a`: start of the full integration domain
- `b`: end of the full integration domain  
- `internal_breaks`: vector of strictly increasing breakpoints inside (a,b)
- `Ns`: vector of integers, one per panel (length = number of intervals)

The function automatically constructs the full list of breakpoints
as `[a; internal_breaks; b]` and returns one `(aᵢ, bᵢ, Nᵢ)` tuple per panel.

# Example
```julia
# Integrate over [0, 1], refine between 0.29 and 0.37
a, b = 0.0, 1.0
internal_breaks = [0.29, 0.37]  
Ns = [16, 128, 16]
panels = make_integration_panels(a, b, internal_breaks, Ns)
# panels == [(0.0, 0.29, 16), (0.29, 0.37, 128), (0.37, 1.0, 16)]
```
"""
function make_integration_panels(a::T, b::T, internal_breaks::Vector{T}, Ns::AbstractVector{Int}) where {T<:Real}
    @assert a < b "Start point must be less than end point"
    @assert all(a .< internal_breaks .< b) "Internal breaks must be inside (a,b)"
    breaks = vcat(a, internal_breaks, b)
    @assert issorted(breaks) "Breakpoints must be strictly increasing"
    @assert length(Ns) == length(breaks) - 1 "Ns length must equal number of panels"
    [(breaks[i], breaks[i+1], Ns[i]) for i in 1:length(Ns)]
end