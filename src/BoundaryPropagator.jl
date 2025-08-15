include("Integrator.jl")
# ───────────── BoundaryPropagator ─────────────

"""
Holds propagators, sources, observation times, initial conditions, reference time,
and an arbitrary integrator providing `.nodes` and `.weights`.
"""
struct BoundaryPropagator{F<:Function,
                          S<:Union{Function,AbstractInterpolation},
                          I,
                          T<:Real}
    propagators :: Vector{F}   # length M
    sources     :: Vector{S}   # length J
    t           :: Vector{T}   # length K
    C0          :: Vector{T}   # length M
    t0          :: T
    integrator  :: I           # TrapezoidalIntegrator{T} or GaussLegendreIntegrator{T}
end

# ───────────── Constructors ─────────────

"""
Constructor reusing a precomputed integrator (no reallocation).
"""
function BoundaryPropagator(
    Gp_in::Union{F,Vector{F}},
    src_in::Union{S,Vector{S}},
    t::Vector{T},
    integrator::I;
    C0 = nothing,
    t0 = nothing
) where {F<:Function, S<:Union{Function,AbstractInterpolation}, I, T<:Real}
    Gp_vec  = isa(Gp_in, Function) ? [Gp_in] : Gp_in
    src_vec = isa(src_in, Function) ? [src_in] : src_in
    M = length(Gp_vec)
    C0_vec = C0 === nothing ? zeros(T, M) : (isa(C0, Number) ? fill(C0, M) : C0)
    @assert length(C0_vec) == M "C0 length $(length(C0_vec)) ≠ #propagators $M"
    t0_val = t0 === nothing ? first(t) : t0
    BoundaryPropagator{F,S,typeof(integrator),T}(Gp_vec, src_vec, t, C0_vec, t0_val, integrator)
end

"""
Default constructor: choose `rule = :trapezoid` or `:gausslegendre`.
- `τ_max` is the integration upper limit (lower limit is 0)
- For `:trapezoid`, set `nτ` (subintervals). For `:gausslegendre`, set `Ngl` (points).
"""
function BoundaryPropagator(
    Gp_in::Union{F,Vector{F}},
    src_in::Union{S,Vector{S}},
    t::Vector{T};
    C0    = nothing,
    t0    = nothing,
    τ_max::T = abs(last(t) - first(t)),
    integr::I = nothing,
    rule::Symbol = :trapezoid,
    nτ::Integer = 100,
    Ngl::Integer = 64
) where {F<:Function, S<:Union{Function,AbstractInterpolation}, I, T<:Real}
    
    if isnothing(integr)
        if rule === :trapezoid
            integr = make_trapezoidal_integrator(zero(T), τ_max, nτ)
        elseif rule === :gausslegendre
            integr = make_gausslegendre_integrator(Ngl, zero(T), τ_max)
        else
            error("Unknown rule: $(rule)")
        end
    end

    BoundaryPropagator(Gp_in, src_in, t, integr; C0=C0, t0=t0)
end

# ─────── Accessors ───────

"""Upper integration limit (last node)."""
τ_max(bp::BoundaryPropagator) = last(bp.integrator.nodes)

"""Number of panel steps (trapezoid) or nodes-1 (Gauss–Legendre)."""
nτ(bp::BoundaryPropagator) = length(bp.integrator.nodes) - 1


"""
Single-point convolution for propagator `p` at time `t_obs` (scalar).
"""
function convolve_at(bp::BoundaryPropagator{F,S,I,T}, p::Integer, t_obs::T) where {F,S,I,T}
    @assert 1 ≤ p ≤ length(bp.propagators)
    acc = zero(T)
    Gp = bp.propagators[p]
    @inbounds @simd for k in eachindex(bp.integrator.nodes)
        τk = bp.integrator.nodes[k]; wk = bp.integrator.weights[k]
        g  = Gp(τk)
        for src in bp.sources
            acc += g * src(t_obs - τk) * wk
        end
    end
    acc + bp.C0[p]
end

function convolve_at(bp::BoundaryPropagator{F,S,I,T}, p::Integer, t_obs::Vector{T}) where {F,S,I,T}
    conv(t::T) = convolve_at(bp, p, t)
    return conv.(t_obs)
end

function convolve_at(
    Gp_in::F,
    src_in::Union{S,Vector{S}},
    t_obs::Vector{T};
    τ_max::T,
    integr = nothing,
    rule::Symbol = :trapezoid,
    nτ::Integer = 100,
    Ngl::Integer = 64,
    C0::T = zero(T)
) where {F<:Function, S<:Union{Function,AbstractInterpolation}, T<:Real}
    conv(t) = convolve_at(Gp_in, src_in, t; τ_max=τ_max, integr=integr, rule=rule, nτ=nτ, Ngl=Ngl, C0=C0)
    return conv.(t_obs)
end

"""
Single-point convolution using raw inputs (no bp required).
Set either `integr` to a prebuilt integrator or choose a rule/parameters.
"""
function convolve_at(
    Gp_in::F,
    src_in::Union{S,Vector{S}},
    t_obs::T;
    τ_max::T,
    integr = nothing,
    rule::Symbol = :trapezoid,
    nτ::Integer = 100,
    Ngl::Integer = 64,
    C0::T = zero(T)
) where {F<:Function, S<:Union{Function,AbstractInterpolation}, T<:Real}

    srcs = isa(src_in, AbstractVector) ? src_in : [src_in]
    local_integr = isnothing(integr) ? (
        rule === :trapezoid    ? make_trapezoidal_integrator(zero(T), τ_max, nτ) :
        rule === :gausslegendre ? make_gausslegendre_integrator(Ngl, zero(T), τ_max) :
        error("Unknown rule: $(rule)")
    ) : integr

    acc = zero(T)
    @inbounds for k in eachindex(local_integr.nodes)
        τk = local_integr.nodes[k]; 
        wk = local_integr.weights[k]
        g  = Gp_in(τk)
        for src in srcs
            acc += g * src(t_obs - τk) * wk
        end
    end
    acc + C0
end

function convolve_at(
    Gp_in::Vector{G},
    src_in::Union{S,Vector{S}},
    t_obs::T;
    τ_max::T,
    integr = nothing,
    rule::Symbol = :trapezoid,
    nτ::Integer = 100,
    Ngl::Integer = 64,
    C0::T = zero(T)
) where {F<:Function, S<:Union{Function,AbstractInterpolation}, T<:Real, G}

    #check that Gp_in is normalized to sum to 1
    # @assert isapprox(sum(Gp_in), 1.0, atol=1e-5)

    srcs = isa(src_in, AbstractVector) ? src_in : [src_in]
    local_integr = isnothing(integr) ? (
        rule === :trapezoid    ? make_trapezoidal_integrator(zero(T), τ_max, nτ) :
        rule === :gausslegendre ? make_gausslegendre_integrator(Ngl, zero(T), τ_max) :
        error("Unknown rule: $(rule)")
    ) : integr

    acc = zero(T)
    @inbounds for k in eachindex(local_integr.nodes)
        τk = local_integr.nodes[k]; 
        wk = local_integr.weights[k]
        g  = Gp_in[k]
        for src in srcs
            acc += g * src(t_obs - τk) * wk
        end
    end
    acc + C0
end

σ   = 0.01
xc  = 0.33
f(x) = exp(-((x - xc)/σ)^2) + 0.1*sin(8x)

integ_ref = make_integrator(:gausslegendre, 4000, 0.0, 1.0)
integ_global = make_integrator(:gausslegendre, 64, 0.0, 1.0)

panels = [
    (0.0, 0.29, 16),
    (0.29, 0.37, 128),  # heavy refinement where f is sharply peaked
    (0.37, 1.0, 16)
]

# integ_panel = make_integrator(:gausslegendre, panels)

# I_ref = dot(f.(integ_ref.nodes), integ_ref.weights)
# I_global = dot(f.(integ_global.nodes), integ_global.weights)
# I_panel = dot(f.(integ_panel.nodes), integ_panel.weights); 

# I_ref ≈ I_panel
# I_ref ≈ I_global


###


# using QuadGK
# τm  = 50000.0
# n = 200000
# gauss_integrator = make_integrator(:gausslegendre, n, 0., τm)
# trapezoid_integrator = make_integrator(:trapezoid, n, 0., τm)

# z = [0.0]
# @btime dot(cos.(gauss_integrator.nodes), gauss_integrator.weights)
# @btime dot(cos.(trapezoid_integrator.nodes), trapezoid_integrator.weights) 

# @btime quadgk(τ -> cos(τ), 0, τm)
# sin(τm)


# α   = 0.3
# gp  = τ -> exp(-α * τ)
# src = t -> cos(4 - t)

# integrand(τ, t) = gp(τ) * src(t - τ)

# t   = collect(0.0:0.1:5.0)
# τm  = 50000.0
# nτ_ = 100000
# Ngl = 5000

# bp_trap = BoundaryPropagator(gp, src, t; τ_max=τm, rule=:trapezoid, nτ=nτ_)
# bp_gl   = BoundaryPropagator(gp, src, t; τ_max=τm, rule=:gausslegendre, Ngl=Ngl)

# tt = 2.5
# println("— Demo —")
# println("τ_max(trap) = ", τ_max(bp_trap), ", nτ(trap) = ", nτ(bp_trap))
# @btime convolve_at(bp_trap, 1, tt)
# @btime convolve_at(bp_gl, 1, tt)
# @btime quadgk(τ -> integrand(τ, tt), 0, τm)


# t_vec = collect(0.:1.:1000)
# @btime convolve_at(bp_trap, 1, t_vec)
# @btime convolve_at(bp_gl, 1, t_vec)
# quad_int(tt) = quadgk(τ -> integrand(τ, tt), 0, τm)[1]
# @btime quad_int.(t_vec)

# quad_int.(t_vec) .- convolve_at(bp_gl, 1, t_vec)