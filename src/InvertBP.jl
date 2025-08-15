
# include("tracer_inverse_gaussian.jl")  # must define TracerInverseGaussian(Γ, Δ)

"""
Estimate (Γ, Δ) for an Inverse-Gaussian boundary propagator from observations.
Returns: (Γ_opt, Δ_opt, IG_Dist, IG_BP_Estimate)
"""
function IG_BP_Inversion(
    src_in::Union{S,Vector{S}},
    t_obs::AbstractVector{T},
    y_obs_in::Union{T,AbstractVector{T}},
    σ_obs_in::Union{T,AbstractVector{T}},
    τ_max::T;
    integr = nothing,            # optionally pass a prebuilt integrator
    nτ::Integer=10_000,
    rule::Symbol = :trapezoid,   # or :gausslegendre
    C0::T = zero(T),             # scalar baseline
    u0::AbstractVector{<:Real} = [1e3, 1e2],  # initial guess for (Γ, Δ)
    lower::AbstractVector{<:Real} = [1.0, 12.0],
    upper::AbstractVector{<:Real} = [Inf, Inf],
) where {S<:Union{Function,AbstractInterpolation}, T<:Real}

    # normalize inputs
    srcs   = isa(src_in, AbstractVector) ? src_in : [src_in]
    y_obs  = y_obs_in isa AbstractVector ? T.(y_obs_in) : fill(T(y_obs_in), length(t_obs))
    σ_vec  = σ_obs_in isa AbstractVector ? T.(σ_obs_in) : fill(T(σ_obs_in), length(t_obs))
    @assert length(t_obs) == length(y_obs) == length(σ_vec) "t_obs, y_obs, σ_obs must have same length"

    # build (or reuse) integrator once
    if isnothing(integr)
        println("No prebuilt integrator provided, building one...")
        if rule === :gausslegendre
            # Gauss-Legendre integration
            integr = make_gausslegendre_integrator(nτ, zero(T), τ_max)
        elseif rule === :trapezoid
            # Trapezoidal integration
            integr = make_trapezoidal_integrator(zero(T), τ_max, nτ)
        else
            error("Unknown rule: $rule")
        end
    end

    # Inverse-Gaussian kernel (expects TracerInverseGaussian(Γ, Δ) to exist)
    IG_Dist(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)
    IG_Dist(τ, λ::AbstractVector{<:Real}) = IG_Dist(τ, λ[1], λ[2])

    # vector forward model (all observations)
    function IG_BP_Estimate(λ, t_obs; τ_max::T, integr, C0::T)
        return convolve_at(τ -> IG_Dist(τ, λ), srcs, t_obs; τ_max=τ_max, integr=integr, C0=C0)
    end

    # objective: weighted least squares
    function residual(λ, t_obs; τ_max::T, integr, C0::T)
        ŷ = IG_BP_Estimate(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)
        return (y_obs .- ŷ) ./ σ_vec
    end

    # objective: weighted least squares
    function J(λ, t_obs; τ_max::T, integr, C0::T)
        ŷ = IG_BP_Estimate(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)
        res = residual(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)
        return dot(res,  res)
    end

    # Exponential mapping (ensure positivity)
    exp_map(z) = exp.(z)

    # Inverse mapping: log
    exp_map_inv(λ) = log.(max.(λ, 1e-12))   # avoid log(0)

    # Transform initial guess to exponential-space
    z0 = exp_map_inv(u0)

    # Transform cost function to exponential-space
    J_z(z) = J(exp_map(z), t_obs; τ_max=τ_max, integr=integr, C0=C0)

    results = optimize(J_z, z0, SimulatedAnnealing(), 
                Optim.Options(iterations=50))
    z_opt = Optim.minimizer(results)
    λ_exp_opt = exp_map.(z_opt)

    J_λ(λ) = J(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)

    if J_λ(λ_exp_opt) < J_λ(u0)
        println("SimulatedAnnealing + Exponential mapping improved initial guess.")
        # If the initial guess is already better, use it
        println("u0 updated from: ", u0, "to: ", λ_exp_opt)
        # u0 .= λ_exp_opt
        u0[1] = clamp(λ_exp_opt[1], lower[1]+1e-12, upper[1]-1e-12)
        u0[2] = clamp(λ_exp_opt[2], lower[2]+1e-12, upper[2]-1e-12)
    end
    println("u0 updated to: ", u0)

    results = optimize(J_λ, lower, upper, 
                      u0, Fminbox(LBFGS()), 
                   Optim.Options(iterations=50))
    Γ_opt, Δ_opt = minimizer(results)

    optimized_IG_BP_Estimate(λ, t) = IG_BP_Estimate(λ, t; τ_max=τ_max, integr=integr, C0=C0)
    return results, Γ_opt, Δ_opt, IG_Dist, optimized_IG_BP_Estimate
end


# include("tracer_inverse_gaussian.jl")  # must define TracerInverseGaussian(Γ, Δ)

"""
Estimate (Γ, Δ) for an Inverse-Gaussian boundary propagator from observations.
Returns: (Γ_opt, Δ_opt, IG_Dist, IG_BP_Estimate)
"""
function IG_BP_Inversion_EqualVars(
    src_in::Union{S,Vector{S}},
    t_obs::AbstractVector{T},
    y_obs_in::Union{T,AbstractVector{T}},
    σ_obs_in::Union{T,AbstractVector{T}},
    τ_max::T;
    integr = nothing,            # optionally pass a prebuilt integrator
    nτ::Integer=10_000,
    rule::Symbol = :gausslegendre,   # or :gausslegendre
    C0::T = zero(T),             # scalar baseline
    u0::AbstractVector{<:Real} = [1e3],  # initial guess for (Γ, Δ)
    lower::AbstractVector{<:Real} = [1.0],
    upper::AbstractVector{<:Real} = [Inf],
) where {S<:Union{Function,AbstractInterpolation}, T<:Real}

    # normalize inputs
    srcs   = isa(src_in, AbstractVector) ? src_in : [src_in]
    y_obs  = y_obs_in isa AbstractVector ? T.(y_obs_in) : fill(T(y_obs_in), length(t_obs))
    σ_vec  = σ_obs_in isa AbstractVector ? T.(σ_obs_in) : fill(T(σ_obs_in), length(t_obs))
    @assert length(t_obs) == length(y_obs) == length(σ_vec) "t_obs, y_obs, σ_obs must have same length"

    # build (or reuse) integrator once
    if isnothing(integr)
        println("No prebuilt integrator provided, building one...")
        if rule === :gausslegendre
            # Gauss-Legendre integration
            integr = make_gausslegendre_integrator(nτ, zero(T), τ_max)
        elseif rule === :trapezoid
            # Trapezoidal integration
            integr = make_trapezoidal_integrator(zero(T), τ_max, nτ)
        else
            error("Unknown rule: $rule")
        end
    end

    # Inverse-Gaussian kernel (expects TracerInverseGaussian(Γ, Δ) to exist)
    IG_Dist(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)
    IG_Dist(τ, λ::AbstractVector{<:Real}) = IG_Dist(τ, λ[1], λ[1])


    # vector forward model (all observations)
    function IG_BP_Estimate(λ, t_obs; τ_max::T, integr, C0::T)
        return convolve_at(τ -> IG_Dist(τ, λ), srcs, t_obs; τ_max=τ_max, integr=integr, C0=C0)
    end

    # objective: weighted least squares
    function residual(λ, t_obs; τ_max::T, integr, C0::T)
        ŷ = IG_BP_Estimate(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)
        return (y_obs .- ŷ) ./ σ_vec
    end

    # objective: weighted least squares
    function J(λ, t_obs; τ_max::T, integr, C0::T)
        ŷ = IG_BP_Estimate(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)
        res = (y_obs .- ŷ) ./ σ_vec
        return dot(res,  res)
    end

    # Exponential mapping (ensure positivity)
    exp_map(z) = exp.(z)

    # Inverse mapping: log
    exp_map_inv(λ) = log.(max.(λ, 1e-12))   # avoid log(0)

    # Transform initial guess to exponential-space
    z0 = exp_map_inv(u0)

    # Transform cost function to exponential-space
    J_z(z) = J(exp_map(z), t_obs; τ_max=τ_max, integr=integr, C0=C0)

    results = optimize(J_z, z0, SimulatedAnnealing(), 
                Optim.Options(iterations=50))
    z_opt = Optim.minimizer(results)
    λ_exp_opt = exp_map.(z_opt)

    J_λ(λ) = J(λ, t_obs; τ_max=τ_max, integr=integr, C0=C0)

    if J_λ(λ_exp_opt) < J_λ(u0)
        println("SimulatedAnnealing + Exponential mapping improved initial guess.")
        # If the initial guess is already better, use it
        println("u0 updated from: ", u0, "to: ", λ_exp_opt)
        # u0 .= λ_exp_opt
        u0[1] = clamp(λ_exp_opt[1], lower[1]+1e-12, upper[1]-1e-12)
    end
    println("u0 updated to: ", u0)

    results = optimize(J_λ, lower, upper, 
                      u0, Fminbox(LBFGS()), 
                   Optim.Options(iterations=50))

    Γ_opt = minimizer(results)
    Δ_opt = Γ_opt

    optimized_IG_BP_Estimate(λ, t) = IG_BP_Estimate(λ, t; τ_max=τ_max, integr=integr, C0=C0)
    return results, Γ_opt, Δ_opt, IG_Dist, optimized_IG_BP_Estimate
end