
export invert_inverse_gaussian, invert_inverse_gaussian_equalvars, total_observations
#############################
# Inversion: free (Γ, Δ)
#############################

@inline function _clamp_to_bounds(x, lo, hi)
    x < lo && return lo + eps(lo)
    isfinite(hi) && x > hi && return hi - eps(hi)
    return x
end


"""
    total_observations(obs::TracerObservation)
    total_observations(obs_vec::AbstractVector{<:TracerObservation})

Returns the total number of observations in a single TracerObservation or 
the sum of observations across a vector of TracerObservation objects.
"""
total_observations(obs::TracerObservation) = obs.nobs
total_observations(obs_vec::AbstractVector{<:TracerObservation}) = sum(obs -> obs.nobs, obs_vec)

"""
    invert_inverse_gaussian(observations; τ_max, integrator, C0=nothing, 
                           u0=[1e3,1e2], lower=[1.0,12.0], upper=[Inf,Inf],
                           warmstart=:anneal, sa_iters=50, lbfgs_iters=200)

Fit an Inverse Gaussian Transit Time Distribution (TTD) to tracer observations.

## Mathematical Formulation

Estimates parameters Γ (mean) and Δ (width) of the Inverse Gaussian distribution:

```
f(τ; Γ, Δ) = √(Δ/(2πτ³)) exp(-(Δ(τ-Γ)²)/(2Γ²τ))
```

The optimization solves:
```
min_{Γ,Δ} Σᵢⱼ wᵢⱼ(yᵢⱼ - ŷᵢⱼ(Γ,Δ))²
```

where `ŷᵢⱼ = ∫₀^τ_max f(τ)·gᵢ(tⱼ-τ)dτ + C0ᵢ` is the convolved model prediction.

## Arguments
- `observations`: TracerObservation(s) containing times, data, uncertainties, source functions
- `τ_max`: Maximum transit time for integration (scalar or vector per observation)
- `integrator`: Numerical integrator for convolution (scalar or vector per observation)
- `C0`: Additive constant offset (nothing→0, scalar, or vector per observation)
- `u0`: Initial parameter guess [Γ₀, Δ₀]
- `lower`, `upper`: Parameter bounds
- `warmstart`: Use simulated annealing initialization (`:anneal` or `:none`)
- `sa_iters`, `lbfgs_iters`: Iteration limits for SA and L-BFGS phases

## Algorithm
1. **Warm Start** (if `:anneal`): Log-parameter simulated annealing to improve initial guess
2. **Main Optimization**: Bounded L-BFGS with box constraints via Fminbox
3. **Model Evaluation**: Final parameter estimates used to compute fitted values

## Returns
InversionResult containing fitted parameters [Γ̂, Δ̂], observation estimates, and optimizer output.

## Example
```julia
result = invert_inverse_gaussian(obs; τ_max=250_000.0, integrator=quad_integrator)
Γ_fit, Δ_fit = result.parameters
```
"""
function invert_inverse_gaussian(observations::Union{G, AbstractVector{G}};
                   τ_max,
                   integrator,
                   C0 = nothing,
                   u0 = [1e3, 1e2],
                   lower = [1.0, 12.0],
                   upper = [Inf, Inf],
                   warmstart::Symbol = :anneal,
                   sa_iters::Int = 50,
                   lbfgs_iters::Int = 200) where {G<:TracerObservation}
    
    observations_vec, estimates = prepare_observations(observations)
    T = tracer_eltype(observations_vec[1])
    
    # broadcastables
    τs   = τ_max      isa AbstractVector ? τ_max      : fill(T(τ_max), length(observations_vec))
    ints = integrator isa AbstractVector ? integrator : fill(integrator, length(observations_vec))
    C0s  = handle_C0_parameter(C0, observations_vec)

    # objective: weighted SSE, compute yhats inline
    function J(λ)
        Γ = T(λ[1]); Δ = T(λ[2])
        d = InverseGaussian(TracerInverseGaussian(Γ, Δ))
        s = zero(T)
        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k]
            ŷ = convolve(d, obs.f_src, obs.t_obs; τ_max=τs[k], integrator=ints[k], C0=C0s[k])
            if obs.σ_obs === nothing
                @inbounds for i in eachindex(obs.y_obs)
                    r = obs.y_obs[i] - ŷ[i]
                    s = muladd(r, r, s)
                end
            else
                σ = obs.σ_obs
                @inbounds for i in eachindex(obs.y_obs)
                    r = (obs.y_obs[i] - ŷ[i]) / σ[i]
                    s = muladd(r, r, s)
                end
            end
        end
        return s
    end

    # warm start (log-parameter anneal)
    if warmstart === :anneal
        exp_map(z)     = exp.(z)
        exp_map_inv(λ) = log.(max.(λ, 1e-12))
        z0 = exp_map_inv(u0)
        sa = Optim.optimize(z -> J(exp_map(z)), z0, SimulatedAnnealing(),
                            Optim.Options(iterations=sa_iters))
        λ_sa = exp_map(Optim.minimizer(sa))
        if J(λ_sa) < J(u0)
            u0 = [ _clamp_to_bounds(λ_sa[1], lower[1], upper[1]),
                   _clamp_to_bounds(λ_sa[2], lower[2], upper[2]) ]
        end
    end

    optimizer_results = Optim.optimize(J, lower, upper, u0, Fminbox(LBFGS()),
                             Optim.Options(iterations=lbfgs_iters))
    Γ̂, Δ̂ = Optim.minimizer(optimizer_results)

    # final yhat write-back
    d_template(Γ̂, Δ̂) = InverseGaussian(TracerInverseGaussian(T(Γ̂), T(Δ̂)))
    dhat = d_template(Γ̂, Δ̂)
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k]
        yhat_k = convolve(dhat, obs.f_src, obs.t_obs; τ_max=τs[k], integrator=ints[k], C0=C0s[k])
        update_estimate!(estimates[k], yhat_k)
    end

    return InversionResult(
        parameters = [Γ̂, Δ̂],
        obs_estimates = estimates,
        distribution = d_template,
        integrator = integrator,
        method = :inverse_gaussian,
        optimizer_output = optimizer_results
    )
end

#############################
# Inversion: constrained Δ ≡ Γ
#############################
"""
    invert_inverse_gaussian_equalvars(observations; τ_max, integrator, C0=nothing,
                                     u0=[1e3], lower=[1.0], upper=[Inf],
                                     warmstart=:anneal, sa_iters=50, lbfgs_iters=200)

Fit constrained Inverse Gaussian TTD with equal mean and width (Δ = Γ).

## Mathematical Formulation

Estimates a single parameter Γ for the constrained Inverse Gaussian distribution:

```
f(τ; Γ) = √(Γ/(2πτ³)) exp(-(Γ(τ-Γ)²)/(2Γ²τ)) = √(Γ/(2πτ³)) exp(-((τ-Γ)²)/(2Γτ))
```

This constraint reduces the parameter space from {Γ, Δ} to {Γ} where Δ = Γ, often 
providing more stable fits when data is limited or when the equal-variance assumption
is physically motivated.

## Algorithm
Identical to `invert_inverse_gaussian` but with the constraint Δ ≡ Γ enforced throughout
the optimization. Uses the same warm-start and L-BFGS approach.

## Returns
InversionResult containing fitted parameter [Γ̂], observation estimates, and optimizer output.

## Example
```julia
result = invert_inverse_gaussian_equalvars(obs; τ_max=250_000.0, integrator=quad_integrator)
Γ_fit = result.parameters[1]  # Γ̂ = Δ̂ 
```
"""
function invert_inverse_gaussian_equalvars(observations::Union{G, AbstractVector{G}};
                             τ_max,
                             integrator,
                             C0 = zero(τ_max),
                             u0 = [1e3],
                             lower = [1.0],
                             upper = [Inf],
                             warmstart::Symbol = :anneal,
                             sa_iters::Int = 50,
                             lbfgs_iters::Int = 200) where {G<:TracerObservation}

    observations_vec, estimates = prepare_observations(observations)
    T = tracer_eltype(observations_vec[1])
    
    τs   = τ_max      isa AbstractVector ? τ_max      : fill(T(τ_max), length(observations_vec))
    ints = integrator isa AbstractVector ? integrator : fill(integrator, length(observations_vec))
    C0s  = handle_C0_parameter(C0, observations_vec)

    function J(λ)
        Γ = T(λ[1])
        d = InverseGaussian(TracerInverseGaussian(Γ, Γ))
        s = zero(T)
        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k]
            ŷ = convolve(d, obs.f_src, obs.t_obs; τ_max=τs[k], integrator=ints[k], C0=C0s[k])
            if obs.σ_obs === nothing
                @inbounds for i in eachindex(obs.y_obs)
                    r = obs.y_obs[i] - ŷ[i]
                    s = muladd(r, r, s)
                end
            else
                σ = obs.σ_obs
                @inbounds for i in eachindex(obs.y_obs)
                    r = (obs.y_obs[i] - ŷ[i]) / σ[i]
                    s = muladd(r, r, s)
                end
            end
        end
        return s
    end

    if warmstart === :anneal
        exp_map(z)     = exp.(z)
        exp_map_inv(λ) = log.(max.(λ, 1e-12))
        z0 = exp_map_inv(u0)
        sa = Optim.optimize(z -> J(exp_map(z)), z0, SimulatedAnnealing(),
                            Optim.Options(iterations=sa_iters))
        λ_sa = exp_map(Optim.minimizer(sa))
        if J(λ_sa) < J(u0)
            u0 = [ _clamp_to_bounds(λ_sa[1], lower[1], upper[1]) ]
        end
    end

    optimizer_results = Optim.optimize(J, lower, upper, u0, Fminbox(LBFGS()),
                             Optim.Options(iterations=lbfgs_iters))
    Γ̂ = Optim.minimizer(optimizer_results)[1]

    d_template(Γ̂) = InverseGaussian(TracerInverseGaussian(T(Γ̂), T(Γ̂)))
    dhat = d_template(Γ̂)

    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k]
        yhat_k = convolve(dhat, obs.f_src, obs.t_obs; τ_max=τs[k], integrator=ints[k], C0=C0s[k])
        update_estimate!(estimates[k], yhat_k)
    end

    return InversionResult(
        parameters = [Γ̂],
        obs_estimates = estimates,
        distribution = d_template,
        integrator = integrator,
        method = :inverse_gaussian_equalvars,
        optimizer_output = optimizer_results
    )
end

# gaussian_ttd = InverseGaussian(TracerInverseGaussian(25, 25))
# unit_source(x) = (x >= 0) ? 1.0 : 0.0
# times = collect(1.:5:250.)
# τm = 250_000.

# break_points = [
# 1.2 * (times[end] - times[1]),   # first break after twice the data span
# 25_000.0               # second break at fixed value, ~ 1000 years
# ]

# nodes_points = [
# 10.0  * break_points[1],                # dense in first panel
# 0.03 * (break_points[2] - break_points[1]), # moderate in middle
# 0.01 * (τm - break_points[2])               # lighter in last
# ]

# nodes_points = Int.(round.(nodes_points))
# panels = make_integration_panels(0., τm, break_points, nodes_points)
# gauss_integrator = make_integrator(:gausslegendre, panels)

# ttd_results = convolve(gaussian_ttd, unit_source, times; τ_max=τm, integrator = gauss_integrator)

# ttd_observations = TracerObservation(times, ttd_results; 
#                                     σ_obs = 0.1 .+ zero(ttd_results), 
#                                     f_src = unit_source)

# optimizer_results, Γ̂, Δ̂, estimates = invert_ig(ttd_observations; τ_max = τm, integrator = gauss_integrator)

# optimizer_results2, Γ̂2, estimates2 = invert_ig_equalvars(ttd_observations; τ_max = τm, integrator = gauss_integrator)

# using BenchmarkTools
# @btime invert_ig_equalvars(ttd_observations; τ_max = τm, integrator = gauss_integrator);
# @btime invert_ig_equalvars([ttd_observations]; τ_max = τm, integrator = gauss_integrator);

# @btime invert_ig([ttd_observations]; τ_max = τm, integrator = gauss_integrator);
# @btime invert_ig(ttd_observations; τ_max = τm, integrator = gauss_integrator);