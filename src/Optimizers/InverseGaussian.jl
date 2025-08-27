
export invert_inverse_gaussian, invert_inverse_gaussian_equalvars, tracer_eltype, total_observations
#############################
# Inversion: free (Γ, Δ)
#############################

@inline function _clamp_to_bounds(x, lo, hi)
    x < lo && return lo + eps(lo)
    isfinite(hi) && x > hi && return hi - eps(hi)
    return x
end

"""
    tracer_eltype(obs::TracerObservation)

Returns the element type of the tracer observation data.
"""
tracer_eltype(obs::TracerObservation) = eltype(obs.y_obs)

"""
    total_observations(obs::TracerObservation)
    total_observations(obs_vec::AbstractVector{<:TracerObservation})

Returns the total number of observations in a single TracerObservation or 
the sum of observations across a vector of TracerObservation objects.
"""
total_observations(obs::TracerObservation) = obs.nobs
total_observations(obs_vec::AbstractVector{<:TracerObservation}) = sum(obs -> obs.nobs, obs_vec)

"""
    invert_ig(observations; τ_max, integrator, C0=0,
              u0=[Γ0,Δ0], lower=[1.0,12.0], upper=[Inf,Inf],
              warmstart=:anneal, sa_iters=50,
              lbfgs_iters=200)

Fit a shared `TracerInverseGaussian(Γ,Δ)` across all `observations::Vector{TracerObservation}`.
Creates `TracerEstimate`s internally and fills `yhat`.

Returns: `optimizer_results, Γ̂, Δ̂, estimates::Vector{TracerEstimate}`.
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
    
    observations_vec = observations isa AbstractVector ? observations : [observations]
    T = tracer_eltype(observations_vec[1])

    # preflight
    any(obs -> obs.f_src === nothing, observations_vec) &&
        throw(ArgumentError("All observations_vec must have f_src defined for inversion."))

    # create estimates from observations_vec
    estimates = [TracerEstimate(obs) for obs in observations_vec]

    # broadcastables
    τs   = τ_max      isa AbstractVector ? τ_max      : fill(T(τ_max), length(observations_vec))
    ints = integrator isa AbstractVector ? integrator : fill(integrator, length(observations_vec))
    
    # Handle C0: nothing -> zeros, scalar -> fill, vector -> use as-is
    if C0 === nothing
        C0s = zeros(T, length(observations_vec))
    elseif C0 isa AbstractVector
        C0s = C0
    else
        C0s = fill(T(C0), length(observations_vec))
    end

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
    invert_ig_equalvars(observations; τ_max, integrator, C0=0,
                        u0=[Γ0], lower=[1.0], upper=[Inf],
                        warmstart=:anneal, sa_iters=50,
                        lbfgs_iters=200)

Fit constrained model with Δ ≡ Γ. Creates and fills `TracerEstimate`s.

Returns: `optimizer_results, Γ̂, estimates::Vector{TracerEstimate}`.
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

    observations_vec = observations isa AbstractVector ? observations : [observations]
    T = tracer_eltype(observations_vec[1])

    any(obs -> obs.f_src === nothing, observations_vec) &&
        throw(ArgumentError("All observations_vec must have f_src defined for inversion."))

    estimates = [TracerEstimate(obs) for obs in observations_vec]

    τs   = τ_max      isa AbstractVector ? τ_max      : fill(T(τ_max), length(observations_vec))
    ints = integrator isa AbstractVector ? integrator : fill(integrator, length(observations_vec))
    C0s  = C0         isa AbstractVector ? C0         : fill(T(C0), length(observations_vec))

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