"""
Maximum Entropy Method (MEM) for Transit Time Distribution inversion.

The Maximum Entropy Method finds the TTD that maximizes entropy while fitting 
observations, providing the "least biased" solution consistent with the data.
This avoids over-interpretation when data constraints are limited.
"""

export max_ent_inversion, gen_max_ent_inversion
"""
    max_ent_inversion(observations; C0=nothing, prior_distribution, support)

Maximum Entropy inversion for discrete Transit Time Distributions.

## Mathematical Formulation

The Maximum Entropy Method solves the constrained optimization problem:

```
max S[p] = -Σᵢ pᵢ ln(pᵢ/mᵢ)    (maximize entropy)
subject to: Σᵢ pᵢ = 1           (normalization)
           ||Gp - d||² = χ²     (fit observations)
```

where:
- `p = [p₁, p₂, ..., pₙ]` is the discrete TTD probability vector
- `m = [m₁, m₂, ..., mₙ]` is the prior distribution  
- `G` is the forward operator (convolution matrix)
- `d` is the observation vector
- `χ²` is the target misfit level

This is solved via Lagrange multipliers, yielding:

```
pᵢ = mᵢ exp(-Σⱼ λⱼ Gⱼᵢ) / Z(λ)
```

where λ = [λ₁, λ₂, ..., λₘ] are Lagrange multipliers found by solving the nonlinear system:

```
∂S/∂λⱼ = Σᵢ Gⱼᵢ pᵢ(λ) - dⱼ = 0    for j = 1, ..., m
```

## Algorithm
1. **Setup**: Initialize λ = 0 (uniform distribution)
2. **Levenberg-Marquardt**: First-pass nonlinear solver with analytic Jacobian
3. **Trust Region**: Second-pass refinement for improved convergence
4. **Distribution**: Compute final p(λ) and fitted observations

The analytic Jacobian is:
```
Jⱼₖ = ∂rⱼ/∂λₖ = Ĉₖ Aⱼ - Bⱼₖ
```

where Ĉₖ, Aⱼ, Bⱼₖ involve convolution integrals with the current distribution p(λ).

## Arguments  
- `observations`: TracerObservation(s) with times, data, source functions
- `prior_distribution`: Prior probability vector m on support points
- `support`: Discrete transit time support points τ = [τ₁, τ₂, ..., τₙ]
- `C0`: Additive offset parameter (nothing→0, scalar, or vector)

## Returns
InversionResult with λ parameters, discrete TTD probabilities, and solver output.

## Example
```julia
τ_support = collect(0.0:100.0:10000.0)
m_prior = ones(length(τ_support)) / length(τ_support)  # uniform prior
result = max_ent_inversion(obs; prior_distribution=m_prior, support=τ_support)
ttd_probs = result.distribution
```
"""
function max_ent_inversion(observations::Union{G, AbstractVector{G}};
                       C0 = nothing,               
                    prior_distribution::Vector,
                   support::Vector) where {G<:TracerObservation}
    observations_vec, estimates = prepare_observations(observations)
    T = tracer_eltype(observations_vec[1])
    τs = support 
    C0s = handle_C0_parameter(C0, observations_vec)

    generate_MaxEntTTD(λ::AbstractVector) = MaxEntTTD(λ, 
                                        observations_vec,
                                        prior_distribution,
                                        support;
                                        weights = nothing,
                                        cache_pdf = true,
                                        debug_checks = false)
    λ2i_map = LambdaIndexMap(observations_vec)
    i2λ_map = IndexLambdaMap(observations_vec)

    function r!(R, λ, p)
        d = generate_MaxEntTTD(λ)
        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k];
            ŷ = convolve(d.probs, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])

            obs_ranges = λ2i_map.ranges[k]
            R[obs_ranges] .= (ŷ .- obs.y_obs)
        end
    end


    function jac_r!(J, λ, p)
        m = total_observations(observations_vec)   # number of residuals (observations)
        n = length(λ)       # number of parameters
        
        # Check Jacobian matrix dimensions
        size(J) == (m, n) || throw(DimensionMismatch("Jacobian matrix size $(size(J)) != ($m, $n)"))

        # Max-Ent distribution evaluated on quadrature nodes
        d = generate_MaxEntTTD(λ)

        # J[i, k] = ∑_j f(t_i - τ_j) * d(τ_j) * ( Ĉ_k - f(t_k - τ_j) )
        @inbounds for i in 1:m               # rows: observations
            tri = i2λ_map.tracer_indices[i]
            obi = i2λ_map.obs_indices[i]

            ti = observations_vec[tri].t_obs[obi]
            fi = observations_vec[tri].f_src

            for k in 1:n                     # cols: surface sources
                trk = i2λ_map.tracer_indices[k]
                ork = i2λ_map.obs_indices[k]

                tk =  observations_vec[trk].t_obs[ork]
                fk = observations_vec[trk].f_src

                lagged_f_ti = fi.(ti .- d.support)
                lagged_f_tk = fk.(tk .- d.support)

                Ĉ_k = dot(lagged_f_tk, d.probs) 
                
                A_i = dot(lagged_f_ti, d.probs)
                B_ik = dot(lagged_f_ti .* lagged_f_tk, d.probs)
                J[i,k] = Ĉ_k * A_i - B_ik

            end
        end

    end

    nobs = total_observations(observations_vec)
	initial_λ = zeros(nobs)

    # Set up nonlinear problem
    fn = NonlinearFunction(r!, jac = jac_r!)  # Can use autodiff for Jacobian
    initial_prob = NonlinearProblem(fn, initial_λ, nothing)
    
    # First optimization pass: Levenberg-Marquardt
    lm_λ_result = solve(initial_prob, LevenbergMarquardt(); 
                      maxiters=250, store_trace=Val(true))
    
    # Second optimization pass: Trust Region with refined initial guess
    refined_prob = NonlinearProblem(fn, lm_λ_result[:], nothing)
    tr_λ_result = solve(refined_prob, TrustRegion(); 
                        maxiters=250, store_trace=Val(true))

    # final yhat write-back
    dhat = generate_MaxEntTTD(tr_λ_result[:])
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k]
        yhat_k = convolve(dhat.probs, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
        update_estimate!(estimates[k], yhat_k)
    end

    return InversionResult(
        parameters = tr_λ_result[:], 
        obs_estimates = estimates,
        distribution = dhat.probs,
        integrator = nothing,
        support = support, 
        method = :max_ent,
        optimizer_output = [lm_λ_result, tr_λ_result]
    )

end

#Generalized Max Ent Inversion 
function gen_max_ent_inversion(observations::Union{G, AbstractVector{G}};
                       C0 = nothing,               
                    prior_distribution::Vector,
                   support::Vector, error_support::Vector, error_prior::Vector) where {G<:TracerObservation}
    observations_vec, estimates = prepare_observations(observations)
    T = tracer_eltype(observations_vec[1])
    τs = support 
    C0s = handle_C0_parameter(C0, observations_vec)

    generate_MaxEntTTD(λ::AbstractVector) = MaxEntTTD(λ, 
                                        observations_vec,
                                        prior_distribution,
                                        support;
                                        weights = nothing,
                                        cache_pdf = true,
                                        debug_checks = false)

    generate_MaxEntTTDErr(λ) = MaxEntTTDErr(λ, 
                                        error_prior,
                                        error_support;
                                        weights = nothing,
                                        cache_pdf = true,
                                        debug_checks = false)

    λ2i_map = LambdaIndexMap(observations_vec)
    function r!(R, λ, p)
        d = generate_MaxEntTTD(λ)
        d_err = [generate_MaxEntTTDErr(λi).expected_error for λi in λ]

        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k];
            ŷ = convolve(d.probs, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
            obs_ranges = λ2i_map.ranges[k]
            err_hat = d_err[obs_ranges]
            R[obs_ranges] .= ((ŷ .+ err_hat) .- obs.y_obs)
        end
    end

    nobs = total_observations(observations_vec)
	initial_λ = zeros(nobs)

    # Set up nonlinear problem
    fn = NonlinearFunction(r!)  # Can use autodiff for Jacobian
    initial_prob = NonlinearProblem(fn, initial_λ, nothing)
    
    # First optimization pass: Levenberg-Marquardt
    lm_λ_result = solve(initial_prob, LevenbergMarquardt(); 
                      maxiters=250, store_trace=Val(true))
    
    # Second optimization pass: Trust Region with refined initial guess
    refined_prob = NonlinearProblem(fn, lm_λ_result[:], nothing)
    tr_λ_result = solve(refined_prob, TrustRegion(); 
                        maxiters=250, store_trace=Val(true))

    # final yhat write-back
    dhat = generate_MaxEntTTD(tr_λ_result[:])
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k]
        yhat_k = convolve(dhat.probs, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
        update_estimate!(estimates[k], yhat_k)
    end

    return InversionResult(
        parameters = tr_λ_result[:], 
        obs_estimates = estimates,
        distribution = dhat.probs,
        integrator = nothing,
        support = support, 
        method = :max_ent_w_err,
        optimizer_output = [lm_λ_result, tr_λ_result]
    )
end