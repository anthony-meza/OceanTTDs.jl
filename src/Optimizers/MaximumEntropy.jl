export max_ent_inversion, gen_max_ent_inversion
#############################
# Inversion: free (Γ, Δ)
#############################

function max_ent_inversion(observations::Union{G, AbstractVector{G}};
                       C0 = nothing,               
                    prior_distribution::Vector,
                   support::Vector) where {G<:TracerObservation}
    observations_vec = observations isa AbstractVector ? observations : [observations]
    T = tracer_eltype(observations_vec[1])
    τs = support 
    nτ = length(τs)
    # preflight
    any(obs -> obs.f_src === nothing, observations_vec) &&
        throw(ArgumentError("All observations_vec must have f_src defined for inversion."))

    # create estimates from observations_vec
    estimates = [TracerEstimate(obs) for obs in observations_vec]

    # Handle C0: nothing -> zeros, scalar -> fill, vector -> use as-is
    if C0 === nothing
        C0s = zeros(T, length(observations_vec))
    elseif C0 isa AbstractVector
        C0s = C0
    else
        C0s = fill(T(C0), length(observations_vec))
    end

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
    observations_vec = observations isa AbstractVector ? observations : [observations]
    T = tracer_eltype(observations_vec[1])
    τs = support 
    nτ = length(τs)
    # preflight
    any(obs -> obs.f_src === nothing, observations_vec) &&
        throw(ArgumentError("All observations_vec must have f_src defined for inversion."))

    # create estimates from observations_vec
    estimates = [TracerEstimate(obs) for obs in observations_vec]

    # Handle C0: nothing -> zeros, scalar -> fill, vector -> use as-is
    if C0 === nothing
        C0s = zeros(T, length(observations_vec))
    elseif C0 isa AbstractVector
        C0s = C0
    else
        C0s = fill(T(C0), length(observations_vec))
    end

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