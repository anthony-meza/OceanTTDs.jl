export invert_tcm
function invert_tcm(observations::Union{G, AbstractVector{G}};
                       C0 = nothing,               
                    prior_distribution::Vector,
                   PΣ ::Matrix,
                   support::Vector) where {G<:TracerObservation}
    
    observations_vec, estimates = prepare_observations(observations)
    T = tracer_eltype(observations_vec[1])
    τs = support 
    C0s = handle_C0_parameter(C0, observations_vec)

    function J(d::Vector{A}, d_prior::Vector{B}, 
                            PΣ::Matrix{B}, 
                            observations_vec::Vector{<:TracerObservation}) where {A, B}
        ss_vec = T[] #holds sum of squares
        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k]
            ŷ = convolve(d, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
            r = (obs.y_obs - ŷ) / obs.σ_obs
            ss_vec = vcat(r.^2)
        end
        #   println(ss_vec)
        dp = d - d_prior
        prior_cost = dp' * PΣ * dp  # This is fine for JuMP
        
        return prior_cost + sum(ss_vec)
    end
    J(x::Vector) = J(x, prior_distribution, PΣ, observations_vec)

    jmodel = JuMP.Model(Ipopt.Optimizer)
    set_time_limit_sec(jmodel, 30.0)
    set_optimizer_attribute(jmodel, "tol", 1e-8)
    set_optimizer_attribute(jmodel, "acceptable_tol", 1e-6)
    set_optimizer_attribute(jmodel, "mu_strategy", "adaptive")
    @variable(jmodel, 0.0 <= x[i = 1:nτ], start = 1/nτ)
    @constraint(jmodel, sum(x) == 1)

    @objective(jmodel, Min, J(x))
    @time JuMP.optimize!(jmodel)

    # final yhat write-back
    dhat = value(x)
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k]
        yhat_k = convolve(dhat, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
        update_estimate!(estimates[k], yhat_k)
    end

    return InversionResult(
        parameters = nothing, 
        obs_estimates = estimates,
        distribution = dhat,
        integrator = nothing,
        support = support, 
        method = :time_correction_method,
        optimizer_output = jmodel
    )
end