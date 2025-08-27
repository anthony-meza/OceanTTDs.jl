# Jacobian Diagnostics Script
# Run this to diagnose issues with your manual Jacobian implementation

using FiniteDiff, LinearAlgebra
using TCMGreensFunctions
function diagnose_jacobian(observations::Union{G, AbstractVector{G}},
                          prior_distribution::Vector,
                          support::Vector;
                          test_point=nothing,
                          C0=nothing,
                          rtol=1e-6,
                          atol=1e-8) where {G<:TracerObservation}
    
    observations_vec = observations isa AbstractVector ? observations : [observations]
    T = tracer_eltype(observations_vec[1])
    τs = support 
    
    # Handle C0
    if C0 === nothing
        C0s = zeros(T, length(observations_vec))
    elseif C0 isa AbstractVector
        C0s = C0
    else
        C0s = fill(T(C0), length(observations_vec))
    end
    
    # Setup
    generate_MaxEntTTD(λ::AbstractVector) = MaxEntTTD(λ, 
                                        observations_vec,
                                        prior_distribution,
                                        support;
                                        weights = nothing,
                                        cache_pdf = true,
                                        debug_checks = false)
    λ2i_map = LambdaIndexMap(observations_vec)
    i2λ_map = IndexLambdaMap(observations_vec)
    
    nobs = total_observations(observations_vec)
    test_λ = test_point === nothing ? randn(nobs) * 0.1 : test_point
    
    println("Testing at λ = ", test_λ[1:min(5,end)], length(test_λ) > 5 ? "..." : "")
    
    # Residual function
    function r!(R, λ, p)
        d = generate_MaxEntTTD(λ)
        @inbounds for k in eachindex(observations_vec)
            obs = observations_vec[k];
            ŷ = convolve(d.probs, obs.f_src, obs.t_obs; τ_support = τs, C0=C0s[k])
            obs_ranges = λ2i_map.ranges[k]
            R[obs_ranges] .= (obs.y_obs .- ŷ)
        end
    end
    
    # Your manual Jacobian (reconstructed from the problematic version)
    function manual_jac!(J, λ, p)
        m = nobs
        n = length(λ)
        
        d = generate_MaxEntTTD(λ)
        
        @inbounds for i in 1:m
            tri = i2λ_map.tracer_indices[i]
            obi = i2λ_map.obs_indices[i]
            
            ti = observations_vec[tri].t_obs[obi]
            fi = observations_vec[tri].f_src
            
            for k in 1:n
                trk = i2λ_map.tracer_indices[k]
                ork = i2λ_map.obs_indices[k]
                
                tk = observations_vec[trk].t_obs[ork]
                fk = observations_vec[trk].f_src  # Use correct source function
                
                lagged_f_ti = fi.(ti .- d.support)
                lagged_f_tk = fk.(tk .- d.support)  # Fixed: use fk not fi
                
                Ĉ_k = dot(lagged_f_tk, d.probs)
                A_i = dot(lagged_f_ti, d.probs)
                B_ik = dot(lagged_f_ti .* lagged_f_tk, d.probs)
                J[i,k] = -(Ĉ_k * A_i - B_ik)  # Note: negative for residual derivative
            end
        end
    end
    
    # Compute both Jacobians
    m, n = nobs, nobs
    manual_J = zeros(m, n)
    manual_jac!(manual_J, test_λ, nothing)
    
    # Finite difference Jacobian
    R_test = zeros(m)
    fd_J = FiniteDiff.finite_difference_jacobian(
        (R, λ) -> r!(R, λ, nothing), 
        R_test, test_λ
    )
    
    # Compute errors
    error_matrix = abs.(manual_J - fd_J)
    max_error = maximum(error_matrix)
    
    # Relative error (avoid division by zero)
    rel_error_matrix = error_matrix ./ (abs.(fd_J) .+ 1e-12)
    max_rel_error = maximum(rel_error_matrix)
    
    # Find problematic entries
    problematic = findall((error_matrix .> atol) .& (rel_error_matrix .> rtol))
    
    println("="^60)
    println("JACOBIAN DIAGNOSIS RESULTS")
    println("="^60)
    println("Matrix size: $(m) × $(n)")
    println("Maximum absolute error: $(max_error)")
    println("Maximum relative error: $(max_rel_error)")
    println("Number of problematic entries: $(length(problematic))")
    
    if !isempty(problematic)
        println("\nWorst 5 problematic entries:")
        worst_indices = problematic[sortperm([error_matrix[idx] for idx in problematic], rev=true)[1:min(5,end)]]
        for idx in worst_indices
            i, k = idx.I
            println("  [$i,$k]: Manual=$(manual_J[i,k]:.6f), FD=$(fd_J[i,k]:.6f), Error=$(error_matrix[i,k]:.6f)")
        end
    end
    
    # Check for common issues
    println("\nDIAGNOSTIC CHECKS:")
    println("- Any NaN in manual Jacobian: $(any(isnan, manual_J))")
    println("- Any Inf in manual Jacobian: $(any(isinf, manual_J))")
    println("- Manual Jacobian norm: $(norm(manual_J))")
    println("- FD Jacobian norm: $(norm(fd_J))")
    println("- Condition number (manual): $(cond(manual_J))")
    
    return (
        max_error = max_error,
        rel_error = max_rel_error,
        error_matrix = error_matrix,
        manual_jac = manual_J,
        fd_jac = fd_J,
        problematic_entries = problematic,
        test_point = test_λ
    )
end

# Example usage - uncomment and modify as needed:
# Load your data first, then run:
# result = diagnose_jacobian(tracer_obs, discrete_prior_variation(Γ̂_IG), τs)