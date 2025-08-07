"""
    boundary_propagator_timeseries(Gp_arr, f_atm, t_vec)
    
Calculate convolutions for matrix of propagators with vector of sources.

Returns 3D array with dimensions [propagator_row, source_column, time].
"""
function boundary_propagator_timeseries(
    Gp_arr::AbstractMatrix{<:Function}, 
    f_atm::AbstractVector{Union{<:Function, <:AbstractInterpolation}}, 
    t_vec::AbstractVector{<:Real}; 
    C0::Union{Nothing, AbstractVector{<:Real}}, 
    t0::Union{Nothing, <:Real}=nothing,
    integration_method = :exponential,
)
    # Get dimensions
    N, M = size(Gp_arr)
    Nt = length(t_vec)
    
    # Check dimensions match
    if M != length(f_atm)
        error("Dimension mismatch: Gp_arr has $M columns but f_atm has $(length(f_atm)) elements")
    end
    
    # Check dimensions match
    if (~isnothing(C0)) 
        if (N != length(C0))
            error("Dimension mismatch: Gp_arr has $N rows but C0 has $(length(C0)) elements")
        end
    end
    
        
    # Initialize result array
    results = zeros(N, M, Nt)
    
    # Calculate convolutions
    for i in 1:N, j in 1:M
        results[i, j, :] .= boundary_propagator_timeseries(Gp_arr[i, j], 
                                                           f_atm[j], 
                                                           t_vec; C0 = C0, t0 = t0, 
                                                           integration_method = integration_method)
    end

    return results
end

"""
    boundary_propagator_timeseries(Gp, f_atm, t_vec)

Calculate convolution of a boundary propagator 
with surface source using numerical integration.

"""
function boundary_propagator_timeseries(
    Gp::Function, 
    f_atm, 
    t_vec;
    C0=nothing,
    t0=nothing, 
    integration_method = :exponential,
)
    ftype = eltype(f_atm(t_vec[1]))
    result = zeros(ftype, length(t_vec))
    initial_value = C0 !== nothing ? C0 : zero(ftype)

    try 
        f_atm(-Inf)
    catch
        error("f_atm must be a univariate function or interpolation (accepts one argument)
        that can handle negative Infinity")
    end
    
    # for (i, t) in enumerate(t_vec)
    #     τ₀ = 0 
    #     integrand_f(τ) = Gp(τ) * f_atm(t - τ)
    #     integrated_f = indefinite_integral(integrand_f, τ₀; method = integration_method, x_nodes=t_vec)
    #     result[i] = initial_value + integrated_f(Inf)
    # end

    for (i, t) in enumerate(t_vec)
        if t == 0
            result[i] = initial_value
        else
            # Simpson's rule for convolution integral, a trick that assumes that 
            # f_atm==0 at t < 0
            integrand_f = tp -> Gp(t - tp) * f_atm(tp)

            #could also be this but havent tested it 
            # integrand_f = τ -> Gp(τ) * f_atm(t - τ)

            result[i] = initial_value + integrate(integrand_f, 0, t; method = integration_method)
        end
    end
    return result
end

"""
    boundary_propagator_timeseries(bp::BoundaryPropagator)
    
Calculate convolution using data stored in a BoundaryPropagator object.
"""
function boundary_propagator_timeseries(bp::BoundaryPropagator; integration_method = :exponential)
    return boundary_propagator_timeseries(bp.Gp_arr, bp.f_atm, 
                                          bp.t_vec; C0=bp.C0, t0=bp.t0, 
                                          integration_method = integration_method)
end