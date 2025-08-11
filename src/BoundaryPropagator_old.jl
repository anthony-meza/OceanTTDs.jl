include("Integrator.jl")

"""
    BoundaryPropagator

Modeling boundary propagation with propagator functions and atmospheric source terms.

# Fields
- `Gp_arr`: Matrix of boundary propagator functions
- `f_atm`: Vector of atmospheric source functions
- `t_vec`: Vector of time points
- `C0`: Initial tracer concentration vector (optional)
- `t0`: Reference time (optional)
"""
struct BoundaryPropagator
    Gp_arr::AbstractMatrix{<:Function}
    f_atm::AbstractVector{Union{Function, AbstractInterpolation}}
    t_vec::AbstractVector{<:Real}
    C0::Union{Nothing, AbstractVector{<:Real}}
    t0::Union{Nothing, <:Real}
    τ_max::Float64
    nτ::Int64
    integrator::TrapezoidalIntegrator

    """
        BoundaryPropagator(Gp_input, f_atm_input, t_vec; C0=nothing, t0=nothing)

    Construct a BoundaryPropagator with flexible input handling.
    """
    function BoundaryPropagator(
        Gp_input::Union{Function, AbstractVector{<:Function}, AbstractMatrix{<:Function}}, 
        f_atm_input::Union{Function, AbstractInterpolation, AbstractVector{Union{Function, AbstractInterpolation}}}, 
        t_vec::AbstractVector{<:Real};
        C0::Union{Nothing, AbstractVector{<:Real}, <:Real}=nothing,
        t0::Union{Nothing, <:Real}=nothing,
        τ_max::Real=abs(t_vec[end] - t_vec[1]),  # Default to max time range
        nτ::Integer=100, 
        integrator::TrapezoidalIntegrator=make_trapezoidal_integrator(0.0, τ_max, nτ)
    )
        # Convert single function to 1×1 matrix
        if isa(Gp_input, Function)
            Gp_arr = fill(Gp_input, 1, 1)
        # Convert vector to n×1 matrix
        elseif isa(Gp_input, AbstractVector)
            Gp_arr = reshape(Gp_input, length(Gp_input), 1)
        else
            Gp_arr = Gp_input
        end
        
        # Convert single atmospheric function to 1-element vector
        if isa(f_atm_input, Union{Function, AbstractInterpolation})
            f_atm = [f_atm_input]
        else
            f_atm = f_atm_input
        end
        
        # Check that dimensions match
        n_rows, n_cols = size(Gp_arr)
        if n_cols != length(f_atm)
            error("Dimension mismatch: Gp_arr has $(n_cols) columns but f_atm has $(length(f_atm)) elements")
        end
        
        # Handle C0 conversion and validation
        C0_vec = nothing
        if C0 !== nothing
            if isa(C0, Number)
                C0_vec = fill(C0, n_rows)
            else
                C0_vec = C0
                if length(C0_vec) != n_rows
                    error("Dimension mismatch: C0 has $(length(C0_vec)) elements but Gp_arr has $(n_rows) rows")
                end
            end
        end
        
        return new(Gp_arr, f_atm, t_vec, C0_vec, t0, τ_max, nτ, integrator)
    end

    """
        BoundaryPropagator(Gp_arr, f_atm, t_vec; C0=nothing, t0=nothing, τ_max=..., nτ=..., integrator=nothing)

    Alternate constructor: directly set all fields, including a precomputed integrator.
    """
    function BoundaryPropagator(
        Gp_arr::AbstractMatrix{<:Function},
        f_atm::AbstractVector{Union{Function, AbstractInterpolation}},
        t_vec::AbstractVector{<:Real};
        C0::Union{Nothing, AbstractVector{<:Real}}=nothing,
        t0::Union{Nothing, <:Real}=nothing,
        τ_max::Real=abs(t_vec[end] - t_vec[1]),
        nτ::Integer=100,
        integrator::Union{Nothing, TrapezoidalIntegrator}=nothing
    )
        # Check that dimensions match
        n_rows, n_cols = size(Gp_arr)
        if n_cols != length(f_atm)
            error("Dimension mismatch: Gp_arr has $(n_cols) columns but f_atm has $(length(f_atm)) elements")
        end

        C0_vec = nothing
        if C0 !== nothing
            if isa(C0, Number)
                C0_vec = fill(C0, n_rows)
            else
                C0_vec = C0
                if length(C0_vec) != n_rows
                    error("Dimension mismatch: C0 has $(length(C0_vec)) elements but Gp_arr has $(n_rows) rows")
                end
            end
        end

        integrator_val = isnothing(integrator) ? make_trapezoidal_integrator(0.0, τ_max, nτ) : integrator
        return new(Gp_arr, f_atm, t_vec, C0_vec, t0, τ_max, nτ, integrator_val)
    end
end


"""
    boundary_propagator_timeseries(bp::BoundaryPropagator)
    
Calculate convolution using data stored in a BoundaryPropagator object.
"""
function boundary_propagator_timeseries(bp::BoundaryPropagator;)
    return boundary_propagator_timeseries(bp.Gp_arr, bp.f_atm, 
                                          bp.t_vec, bp.integrator; 
                                          C0=bp.C0, t0=bp.t0)
end

"""
    boundary_propagator_timeseries(Gp_arr, f_atm, t_vec)
    
Calculate convolutions for matrix of propagators with vector of sources.

Returns 3D array with dimensions [propagator_row, source_column, time].
"""
function boundary_propagator_timeseries(
    Gp_arr::AbstractMatrix{<:Function}, 
    f_atm::AbstractVector{Union{Function, AbstractInterpolation}}, 
    t_vec::AbstractVector{<:Real}, 
    integrator::TrapezoidalIntegrator; 
    C0::Union{Nothing, AbstractVector{<:Real}}=nothing, 
    t0::Union{Nothing, <:Real}=nothing, 
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
                                                           t_vec, integrator; C0 = C0, t0 = t0)
    end

    return results
end



"""
    boundary_propagator_timeseries(Gp, f_atm, t_vec)
    
Calculate convolution of a boundary propagator with surface source using discrete Simpson's integration.
"""
function boundary_propagator_timeseries(
    Gp::Function, 
    f_atm, 
    t_vec, 
    integrator::TrapezoidalIntegrator;
    C0=nothing,
    t0=nothing, 
    )
    ftype = eltype(f_atm(t_vec[1]))
    result = zeros(ftype, length(t_vec))
    initial_value = C0 != nothing ? C0 : zero(ftype)
    initial_time = t0 != nothing ? t0 : zero(ftype)

    integrand_f(τ, t, f) = Gp(τ) * f(t - τ)

    # τmax = integrator.nodes[end]
    for (i, t) in enumerate(t_vec)
        if t == 0
            result[i] = initial_value
        else
            # Simpson's rule for convolution integral
            # integrand_f = tp -> Gp(t - tp) * f_atm(tp)
            # result[i] = initial_value + integrate(integrand_f, integrator)
            #this assumes t = 0 
            # nodes_subset = integrator.nodes[integrator.nodes .<= (t - initial_time)]
            # weights_subset = integrator.weights[integrator.nodes .<= (t - initial_time)]
            # integrator_subset = TrapezoidalIntegrator{Float64}(nodes_subset, weights_subset)
            # integrator_subset = make_trapezoidal_integrator(0.0, t - initial_time, 2500)

            result[i] = initial_value + integrate(τ -> integrand_f(τ, t, f_atm), integrator)
        end
    end
    return result
end

"""
    replace_integrator(bp::BoundaryPropagator, new_integrator::TrapezoidalIntegrator)

Return a new BoundaryPropagator with the integrator field replaced by `new_integrator`.
"""
function replace_integrator(bp::BoundaryPropagator, new_integrator::TrapezoidalIntegrator)
    BoundaryPropagator(
        bp.Gp_arr,
        bp.f_atm,
        bp.t_vec;
        C0=bp.C0,
        t0=bp.t0,
        τ_max=bp.τ_max,
        nτ=bp.nτ
    ) |> x -> (x = setfield!(x, :integrator, new_integrator); x)
end

"""
    replace_Gp_arr(bp::BoundaryPropagator, new_Gp_arr)

Return a new BoundaryPropagator with the Gp_arr field replaced by `new_Gp_arr`.
"""
function replace_Gp_arr(bp::BoundaryPropagator, new_Gp_arr)
    BoundaryPropagator(
        new_Gp_arr,
        bp.f_atm,
        bp.t_vec;
        C0=bp.C0,
        t0=bp.t0,
        τ_max=bp.τ_max,
        nτ=bp.nτ,
        integrator=bp.integrator
    )
end



