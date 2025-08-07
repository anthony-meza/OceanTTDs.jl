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
    f_atm::AbstractVector{Union{<:Function, <:AbstractInterpolation}}
    t_vec::AbstractVector{<:Real}
    C0::Union{Nothing, AbstractVector{<:Real}}
    t0::Union{Nothing, <:Real}
    
    """
        BoundaryPropagator(Gp_input, f_atm_input, t_vec; C0=nothing, t0=nothing)

    Construct a BoundaryPropagator with flexible input handling.
    """
    println("Hello from BoundaryPropagator constructor!")
    function BoundaryPropagator(
        Gp_input::Union{<:Function, <:AbstractVector{<:Function}, <:AbstractMatrix{<:Function}}, 
        f_atm_input::Union{<:Function, <:AbstractInterpolation, 
                          <:AbstractVector{Union{<:Function, <:AbstractInterpolation}}}, 
        t_vec::AbstractVector{<:Real};
        C0::Union{Nothing, <:AbstractVector{<:Real}, <:Real}=nothing,
        t0::Union{Nothing, <:Real}=nothing
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
        if isa(f_atm_input, Union{<:Function, <:AbstractInterpolation})
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
        
        return new(Gp_arr, f_atm, t_vec, C0_vec, t0)
    end
end