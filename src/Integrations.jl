integration_methods = Dict(:quadgk => QuadGKJL(), 
                            :trapezoidal => TrapezoidalRule(), 
                            :simpsons => SimpsonsRule())

function indefinite_integral(f::Function, lb; method = :quadgk, x_nodes::Union{AbstractVector{<:Real}, Nothing}=nothing)
    return x -> integrate(f, lb, x; method=method, x_nodes=x_nodes)
end

function integrate(f::Function, lb, ub; 
                    method = :quadgk, 
                    x_nodes::Union{AbstractVector{<:Real}, Nothing}=nothing)
    # Check if f is a univariate function
    integration_method = integration_methods[method]
    if isnothing(integration_method)
        error("Method $method is not supported. Available methods: $(keys(integration_methods))")
    end 

    # Check if f is a univariate function
    try
        f(lb + 1e-12)
    catch
        error("f must be a univariate function (accepts one argument)")
    end
    if method == :quadgk
        integrand = (x, p) -> f(x)
        problem = IntegralProblem(integrand, (lb, ub))
    elseif method in [:trapezoidal, :simpsons]
        if isnothing(x_nodes)
            error("x_nodes must be provided for quadrature methods")
        end
        y_nodes = f.(x_nodes)
        problem = SampledIntegralProblem(y_nodes, x_nodes)
    else
        error("Method $method is not implemented")
    end

    integral = solve(problem, integration_method)
    result = integral.u
    return result
end

