function integrate_discrete_error(
    Gp_in::F,
    support = nothing,
    C0::T = zero(T)
) where {F<:Function, T<:Real}

    # acc = zero(T)
    integrand = Gp_in.(support)
    # acc = dot(integrand, support)
    acc  =  sum(integrand)
    return acc + C0
end

# === Unified constructor (entry point) ===
"""
    MaxEntDist(λ, t_obs, f_atm, μ, integrator; implementation=:direct)

Builds a callable MaxEnt distribution object.

- `implementation = :direct` → exp-space (stores `Z`)
- `implementation = :stable` → log-space (stores `log(Z)`) for numerical stability
"""
function DiscErrDist(λ::Vector{G},
                    t_obs::Vector{T},
                    ν,
                    support::Vector{T};
                    implementation::Symbol = :stable) where {T,G}
    #### SUPPORT CAN BE EITHER VECTOR OR MATRIX 
    #### 
    if implementation === :direct

        return nothing # not implemented yet

    elseif implementation === :stable
        #direct implementation of a stable sum, can be slow because creating many arrays
        Zlogs = zero(λ)
        for i in eachindex(t_obs, λ)
            IndefDEDnum(s) = DiscErrNumeratorStable(s, λ[i], ν)
            # println("λ[$i]: ", λ[i])
            a = IndefDEDnum.(support)
            a_max = maximum(a)
            a_stable_sum = sum(exp.(a .- a_max)) #sum_oro doesn't work with NLSolve
            Zlogs[i] = a_max + log(a_stable_sum)
        end

        return DiscErrDistStable{T,G,typeof(ν)}(λ, ν, Zlogs)

    else
        throw(ArgumentError("implementation must be :direct or :stable"))
    end
end

# === Stable (log-space) struct ===
struct DiscErrDistStable{T,G,F1}
    λ::Vector{G}
    ν::F1
    Zlogs::Vector{G}
end

function (d::DiscErrDistStable)(λi::G, support::T) where {T,G}
    # ws = zero(d.λ)
    # numerator = DiscErrNumeratorStable(support, λi, d.ν, d.t_obs)
    # @inbounds for i in eachindex(d.λ)
    #     ws[i] = exp(numerator[i] - d.Zlogs[i])
    # end
    if λi ∉ d.λ
        println("λi: ", λi)
        println("d.λ: ", d.λ)
        error("λi not in d.λ")
    end
    
    i = findfirst(isequal(λi), d.λ)

    numerator = DiscErrNumeratorStable(support, λi, d.ν)
    return exp(numerator - d.Zlogs[i])
end



function DiscErrNumeratorStable(support_i::T,
                                   λi::G,
                                   ν) where {T,G}
    #direct summation
    # s = zero(T)
    # @inbounds @simd for i in eachindex(t_obs)
    #     # s = muladd(λ[i], f_atm(t_obs[i] - τ), s)
    # end

    return log(ν(support_i)) + (- (λi * support_i))
end


# s = 0.
# z = [1e6, 1e-11, -1e6]
# for i in eachindex(z)
#     s += z[i]
# end
# s

# s = 0.
# z = [1e6, 1e-11, -1e6]
# for i in eachindex(z)
#     sa, se = two_sum(s, z[i])
#     s = (sa)
# end
# s
# sum_oro(zi for zi in z)
# # using BenchmarkTools
# # x = collect(1:1e4)
# # y = zeros(length(x))
# # @btime @. y = sin(x) + 3*x^2;
# # @btime y .= sin.(x) .+ 3 .* x.^2;



# function logsumexp_online(a)
#     m = -Inf; s = 0.0
#     @inbounds for x in a
#         if x <= m
#             s += exp(x - m)
#         else
#             s = s * exp(m - x) + 1
#             m = x
#         end
#     end
#     m + log(s)
# end

# a = randn(1000) .+ 20 .* rand()  # arbitrary
# m = maximum(a)
# two_pass = m + log(sum(exp.(a .- m)))
# isapprox(logsumexp_online(a), two_pass; rtol=1e-12)