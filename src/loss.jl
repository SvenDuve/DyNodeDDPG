function lossCritic(Y::Matrix{Float64}, S::Matrix{Float32}, A::Matrix{Float64})
    V = Qθ(vcat(S, A))
    (Y .- V) .^ 2 |> sum
end


function maxActor(S::Matrix{Float32})
    Qθ(vcat(S, μϕ(S))) |> sum |> mean
end

# function lossModel(m::DyNodeModel, s, a, r, s′)
#     @show size(r)
#     ŝ = []
#     state = s[:, 1]
#     for (i, action) in enumerate(a)
#         state = fθ(vcat(vcat(state...), action))
#         append!(ŝ, [state])
#     end
#     # @show vcat(hcat(ŝ...))
#     model_loss = Flux.mse(vcat(hcat(ŝ...)), s′)

#     return model_loss, ŝ
# end


# function lossReward(m::DyNodeModel, s, a, r, s′)
#     r̂ = []
#     for (i, action) in enumerate(a)
#         reward = Rϕ(vcat(vcat(s′[:, i]...), action))
#         append!(r̂, reward)
#     end

#     reward_loss = Flux.mse(reshape(r̂, (1, 5)), r)
#     return reward_loss, r̂
# end
function lossReward(m::DyNodeModel, s, a, r, s′)
    r̂ = [Rϕ(vcat(vcat(s′[:, k]...), action)) for (k, action) in enumerate(a)]
    reward_loss = Flux.mse(hcat(r̂...), r)
    return reward_loss
end


# function transition(m::DyNodeModel, s, a, r, s′)
#     # ŝ = Buffer(s, size(s))
#     ŝ = []
#     # @show ŝ
#     #    buf = Buffer(xs, length(xs), 5)
#     # @show typeof(ŝ)
#     # @show typeof(ŝ[1])
#     state = s[:, 1]
#     for (i, action) in enumerate(a)
#         state = fθ(vcat(vcat(state...), action))
#         # @show i
#         # @show ŝ[i]
#         # @show typeof(state)
#         # @show vec(reduce(hcat, state))
#     end
#     return ŝ
# end