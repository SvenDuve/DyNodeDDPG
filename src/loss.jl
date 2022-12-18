function lossCritic(Y::Matrix{Float64}, S::Matrix{Float32}, A::Matrix{Float64})
    V = Qθ(vcat(S, A))
    (Y .- V) .^ 2 |> sum
end


function maxActor(S::Matrix{Float32})
    Qθ(vcat(S, μϕ(S))) |> sum |> mean
end



function lossReward(m::DyNodeModel, s, a, r, s′)
    r̂ = [Rϕ(vcat(vcat(s′[:, k]...), action)) for (k, action) in enumerate(a)]
    reward_loss = Flux.mse(hcat(r̂...), r)
    return reward_loss
end


function rewardLoss(m::DyNodeModel, S, A, R, S′)

    R̂ = Zygote.Buffer(R)

    for j in collect(1:size(A)[3])

        R̂[:,:,j] = transition(S[:,:,j], A[:,:,j], R[:,:,j])[2]

    end

    return Flux.mse(copy(R̂), R)

end






function dyNodeLoss(m::DyNodeModel, S, A, R, S′)

    Ŝ = Zygote.Buffer(S)

    for j in collect(1:size(A)[3])

        Ŝ[:,:,j] = transition(S[:,:,j], A[:,:,j], R[:,:,j])[1]

    #    Ŝ[:, :, i], R̂[:, :, i] = transition(S[:, :, i], A[:, :, i], R[:, :, i], S′[:, :, i])
    end

    return (1 / p.batch_size) * (1 / p.batch_length) * sum(abs.(copy(Ŝ) - S′))

end




function transition(s, a, r)

    ŝ = Zygote.Buffer(s)
    r̂ = Zygote.Buffer(r)
    state = s[:,1]
    for i in collect(1:length(a))
        state = euler(state, a[:,i])
        ŝ[:,i] = state
        r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
    end
    
    return copy(ŝ), copy(r̂)
end


function euler(s, a)
    x = vcat(s, a)
    return s + 0.001 * fθ(x)
end