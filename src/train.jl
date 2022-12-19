
function train(agent::DDPGAgent)

    S, A, R, S′ = sampleBuffer(DDPGAgent())
    # @show size(S), size(A), size(R), size(S′)

    A′ = μϕ′(S′)
    V′ = Qθ′(vcat(S′, A′))
    Y = R + p.γ * V′

    # critic
    θ = params(Qθ)
    dθ = gradient(() -> lossCritic(Y, S, A), θ)
    update!(Optimise.Adam(p.η_critic), params(Qθ), dθ)

    # actor
    ϕ = params(μϕ)
    dϕ = gradient(() -> -maxActor(S), ϕ)
    update!(Optimise.Adam(p.η_actor), params(μϕ), dϕ)


    for (base, target) in zip(params(Qθ), params(Qθ′))
        target .= p.τ * base .+ (1 - p.τ) * target
    end

    for (base, target) in zip(params(μϕ), params(μϕ′))
        target .= p.τ * base .+ (1 - p.τ) * target
    end

end


function train(m::DyNodeModel)

    S, A, R, S′ = sampleBuffer(m)

    model_loss = []
    reward_loss = []

    for i in 1:p.batch_size
    
        dθ = gradient(() -> dyNodeLoss(m, S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(fθ))
        update!(Optimise.Adam(0.005), params(fθ), dθ)


        dϕ = gradient(() -> rewardLoss(m, S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(Rϕ))
        update!(Optimise.Adam(0.005), params(Rϕ), dϕ)

        append!(model_loss, dyNodeLoss(m, S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
        append!(reward_loss, rewardLoss(m, S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
    
    end

    @show mean(model_loss)
    @show mean(reward_loss)

    return model_loss, reward_loss

end



