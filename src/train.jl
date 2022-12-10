
function train(agent::DDPGAgent)

    S, A, R, S′ = sampleBuffer(DDPGAgent())
    # @show size(S), size(A), size(R), size(S′)

    A′ = μϕ′(S′)
    V′ = Qθ′(vcat(S′, A′))
    Y = R + p.γ * V′


    θ = params(Qθ)
    dθ = gradient(() -> lossCritic(Y, S, A), θ)
    update!(opt_critic, params(Qθ), dθ)

    ϕ = params(μϕ)
    dϕ = gradient(() -> -maxActor(S), ϕ)
    update!(opt_actor, params(μϕ), dϕ)

    for (base, target) in zip(params(Qθ), params(Qθ′))
        target .= p.τ * base .+ (1 - p.τ) * target
    end

    for (base, target) in zip(params(μϕ), params(μϕ′))
        target .= p.τ * base .+ (1 - p.τ) * target
    end

end


function train(m::DyNodeModel)

    S, A, R, S′ = sampleBuffer(m)
    Ŝ = []
    R̂ = []
    model_loss = 0
    reward_loss = 0
    for i in 1:p.batch_size
        # ŝ = []
        # r̂ = []
        ŝ, r̂ = transition(S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i])
        # state = S[:,1,i]
        # ŝ = [fθ(vcat(vcat(state...), action)) for action in A[:,:,i]]
        # r̂ = [Rϕ(vcat(vcat(S′[:, k, i]...), action)) for (k, action) in enumerate(A[:,:,i])]
        # for action in a
        #     state = fθ(vcat(vcat(state...), action))
        #     append!(ŝ, [state])
        # end
        # @show size(R[:,:,i]), R[:,:,i] 

        θ = params(fθ)
        dθ = gradient(() -> Flux.mse(vcat(hcat(ŝ...)), S′[:,:,i]), θ)
        update!(opt_model, params(fθ), dθ)
    
        ϕ = params(Rϕ)
        dϕ = gradient(() -> Flux.mse(hcat(r̂...), R[:,:,i]), ϕ)
        update!(opt_reward, params(Rϕ), dϕ)
    
        model_loss = Flux.mse(vcat(hcat(ŝ...)), S′[:,:,i])
        reward_loss = Flux.mse(hcat(r̂...), R[:,:,i])

        append!(Ŝ, ŝ)
        append!(R̂, r̂)
        
    end
    
    
    # θ = params(fθ)
    # dθ = gradient(() -> Flux.mse(vcat(hcat(Ŝ...)), S′), θ)
    # update!(opt_model, params(fθ), dθ)

    # ϕ = params(Rϕ)
    # dϕ = gradient(() -> Flux.mse(hcat(R̂...), R), ϕ)
    # update!(opt_reward, params(Rϕ), dϕ)

    # model_loss = Flux.mse(vcat(hcat(Ŝ...)), S′)
    # reward_loss = Flux.mse(hcat(R̂...), R)

    @show model_loss
    @show reward_loss

    return -1
end



function transition(s, a, r, s′)

    ŝ = []
    r̂ = []
    state = s[:,1]
    for (i, action) in enumerate(a)
        state = fθ(vcat(vcat(state...), action))
        append!(ŝ, [state])
        reward = Rϕ(vcat(vcat(s′[:,i]...), action))
        append!(r̂, reward)
    end

    return (ŝ, r̂)
end


# callback() = begin
#     global iter += 1
#     if iter % 10 == 1

# end