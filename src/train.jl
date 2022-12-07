
function train(agent::DDPGAgent)

    S, A, R, S′ = sampleBuffer()
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