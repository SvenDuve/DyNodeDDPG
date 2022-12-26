
function trainAgent(agent::DDPGAgent, pms::Parameter)
    #@show pms
    println("Welcome to this function.")
    gym = pyimport("gym")
    global env = gym.make(pms.environment)

    global p = resetParameters(pms)

    setNoise(p)
    # if p.noise == "gaussian"
    #     global noise = GaussianNoise(0.0f0, 0.1f0)
    # else
    #     global noise = OrnsteinUhlenbeck(0.0f0, 0.15f0, 0.5f0, [0.0f0])
    # end

    # set buffer
    global 𝒟 = []

    # get critic

    global Qθ = setNetwork(Critic())
    global Qθ′ = deepcopy(Qθ)

    # get actor

    global μϕ = setNetwork(Actor())
    global μϕ′ = deepcopy(μϕ)


    scores = zeros(100)
    e = 1
    idx = 1

    while e <= p.max_episodes

        ep = Episode(env, agent, p)()


        for (s, a, r, s′, t) in ep.episode
            remember(p.mem_size, s, a, r, s′, t)
            p.frames += 1

            # if length(𝒟) >= p.train_start# && π.train
            if p.frames >= p.train_start# && π.train
                train(agent)
            end
            
        end
        

        scores[idx] = ep.total_reward
        idx = idx % 100 + 1
        avg = mean(scores)
        if (e-1) % 10 == 0
            println("Episode: $e | Score: $(ep.total_reward) | Avg score: $avg | Frames: $(p.frames)")
        end
        e += 1

        append!(p.total_rewards, ep.total_reward)

    end

    return μϕ, p

end #trainAgent



function dyNode(m::DyNodeModel, pms::Parameter)

    # To Do's:
    # to set up dynode_batch_size -> 64 in the paper

    # interactions with the real World


    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)

    #setNoise(p)

    # set buffer
    global 𝒟 = []

    # global fθ = setNode(m, p)
    global fθ = setNetwork(m) # Code up a Network that will be solved with euler steps
    global Rϕ = setNetwork(Rewards())


    for i in 1:p.Sequences
        ep = Episode(env, m, p)()
        for (s, a, r, s′, t) in ep.episode
            remember(p.mem_size, s, a, r, s′, t)
        end

        model_loss, reward_loss = train(m)
        # alt_train(m)
        if i % 10 == 0
            println("Iteration $i")
        end
        append!(p.model_loss, model_loss)
        append!(p.reward_loss, reward_loss)
    end
    return p, fθ, Rϕ
end





function NODEAgent(m::NodeModel, pms::Parameter)

    # To Do's:
    # to set up dynode_batch_size -> 64 in the paper

    # interactions with the real World


    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)

    #setNoise(p)
    #@show noise

    # set buffer
    global 𝒟 = []

    global fθ = setNode(m, p)
    global Rϕ = setNetwork(Rewards())


    for i in 1:p.Sequences
        ep = Episode(env, m, p)()
        for (s, a, r, s′, t) in ep.episode
            remember(p.mem_size, s, a, r, s′, t)
        end

        model_loss, reward_loss = train(m)
        if i % 10 == 0
            println("Iteration $i")
        end
        append!(p.model_loss, model_loss)
        append!(p.reward_loss, reward_loss)
    end
    
    return p, fθ, Rϕ

end


