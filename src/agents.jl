
function trainAgent(agent::DDPGAgent, pms::Parameter)
    #@show pms
    println("Welcome to this function.")
    gym = pyimport("gym")
    global env = gym.make(pms.environment)

    global p = resetParameters(pms)

    # set buffer
    global 𝒟 = []

    # get critic

    global Qθ = setNetwork(Critic())
    global Qθ′ = deepcopy(Qθ)

    # get actor

    global μϕ = setNetwork(Actor())
    global μϕ′ = deepcopy(μϕ)

    # copy actor
    #@show Qθ, μϕ



    # set optimiszers

    global opt_critic = Optimise.Adam(p.η_critic)
    global opt_actor = Optimise.Adam(p.η_actor)

    # set training conditions

    # global ou = OrnsteinUhlenbeck(0.0f0, 0.15f0, 0.2f0, [0.0f0])

    scores = zeros(100)
    e = 1
    idx = 1

    while e <= p.max_episodes

        ep = Episode(env, agent, p)()


        for (s, a, r, s′, t) in ep.episode
            remember(p.mem_size, s, a, r, s′, t)
            p.frames += 1

            if length(𝒟) >= p.batch_size# && π.train
                train(agent)
            end

        end

        scores[idx] = ep.total_reward
        idx = idx % 100 + 1
        avg = mean(scores)
        println("Episode: $e | Score: $(ep.total_reward) | Avg score: $avg | Frames: $(p.frames)")
        e += 1

    end

    #@show ep.total_reward

    # loop while some conditions

    # interact with the environment by generating some Episode
    # store episode in buffer

    # create a minibatch from the buffer

    # get all parameters for the loss functions

    # Do gradient ascent/ descent on the loss functions

    # update target for critic

    # update target for actor

    return -1

end #trainAgent



function dyNode(m::DyNodeModel, pms::Parameter)

    # To Do's:
    # to set up dynode_batch_size -> 64 in the paper

    # interactions with the real World


    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)


    # set buffer
    global 𝒟 = []

    # global fθ = setNode(m, p)
    global fθ = setNetwork(m) # Code up a Network that will be solved with euler steps
    global Rϕ = setNetwork(Rewards())

    # global optR = Flux.setup(Optimise.Adam(), Rϕ)#` and pass this `opt` to `train!
    # global opt_model = Optimise.Adam(0.005)
    # global opt_reward = Optimise.Adam(0.005)

    for i in 1:p.Sequences
        ep = Episode(env, m, p)()
        for (s, a, r, s′, t) in ep.episode
            remember(p.mem_size, s, a, r, s′, t)
        end

        train(m)
        if i % 10 == 0
            println("Iteration $i")
        end
    end

    return fθ, Rϕ, 𝒟

end