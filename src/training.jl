
function trainAgent(agent::DDPGAgent, p::Parameters)

    println("Welcome to this function.")

    # get critic

    Qθ = setNetwork(Critic(), p)

    # copy critic

    Qθ′ = deepcopy(Qθ)

    # get actor

    μϕ = setNetwork(Actor(), p)

    # copy actor

    μϕ′ = deepcopy(μϕ)

    @show Qθ, μϕ

    # set optimiszers

    opt_critic = Optimise.Adam(p.η_critic)
    opt_actor = Optimise.Adam(p.η_actor)

    # set training conditions

    ep = Episode(env, AgentPolicy())

    @show ep.total_reward

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