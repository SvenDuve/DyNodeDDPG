# function setNetwork(nn::Critic, p::Parameter)
function setNetwork(nn::Critic)

    return Chain(Dense(p.state_size + p.action_size, p.critic_hidden[1][1], elu),
                Chain([Dense(el[1], el[2], elu) for el in p.critic_hidden]...),
                Dense(p.critic_hidden[end][2], 1))

end

# issue here with the network creation, review loop

function setNetwork(nn::Actor)

    return Chain(Dense(p.state_size, p.actor_hidden[1][1], elu),
                Chain([Dense(el[1], el[2], elu) for el in p.actor_hidden]...),
                Dense(p.actor_hidden[end][2], p.action_size, tanh),
                x -> x * p.action_bound)


end



function setNetwork(nn::Rewards)

    return Chain(Dense(p.state_size + p.action_size, p.reward_hidden[1][1], relu),
                Chain([Dense(el[1], el[2], relu) for el in p.reward_hidden]...),
                Dense(p.reward_hidden[end][2], 1, tanh))


end


function setNetwork(nn::DyNodeModel) # Generate a NN to be solved with Euler updates

    return Chain(Dense(p.state_size + p.action_size, 200, elu),
                Dense(200, 200, elu),
                Dense(200, 200, elu),
                Dense(200, 200, elu),
                Dense(200, 2))

end