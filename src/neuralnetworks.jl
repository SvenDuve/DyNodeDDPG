# function setNetwork(nn::Critic, p::Parameter)
function setNetwork(nn::Critic)

    m = Chain(Dense(p.state_size + p.action_size, p.critic_hidden[1][1], elu),
        Chain([Dense(el[1], el[2], elu) for el in p.critic_hidden]...),
        Dense(p.critic_hidden[end][2], 1))

    return m

end

# issue here with the network creation, review loop

function setNetwork(nn::Actor)

    m = Chain(Dense(p.state_size, p.actor_hidden[1][1], elu),
                    Chain([Dense(el[1], el[2], elu) for el in p.actor_hidden]...), 
                        Dense(p.actor_hidden[end][2], p.action_size, tanh),
                        x -> x * p.action_bound)

    return m

end