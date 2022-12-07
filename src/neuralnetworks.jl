function setNetwork(nn::Critic, p::Parameters)

    m = Chain(Dense(p.state_size + p.action_size, p.critic_hidden[1], elu),
                    Chain([Dense(el, el, elu) for el in p.critic_hidden[2:end]]...), 
                        Dense(p.critic_hidden[end], 1))

    return m

end


function setNetwork(nn::Actor, p::Parameters)

    m = Chain(Dense(p.state_size, p.actor_hidden[1], elu),
                    Chain([Dense(el, el, elu) for el in p.actor_hidden[2:end]]...), 
                        Dense(p.actor_hidden[end], p.action_size, tanh),
                        x -> x * p.action_bound)

    return m

end