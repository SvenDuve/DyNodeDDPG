

mutable struct Parameters
    state_size::            Int
    action_size::           Int
    action_bound::          Float64
    batch_size::            Int
    mem_size::              Int
    frames::                Int
    max_episodes::          Int
    max_episodes_length::   Int
    γ::                     Float64
    τ::                     Float64
    η_actor::               Float64
    η_critic::              Float64
    function Parameters()
        new()
    end
end


mutable struct AgentPolicy
    
    train::     Bool

    function AgentPolicy(train=true)
        new(train)
    end
end





function greetings()
    println("Heeeeelllooo RL")
end