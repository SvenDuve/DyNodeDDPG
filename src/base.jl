

mutable struct Parameters
    environment::String
    state_size::Int
    action_size::Int
    action_bound::Float64
    batch_size::Int
    mem_size::Int
    frames::Int
    max_episodes::Int
    max_episodes_length::Int
    critic_hidden::Array
    actor_hidden::Array
    γ::Float64
    τ::Float64
    η_actor::Float64
    η_critic::Float64
    function Parameters()
        new()
    end
end

mutable struct DDPGAgent end


mutable struct AgentPolicy

    train::Bool

    function AgentPolicy(train=true)
        new(train)
    end
end

mutable struct Critic end

mutable struct Actor end



mutable struct OrnsteinUhlenbeck
    μ
    θ
    σ
    X
end


mutable struct Episode
    env::PyObject
    π::AgentPolicy
    total_reward::Float64 # total reward of the episode
    last_reward::Float64
    niter::Int     # current step in this episode
    freq::Int       # number of steps between choosing actions
    maxn::Int       # max steps in an episode - should be constant during an episode
    episode::Array

    function Episode(env::PyObject, π::AgentPolicy)
        total_reward, last_reward = 0.0, 0.0
        niter = 1
        freq = 1
        maxn = 1000
        episode = []
        new(env, π, total_reward, last_reward, niter, freq, maxn, episode)
    end
end





function (e::Episode)()

    s::Vector{Float32} = e.env.reset()
    r::Float64 = 0.0
    a::Vector{Float64} = [0.0]
    t::Bool = false

    for i in 1:e.maxn
        a = action(e.π, s)
        s′, r, t, _ = e.env.step(a)
        #@show a, s′, r, t
        e.total_reward += r
        #@show e.total_reward
        append!(e.episode, [(s, a, r, s′, t)])
        s = s′
        if t
            s = env.reset()
        end
    end

    return e

end






function action(π::AgentPolicy, s::Vector{Float32})
    vcat(clamp.(μϕ(s) .+ vcat([𝒩(ou) for i in 1:p.action_size]...) * π.train, -p.action_bound, p.action_bound)...)
end


function greetings()
    println("Heeeeelllooo RL")
end