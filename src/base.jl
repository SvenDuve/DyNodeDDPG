
@with_kw mutable struct Parameter
    environment::String = "MountainCarContinuous-v0"
    state_size::Int = 2
    action_size::Int = 1
    action_bound::Float64 = 1.0
    batch_size::Int = 128
    mem_size::Int = 1000000
    frames::Int = 0
    max_episodes::Int = 2000
    max_episodes_length::Int = 1000
    critic_hidden::Array = [(200, 200)]
    actor_hidden::Array = [(200, 200)]
    γ::Float64 = 0.99
    τ::Float64 = 0.001
    η_actor::Float64 = 0.0001
    η_critic::Float64 = 0.001
end




function resetParameters(p)
    newP = Parameter(p; state_size=env.observation_space.shape[1],
        action_size=env.action_space.shape[1],
        action_bound=env.action_space.high[1])
    return newP
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


function 𝒩(ou::OrnsteinUhlenbeck)
    dx = ou.θ .* (ou.μ .- ou.X)
    dx = dx .+ ou.σ .* randn(Float32, length(ou.X))
    ou.X = ou.X .+ dx
end

global ou = OrnsteinUhlenbeck(0.0f0, 0.15f0, 0.2f0, [0.0f0])

mutable struct Episode
    env::PyObject
    π::AgentPolicy
    p::Parameter
    total_reward::Float64 # total reward of the episode
    last_reward::Float64
    niter::Int     # current step in this episode
    freq::Int       # number of steps between choosing actions
    maxn::Int       # max steps in an episode - should be constant during an episode
    episode::Array

    function Episode(env::PyObject, π::AgentPolicy, p::Parameter)

        total_reward, last_reward = 0.0, 0.0
        niter = 1
        freq = 1
        maxn = 1000
        episode = []
        new(env, π, p, total_reward, last_reward, niter, freq, maxn, episode)
    end
end





function (e::Episode)()

    s::Vector{Float32} = e.env.reset()
    r::Float64 = 0.0
    a::Vector{Float64} = [0.0]
    t::Bool = false

    for i in 1:e.maxn
        a = action(e.π, s, e.p)
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






function action(π::AgentPolicy, s::Vector{Float32}, p::Parameter)
    vcat(clamp.(μϕ(s) .+ vcat([𝒩(ou) for i in 1:p.action_size]...) * π.train, -p.action_bound, p.action_bound)...)
end


function greetings()
    println("Heeeeelllooo RL")
end