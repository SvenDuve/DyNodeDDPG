
@with_kw mutable struct Parameter
    environment::String = "MountainCarContinuous-v0"
    state_size::Int = 2
    action_size::Int = 1
    action_bound::Float64 = 1.0
    action_bound_high::Array = [1.0]
    action_bound_low::Array = [-1.0]
    batch_size::Int = 128
    batch_length::Int = 40
    mem_size::Int = 1000000
    frames::Int = 0
    train_start::Int = 1000 
    max_episodes::Int = 2000
    max_episodes_length::Int = 1000
    critic_hidden::Array = [(200, 200)]
    actor_hidden::Array = [(200, 200)]
    reward_hidden::Array = [(200, 200)]
    dynode_hidden::Array = [(200, 200)]
    γ::Float64 = 0.99
    noise_type::String = "gaussian"
    τ_actor::Float64 = 0.1
    τ_critic::Float64 = 0.5
    η_actor::Float64 = 0.0001
    η_critic::Float64 = 0.01
    Sequences::Int = 10
    H::Int = 200
    m::Int = 1000
    dT::Float64 = 0.01
    model_loss::Array = []
    reward_loss::Array = []
    total_rewards::Array = []
end




function resetParameters(p)
    
    newP = Parameter(p; state_size=env.observation_space.shape[1],
        action_size=env.action_space.shape[1],
        action_bound=env.action_space.high[1],
        action_bound_high=env.action_space.high,
        action_bound_low=env.action_space.low)
    return newP
end


mutable struct DDPGAgent

    train::Bool

    function DDPGAgent(train=true)
        new(train)
    end
end

mutable struct DyNodeModel

    train::Bool

    function DyNodeModel(train=true)
        new(train)
    end

end

mutable struct NodeModel

    train::Bool

    function NodeModel(train=true)
        new(train)
    end

end


mutable struct AgentPolicy

    train::Bool

    function AgentPolicy(train=true)
        new(train)
    end
end


mutable struct Critic end

mutable struct Actor end

mutable struct Rewards end



mutable struct OrnsteinUhlenbeck
    μ
    θ
    σ
    X
end




mutable struct GaussianNoise
    μ
    σ
end






function 𝒩(ou::OrnsteinUhlenbeck)
    dx = ou.θ .* (ou.μ .- ou.X)
    dx = dx .+ ou.σ .* randn(Float32, length(ou.X))
    ou.X = ou.X .+ dx
end

function 𝒩(gn::GaussianNoise)
    rand(Normal(gn.μ, gn.σ))
end


function setNoise(p::Parameter) 
    if p.noise_type == "gaussian"
        global noise = GaussianNoise(0.0f0, 0.1f0)
    else
        global noise = OrnsteinUhlenbeck(0.0f0, 0.15f0, 0.5f0, [0.0f0])
    end
end


mutable struct Episode
    env::PyObject
    π
    p::Parameter
    total_reward::Float64 # total reward of the episode
    last_reward::Float64
    niter::Int     # current step in this episode
    freq::Int       # number of steps between choosing actions
    maxn::Int       # max steps in an episode - should be constant during an episode
    episode::Array

    function Episode(env::PyObject, π, p::Parameter)

        total_reward, last_reward = 0.0, 0.0
        niter = 1
        freq = 1
        maxn = p.max_episodes_length
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






function action(π::DDPGAgent, s::Vector{Float32}, p::Parameter)
    vcat(clamp.(μϕ(s) .+ vcat([𝒩(noise) for i in 1:p.action_size]...) * π.train, -p.action_bound, p.action_bound)...)
end


function action(π::DyNodeModel, s::Vector{Float32}, p::Parameter)

    # return [rand((el[1]:0.01:el[2])) |> Float32 for el in zip(p.action_bound_low, p.action_bound_high)]
    return env.action_space.sample()
end


function action(π::NodeModel, s::Vector{Float32}, p::Parameter)

    # return [rand((el[1]:0.01:el[2])) |> Float32 for el in zip(p.action_bound_low, p.action_bound_high)]
    return env.action_space.sample()
end




function greetings()
    println("Heeeeelllooo RL")
end