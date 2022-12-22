# General to do

# in train.jl:
# try to combine loss and parameters of the model and the rewards,
# this way we perhaps only take gradients once per iteration.

# in agent.jl:
# implement a a model based RL algo combining Dynode and DDPGAgent

# create diagnostic output in the form of:
# 1. accuracy, precision etc. got to find some that makes sense for 
# continuous action spaces
# 2. track environment steps as opposed to model steps




# using Flux
# import Flux.params
# using Conda
# using PyCall





# Conda.add("gym") 
# #Conda.add("NumPy")
# Conda.add("wheel")
# Conda.add("gym")
# Conda.add("box2d-py")
# Conda.add("pyglet")

# gym = pyimport("gym")
# #env = gym.make("Pendulum-v1")
# env = gym.make("BipedalWalker-v3")
# #env = gym.make("MountainCarContinuous-v0")
# #env = gym.make("Pendulum-v1")

# using Pkg
# Pkg.activate(".")
# using DyNodeDDPG


trainAgent(DDPGAgent(), Parameter(environment="Pendulum-v1",
    critic_hidden=[(64, 64)],
    actor_hidden=[(64, 64)],
    batch_size=64,
    max_episodes=200))

# trainAgent(DDPGAgent(), Parameter(environment="MountainCarContinuous-v0",
#     critic_hidden=[(64, 64)],
#     actor_hidden=[(64, 64)],
#     batch_size=64))

# trainAgent(DDPGAgent(), Parameter(environment="BipedalWalker-v3",
#     critic_hidden=[(200, 200)],
#     actor_hidden=[(200, 200)],
#     batch_size=128,
#     max_episodes=200))




using Pkg
Pkg.activate(".")
using DyNodeDDPG
using Flux
using Flux.Optimise
using Zygote
import Flux.params
using Statistics


actor, pmss = trainAgent(DDPGAgent(), Parameter(environment="MountainCarContinuous-v0",
    mem_size=100000,
    critic_hidden=[(64, 64), (64, 64)],
    actor_hidden=[(32, 32), (32, 32)],
    batch_size=64,
    max_episodes=200,
    η_actor = 0.001,
    η_critic = 0.001,
    τ_actor=0.1,
    τ_critic=0.025));

# best so far τ_actor = 0.2 and τ_critic = 0.05

plot(pmss.total_rewards)



pmss, dynamics, reward = dyNode(DyNodeModel(), 
                            Parameter(max_episodes_length=200, 
                                        batch_size=1,
                                        batch_length=40,
                                        Sequences=300,
                                        dT = 0.001, 
                                        reward_hidden=[(32, 32), (32, 32)],
                                        dynode_hidden=[(64, 64), (64, 64)]));



sum(pmss.model_loss)
sum(pmss.reward_loss)



plot(pmss.model_loss, c=:red) 
plot!(twinx(), pmss.reward_loss)

# dyNode(DyNodeModel(), Parameter(environment="BipedalWalker-v3", max_episodes_length=200))

using Plots
using Statistics
gr()

meanModelLoss = [mean(pmss.model_loss[i-9:i]) for i in collect(10:300)]
meanRewardLoss = [mean(pmss.reward_loss[i-9:i]) for i in collect(10:300)]


plot(meanModelLoss, c=:red)
plot!(twinx(), meanRewardLoss, c=:green)


using Conda
using PyCall

gym = pyimport("gym")
env = gym.make("MountainCarContinuous-v0")




K = 5

A = []

for k in 1:K  
    append!(A, [2 .* (rand(5) .- 0.5)])
end

R = []
r = []


@show Seq
s = env.reset()
@show s
for i in 1:5
    a = env.action_space.sample()
    append!(r, reward(vcat(s, a)) |> first)
    s = dynamics(vcat(s, a))
    @show(env.step(a))
    @show s
end
@show r
append!(R, sum(r))
r = []


act = A[argmax(R)][1]

s = env.step([act])


K = 5

A = []

for k in 1:K  
    append!(A, [1/10 .* (rand(5) .- 0.5)])
end

R = []
r = []


for Seq in A
    s = [-0.5782173, 0.0004026636]
    for a in Seq
        append!(r, reward(vcat(s, a)) |> first)
        s = dynamics(vcat(s, a))
    end
    append!(R, sum(r))
    r = []
end


act = A[argmax(R)][1]

s = env.step([act])



function eulers_method(f, α, a, b, N)

    n1 = N + 1
    u = zeros(n1, 2)

    h = (b - a) / N 
    u[1, 1] = a 
    u[1, 2] = α

    for i in 2:n1
        u[i, 2] = u[i-1] + h * f(u[i-1, 1], u[i-1, 2])
        u[i, 1] = a + (i - 1) * h
    end

    return u
end



df(t, y) = y - t^2 +1



u0 = [0.5, 0.0]
a = 0
b = 2
n = 10
tlin = 0:0.2:2

eulers_method(df_, u0, a, b, n)












using Flux

df_ = Chain(Dense(3, 10, elu), Dense(10, 2))




function eulers_method(f, α, a, b, N)

    n1 = N + 1
    u = zeros(n1, 2)

    h = (b - a) / N 
    u[1, 1] = a 
    u[1, 2] = α

    for i in 2:n1
        u[i, 2] = u[i-1] + h * f(u[i-1, 1], u[i-1, 2])
        u[i, 1] = a + (i - 1) * h
    end

    return u
end


dt = 0.01

() -> (x^2) |> 5




function euler_step(df, dt, state, action)

    return euler_update(state, df(vcat(state, action)), dt)

end

function euler_update(h, df, dt)

    return h + dt * df

end


euler_step(df_, 0.01, [0.5, 0.2], 0.3)



















println("hello")

# # Initialise openai gym
# # p.state_size = env.observation_space.shape[1]
# # p.action_size = env.action_space.shape[1]
# # p.action_bound = env.action_space.high[1]
# # p.critic_hidden = [(200, 200)]
# # p.actor_hidden = [(200, 200)]
# # p.batch_size = 128
# # p.mem_size = 1000000
# # p.frames = 0
# # p.max_episodes = 200
# # p.max_episodes_length = 1000
# # p.γ = 0.99f0     # discount rate
# # p.τ = 0.001f0 # for running average while updating target networks
# # p.η_actor = 0.0001f0   # Learning rate, for the optimiser
# # p.η_critic = 0.001f0 # Learning Rate



# using Flux
# using DiffEqFlux
# using OrdinaryDiffEq
# using DifferentialEquations
# using SciMLSensitivity

# u0 = [1.0; 1.0; 1.0]

# down = Dense(3, 2)

# nn = Chain(Dense(2, 50, tanh), Dense(50, 2))

# nn_ode = NeuralODE(nn, (0.0f0, 0.01), Tsit5(),
#     save_everystep=false,
#     reltol=1e-3, abstol=1e-3,
#     save_start=false)


# m = Chain(down, nn)

# nn_ode(u0)

# using Random
# rng = Random.default_rng()

# model_gpu = Chain(Dense(2, 50, tanh), Dense(50, 2))
# p, re = Flux.destructure(model_gpu)
# dudt!(u, p, t) = re(p)(u)

# # Simulation interval and intermediary points
# tspan = (0.0f0, 1.0f0)
# tsteps = 0.0f0:1.0f-2:1.0f0

# u0 = Float32[1.0; 1.0]
# prob_gpu = ODEProblem(dudt!, u0, tspan, p)


# # Runs on a GPU
# sol_gpu = solve(prob_gpu, Tsit5(), saveat=tsteps)




# function lotka(du, u, p, t)
#     du[1] = p[1] * u[1] - p[2] * u[1] * u[2] * u[3]
#     du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] * u[3]
# end

# p = [1.5, 1.0, 3.0, 1.0]
# prob = ODEProblem(lotka, [1.0, 1.0, 1.0], (0.0, 10.0), p)
# sol = solve(prob)



# plot(sol)


# using Conda
# using PyCall
# using Plots


# f(x) = 2x + 0.5
# g(x) = -0.3x + 2

# f(2)
# f(3)
# f(4)

# plot(f)
# plot!(g)


# h(x) = x^2 - 3
# j(x) = 0.1x^3

# plot(h)
# plot!(j)



# u0 = Float32[0.8; 0.8; 0.8]
# tspan = (0.0f0, 25.0f0)

# ann = Chain(Dense(3, 10, tanh), Dense(10, 2))

# p1, re = Flux.destructure(ann)
# p2 = 1.1
# p3 = [p1; p2]
# ps = Flux.params(p3)

# function dudt_(du, u, p, t)
#     s, a = u[1:2], u[3]
#     du[1] = re(p[1:62])(u)[1]
#     du[2] = p2 * du[1]
# end
# prob = ODEProblem(dudt_, u0, tspan, p3)
# concrete_solve(prob, Tsit5(), u0, p3, abstol=1e-8, reltol=1e-6)





# fθ = Chain(Dense(3, 50, tanh), Dense(50, 2))

# X = hcat(𝒟...)

# s = hcat(X[1, :]...)
# a = hcat(X[2, :]...)
# r = hcat(X[3, :]...)
# s′ = hcat(X[4, :]...)


# x = hcat(s', a')


# Ŝ = []
# state = s'[1, :]

# for (obs, action) in zip(eachcol(s), a)
#     state = state + 0.001 * fθ(hcat(obs..., action)')
#     println(f_n(hcat(obs..., action)'))
#     append!(Ŝ, state)
# end

# Ŝ = reshape(Ŝ, (2, 200))



# import Flux.params
# using Flux.Optimise

# θ = params(fθ)
# dθ = gradient(() -> Flux.mse(Ŝ, s′), θ)

# update!(Optimise.Adam(0.1), params(fθ), dθ)

# θ == params(θ)




# using DifferentialEquations
# f(u, p, t) = 1.01 * u
# u0 = 1 / 2
# tspan = (0.0, 0.1)
# prob = ODEProblem(f, u0, tspan)

# sol = solve(prob, saveat=0.01)



# nn = Chain(Dense(2, 50, tanh), Dense(50, 2))
# u0 = [1.0; 0.5]
# tspan = (0.0, 0.1)
# prob = NeuralODE(nn, tspan)
# prob(u0)



# nn = Chain(Dense(3, 50, tanh), Dense(50, 2))
# u0 = [1.0; 0.5; 0.25]
# tspan = (0.0, 0.1)
# prob = NeuralODE(nn, tspan)
# prob(u0)









# nn = Chain(Dense(2, 50, tanh), Dense(50, 2))

# nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
#     save_everystep=false,
#     reltol=1e-3, abstol=1e-3,
#     save_start=false)


# f_n(u0)