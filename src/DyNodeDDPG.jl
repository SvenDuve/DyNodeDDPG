module DyNodeDDPG

using Statistics
using Parameters
using UnPack
using Flux, Flux.Optimise
import Flux.params
using Optimisers
using Distributions
import StatsBase.sample
import StatsBase.AnalyticWeights
using NNlib, Random, Zygote
import Zygote.Buffer
using Conda, PyCall
using DiffEqFlux, DifferentialEquations



include("base.jl")
include("buffer.jl")
include("neuralnetworks.jl")
include("node.jl")
include("train.jl")
include("agents.jl")
include("loss.jl")
include("solver.jl")

#import trainAgent

export Parameter,
    resetParameters,
    DDPGAgent,
    DyNodeModel,
    AgentPolicy,
    Critic,
    Actor,
    Rewards,
    OrnsteinUhlenbeck,
    𝒩,
    Episode,
    action,
    remember,
    sampleBuffer,
    lossCritic,
    maxActor,
    lossReward,
    setNetwork,
    setNode,
    train,
    trainAgent,
    greetings,
    dyNode,
    transition,
    solveDyNodeStep,
    dyNodeLoss


#greet() = print("Hello World!")

end # module DyNodeDDPG
