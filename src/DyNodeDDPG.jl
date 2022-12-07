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
using Conda, PyCall



include("base.jl")
include("buffer.jl")
include("neuralnetworks.jl")
include("node.jl")
include("train.jl")
include("agents.jl")
include("loss.jl")

#import trainAgent

export Parameter, DDPGAgent, AgentPolicy, trainAgent, greetings, action, Critic, setNetwork

#greet() = print("Hello World!")

end # module DyNodeDDPG
