module DyNodeDDPG

using Statistics
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
include("training.jl")


export Parameters, DDPGAgent, AgentPolicy, trainAgent, greetings, action, Critic, setNetwork

#greet() = print("Hello World!")

end # module DyNodeDDPG
