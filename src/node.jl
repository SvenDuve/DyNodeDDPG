
function setNode(m::DyNodeModel, p::Parameter)

    down = Chain(Dense(p.state_size + p.action_size, 10))
    # down = Flux.Chain(Flux.Dense(p.state_size + p.action_size, 200)) #|> gpu
    dudt = Chain(Dense(10, 10, elu),
        Dense(10, 10, elu),
        Dense(10, 10, elu),
        Dense(10, 10))

    # nn = Flux.Chain(Flux.Dense(200, 200, elu),
    #     Flux.Dense(200, 200, elu),
    #     Flux.Dense(200, 200, elu))

    nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(),
        save_everystep=false,
        reltol=1e-3, abstol=1e-3,
        save_start=false) #|> gpu

    fc = Flux.Chain(Flux.Dense(10, 2))

    return Flux.Chain(down, nn_ode, fc)
    # return nn_ode

end


function setNode(m::NodeModel, p::Parameter)

    down = Dense(p.state_size + p.action_size, 10)

    dudt = Chain(Dense(10, 10, elu),
        Dense(10, 10, elu),
        Dense(10, 10, elu),
        Dense(10, 10))

#    nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(),
    nn_ode = NeuralODE(dudt, (0.0f0, p.dT), Tsit5(),
        save_everystep=false,
        reltol=1e-3, abstol=1e-3,
        save_start=false) #|> gpu

    fc = Dense(10, p.state_size)

    return Flux.Chain(down, nn_ode, first, fc)
    # return nn_ode

end