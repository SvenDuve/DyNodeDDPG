
function setNode(m::DyNodeModel, p::Parameter)

    # down = Flux.Chain(Flux.Dense(p.state_size + p.action_size, 200)) #|> gpu
    dudt = Chain(Dense(p.state_size + p.action_size, 200, elu),
        Dense(200, 200, elu),
        Dense(200, 200, elu),
        Dense(200, 200, elu),
        Dense(200, 3))

    # nn = Flux.Chain(Flux.Dense(200, 200, elu),
    #     Flux.Dense(200, 200, elu),
    #     Flux.Dense(200, 200, elu))

    nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(),
        save_everystep=false,
        reltol=1e-3, abstol=1e-3,
        save_start=false) #|> gpu

    fc = Flux.Chain(Flux.Dense(3, 2))

    return Flux.Chain(nn_ode, fc)
    # return nn_ode

end