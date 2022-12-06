

function trainAgent(agent::DDPGAgent, parameters::Parameters)

    # get critic

    # copy critic

    # get actor

    # copy actor


    # set optimiszers

    # set training conditions

    # loop while some conditions

        # interact with the environment by generating some Episode
        # store episode in buffer

        # create a minibatch from the buffer

        # get all parameters for the loss functions

        # Do gradient ascent/ descent on the loss functions

        # update target for critic

        # update target for actor

    return -1
end #trainAgent