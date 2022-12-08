

function remember(mem_size, s::Vector{Float32}, a::Vector{Float64}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟) >= mem_size
        deleteat!(𝒟, 1)
    end
    push!(𝒟, [s, a, r, s′, t])
end #remember


function sampleBuffer(agent::DDPGAgent)
    minibatch = sample(𝒟, p.batch_size)
    X = hcat(minibatch...)
    S = hcat(X[1, :]...)
    A = hcat(X[2, :]...)
    R = hcat(X[3, :]...)
    S′ = hcat(X[4, :]...)
    return (S, A, R, S′)
end #sampleBuffer





function sampleBuffer(agent::DyNodeModel)

    numEps = size(𝒟_dyNode)[1] ÷ p.max_episodes_length
    @show numEps
    transInds = vcat([collect(((i-1)*p.max_episodes_length+1:i*p.max_episodes_length-p.batch_length)) for i in 1:Int(numEps)]...)
    @show transInds
    indStart = sample(transInds, p.batch_size) # to set up dynode_batch_size -> 64 in the paper
    @show indStart
    slices = [collect(i:i+p.batch_length-1) for i in indStart]
    @show slices


    # minibatch = sample(𝒟, p.batch_size)
    # X = hcat(minibatch...)
    # S = hcat(X[1, :]...)
    # A = hcat(X[2, :]...)
    # R = hcat(X[3, :]...)
    # S′ = hcat(X[4, :]...)
    # return (S, A, R, S′)
    return slices
end #sampleBuffer



# def sample_dynode(self):

#         inter =  self.capacity//self.episode_length if self.full else self.idx//self.episode_length

#         numbs = np.concatenate([np.arange((i)*self.episode_length, 
#                                 ((i+1)*self.episode_length)-self.batch_length) for i in range(inter)])

#         idxs = np.random.choice(numbs, size=self.dynode_batch_size)
#         idxses = np.asarray([np.arange(idx, idx + self.batch_length) for idx in idxs])
#         vec_idx = idxses.transpose().reshape(-1)

#         obses = torch.as_tensor(self.obses[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
#                                 *self.obs_shape), device=self.device).float()
#         actions = torch.as_tensor(self.actions[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
#                                 *self.action_shape), device=self.device)
#         rewards = torch.as_tensor(self.rewards[vec_idx].reshape(self.batch_length, self.dynode_batch_size, 1),
#                                 device=self.device)
#         next_obses = torch.as_tensor(self.next_obses[vec_idx].reshape(self.batch_length, self.dynode_batch_size,
#                                 *self.obs_shape),  device=self.device).float()
#         not_dones = torch.as_tensor(self.not_dones[vec_idx].reshape(self.batch_length, self.dynode_batch_size, 1),
#                                 device=self.device)
#         return obses, actions, rewards, next_obses, not_dones

