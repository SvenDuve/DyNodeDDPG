

function remember(mem_size, s::Vector{Float32}, a::Vector{Float64}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟) >= mem_size
        deleteat!(𝒟, 1)
    end
    push!(𝒟, [s, a, r, s′, t])
end #remember


function sampleBuffer()
    minibatch = sample(𝒟, p.batch_size)
    X = hcat(minibatch...)
    S = hcat(X[1, :]...)
    A = hcat(X[2, :]...)
    R = hcat(X[3, :]...)
    S′ = hcat(X[4, :]...)
    return (S, A, R, S′)
end #sampleBuffer
