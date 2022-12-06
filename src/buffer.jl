

function remember(s::Vector{Float32}, a::Vector{Float64}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟) >= p.mem_size
        deleteat!(𝒟, 1)
    end
    push!(𝒟, [s, a, r, s′, t])
end