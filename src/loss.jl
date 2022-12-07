function lossCritic(Y::Matrix{Float64}, S::Matrix{Float32}, A::Matrix{Float64})
    V = Qθ(vcat(S, A))
    (Y .- V) .^ 2 |> sum
end


function maxActor(S::Matrix{Float32})
    Qθ(vcat(S, μϕ(S))) |> sum |> mean
end