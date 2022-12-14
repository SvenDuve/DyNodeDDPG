function solveDyNodeStep(nn, s, a; ΔT=0.001)

    ŝ = s' + ΔT * nn(vcat(vcat(s...), a))'

    return ŝ

end #sovlveDyNodeStep