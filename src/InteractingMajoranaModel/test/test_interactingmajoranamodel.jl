using FormalAlgebra
using Test
using InteractingMajoranaModel
using ITensors


@testset "ladder to Pauli-representation current" begin
    L = 18
    j = 9
    U = 0.3

    ################### majorana-side current ##################
    η = Operator.(Array{Bool}.(majorana_basis(L)))
    ε(j) = -( im*η[j]*η[j+1] + 0.3*η[j-1]*η[j]*η[j+1]*η[j+2] )
    H = sum(ε(j) for j = 2:L-2)

    dtε = im*(H*ε(j) - ε(j)*H)
    @test (dtε - (current(L,U,j-1) - current(L,U,j)) ).V |> length == 0 


    ############## pauli-side current (formal) ################
    Hσ = jordanwigner(H)
    currentσ(L,U,j) = current(L,U,2*j+1) |> jordanwigner
    εσ(j) = ( ε(2*j) + ε(2*j+1) ) |> jordanwigner

    j = 4
    dtε2 =  im*(Hσ*εσ(j) - εσ(j)*Hσ)
    divσ2 = currentσ(L,U,j-1) - currentσ(L,U,j)
    @test length( (dtε2 - divσ2).V ) == 0

    ################## energy density to tensor ######################
    # check jordan-wigner against hand-coded energy density

    L = 8
    U = 0.3

    η = Operator.(Array{Bool}.(majorana_basis(2*3)))

    εsmall(j) = -( im*η[j]*η[j+1] + U*η[j-1]*η[j]*η[j+1]*η[j+2] ) 
    εσp(j) = ( εsmall(2*j) + εsmall(2*j+1) ) |> jordanwigner
    s = siteinds("S=1/2", L; conserve_qns=false)
    bond_energy_ops = majorana_energy_density_tensors_msymmetric(U,s)

    j = 2
    εj = εσp(1)
    @test ( cdw_op(εj, s[j:j+2]) - bond_energy_ops[2] |> norm ) < 1e-10


    ################ current tensor vs energy density tensor #########
    # 

    function embed_itensor_operator(A :: ITensor, ss :: Vector{<:Index})
        new_ss = setdiff(ss, inds(A))
        for s in new_ss
            A *= delta(s,s')
        end
        return A
    end
    
    ITensors.set_warn_order(17)

    # needs to be L = 8, otherwise current doesn't match. I think this is a boundary thing.
    L = 8
    U = 0.3
    
    s = siteinds("S=1/2", L; conserve_qns=false)
    bond_energy_ops = majorana_energy_density_tensors_msymmetric(U,s)
    εs = [embed_itensor_operator(A, s) for A in bond_energy_ops]
    H = sum(εs)
    curr = [embed_itensor_operator(A,s) for A in majorana_energy_current_ops(U,s)]


    divj = curr[2] - curr[3] #energy density 4

    dtε = im*mapprime(prime(H)*εs[4] - prime(εs[4])*H, 2,1)
    @test norm(dtε - divj) < 1e-10


    #=
    ε(j) |> prettyprint
    println("------------------------")
    ε(j) |> jordanwigner |> prettyprint
    println("------------------------")
    ε(j+1) |> jordanwigner |> prettyprint
    println("------------------------")
    ε(j) + ε(j+1) |> jordanwigner
    @show j
    εσ1 |> prettyprint
    println("========================\ndtε")
    dtε |> prettyprint
    println("========================\n divσ1")
    divσ |> prettyprint
    println("========================\n dtε - divσ1")
    dtε - divσ |> prettyprint
    println("========================\ncurrents")
    current(L,U,j-1) |> jordanwigner |> prettyprint
    println("------------------------\n")
    current(L,U,j+1) |> jordanwigner |> prettyprint
    println("\nSure looks like one is the translate of the other!")

    println("========================\n jσ\n")
    =#
end




