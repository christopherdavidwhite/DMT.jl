module InteractingMajoranaModel

using ITensors
using DMT
using ProgressMeter
using DrWatson
using FormalAlgebra

export getχ
export infrastructure_tensors
export majorana_energy_density_tensors_msymmetric
export majorana_energy_density_tensors_paulisymmetric
export majorana_energy_current_ops
export majorana_energy_current_mpo
export majorana_energy_current_vecs
export energydensity_mpo
export trottergates_itensor_example
export apply_trotterstep_itensorexample!
export run_te
export current
export currentσ



function current(L,U,j)
    @assert 3 < j 
    @assert j <= L-5
    η = Operator.(Array{Bool}.(majorana_basis(L)))
    P(k) = η[k]η[k+2]
    A(k) = η[k]*η[k+1]*η[k+2]*η[k+4]
    B(k) = η[k]*η[k+2]*η[k+3]*η[k+4]
    C(k) = η[k]*η[k+1]*η[k+2]*η[k+4]*η[k+5]*η[k+6]
    D(k) = η[k]*η[k+4]
    current = -im*2* ( P(j) - U *   ( +im*A(j-2) + im*A(j-1) + im* B(j-1) + im*B(j))
                            + U^2 * ( D(j-1) - C(j-3) - C(j-2) - C(j-1) ) )
    return current
end

currentσ(L,U,j) = current(2*L,U,2*j+1) |> jordanwigner

function getχ(ψ :: MPS)
    L = length(ψ)
    χ = zeros(Int, L-1)
    for j = 1:L-1
	α = commoninds(ψ.data[j], ψ.data[j+1])
	χ[j] = dim(α)
    end
    
    return χ
end

function infrastructure_tensors(s)
    L = length(s)
    C = [combiner(s[j], s[j]') for j = 1:L]
    
    B, sharpind = [ bchg_tensor([op("I", s[j]),
				 op("X", s[j]),
				 op("Y", s[j]),
				 op("Z", s[j])],
				C[j])
		    for j = 1:L ] |> unzip

    return (B,C,sharpind)
end

function majorana_energy_current_ops(U,s)
    L = length(s)
    curr = currentσ(5,U,2)
    return [cdw_op(curr, s[j-1:j+3]) for j = 2:L-3 ]
end

function double_hermitian(tensor, B, C)
    dt = tensor
    for c in C dt *= c end
    for b in B dt *= b end
    @cassert imag(dt) |> norm .< 1e-10
    return real(dt)
end


# The 2^2.5 comes from my convention about where to cancel the Hilbert space factor of 2^L.
#
# I want to silently map all my operators A
#    A -> 2^(L/2) A
# so that
#   \| A \|^2 = tr A* A
# or
#   < A, B > = tr A* B
# is something reasonable (i.e. independent of system size).
# This plays really nicely with my choice of orthogonal, Hermitian basis:
# an MPO representation of σ^x_j, for example, has tensors
#
#  [1]     [1] [0] [1]     [1]
#  [0] ... [0] [1] [0] ... [0]
#  [0]     [0] [0] [0]     [0]
#  [0]     [0] [0] [0]     [0] .
#
# Easy! Orthonormal!
#
# The problem comes when I take an arbitrary 5-site operator and apply double_hermitian.
# double_herimitian just turns this into a 5-leg tensor.
# But when I take an expectation value against some other operator,
# I should think of that 5-leg tensor as an MPO
#
#  [1]     [1]   [       ]  [1]     [1]
#  [0] ... [0]   [   A   ]  [0] ... [0]
#  [0]     [0]   [       ]  [0]     [0]
#  [0]     [0]   [       ]  [0]     [0] .
#
# Each of the [1 0 0 0] "identity" tensors carries an implicit 1/sqrt(2).
# So far so good.
# 
# But that only gets us up to 2^{ (L-5)/2 }.
# We need to account for the extra 2^(5/2) from the five sites where that tensor sits.
# That's the 2^2.5 here.

majorana_energy_current_vecs(B,C, current_ops) = [ double_hermitian(curr, B[j:j+4], C[j:j+4]) / 2^2.5 for (j,curr) = enumerate(current_ops) ]

function majorana_energy_current_mpo(sharpind, current_vecs,j)
    L = length(sharpind)
    Aψ = [onehot(sharpind[j] => 1) for j = 1:L]
    Aψ[j:j+4] .= MPS(current_vecs[j], sharpind[j:j+4])
    ψ = MPS(Aψ)
    orthogonalize!(ψ,1)
    dmc!(ψ)
    @cassert check_dmc(ψ,1,quiet=false)
    return ψ
end

function majorana_energy_density_tensors_msymmetric(U, s)
    L = length(s)
    
    gates = ITensor[]
    bond_energy_ops = ITensor[]
    for j in 1:(L - 2)
	s1 = s[j]
	s2 = s[j+1]
        s3 = s[j+2]
	hj =    op("I", s1) * op("Z", s2) * op("I", s3) + 
                op("X", s1) * op("X", s2) * op("I", s3) +
	    U * op("X", s1) * op("I", s2) * op("X", s3) +
	    U * op("Z", s1) * op("Z", s2) * op("I", s3)
        #=
	if  j == 2
	    hj +=     op("X", s1) * op("X", s2) * op("I", s3) +
	          U * op("Z", s1) * op("Z", s2) * op("I", s3) +
                  U * op("Z",s1)*op("I",s2)*op("I",s3)
        elseif j == L-1
	    hj +=  op("I",s1)*op("I",s2)*op("Z",s3)
	end
        =#


	push!(bond_energy_ops, hj)
    end
    return bond_energy_ops
end

function majorana_energy_density_tensors_paulisymmetric(U, s)
    L = length(s)
    gates = ITensor[]
    bond_energy_ops = ITensor[]
    
    for j in 1:(L - 2)
	s1 = s[j]
	s2 = s[j + 1]
        s3 = s[j + 2]
	hj =
	    U * op("X", s1) * op("I", s2) * op("X", s3) +
            
	    1/2 * op("X", s1) * op("X", s2) * op("I", s3) +
	    1/2 * op("I", s1) * op("X", s2) * op("X", s3) +
            
	    U * 1/2 * op("Z", s1) * op("Z", s2) * op("I", s3) +
	    U * 1/2 * op("I", s1) * op("Z", s2) * op("Z", s3) +
            
            1/3 * op("Z",s1)*op("I",s2)*op("I",s3) +
            1/3 * op("I",s1)*op("Z",s2)*op("I",s3) + 
            1/3 * op("I",s1)*op("I",s2)*op("Z",s3)
        
	if     j == 1
	    hj +=     1/2 * op("X", s1) * op("X", s2) * op("I", s3) +
	          U * 1/2 * op("Z", s1) * op("Z", s2) * op("I", s3) +
                      2/3 * op("Z",s1)*op("I",s2)*op("I",s3)
        elseif j == 2
            hj += 1/3 * op("Z",s1)*op("I",s2)*op("I",s3)
        elseif j == L-3
            hj += 1/3 * op("I",s1)*op("I",s2)*op("Z",s3)
        elseif j == L-2
	    hj +=     1/2 * op("I", s1) * op("X", s2) * op("X", s3) +
	          U * 1/2 * op("I", s1) * op("Z", s2) * op("Z", s3) +
                      2/3 * op("I",s1)*op("I",s2)*op("Z",s3)
	end



	push!(bond_energy_ops, hj)
    end
    return bond_energy_ops
end

function energydensity_mpo(sharpind, B, C, bond_energy_ops :: Vector{ITensor})
    L = length(sharpind)
    Aψ = [onehot(sharpind[j] => 1) for j = 1:L]

    # eg: L = 5
    # ( L - 1 ) / 2 = 2
    # energy density on 2,3,4
    # - * * * -
    j = Int((L-1)/2) #center
    
    hj = bond_energy_ops[j]*C[j]*C[j+1]*C[j+2]*B[j]*B[j+1]*B[j+2]
    @cassert imag(hj) |> norm < 1e-10
    hjr = real(hj)

    U,S,V = svd(hjr, sharpind[j])
    
    Aψ[j] = U*S

    Up,Sp,Vp = svd(V, sharpind[j+1], commoninds(V,S)...)
    
    Aψ[j+1] = Up*Sp
    Aψ[j+2] = Vp
    
    ψ = MPS(Aψ)
    
    orthogonalize!(ψ,1)
    dmc!(ψ)
    @cassert check_dmc(ψ,1,quiet=false)
    return ψ
end

function trottergates_itensor_example(B               :: Vector{ITensor},
                                      C               :: Vector{ITensor},
                                      bond_energy_ops :: Vector{ITensor},
                                      dt :: Number)
    gates = ITensor[]
    for (j,hj) = enumerate(bond_energy_ops)
        Gj = exp(-im * dt / 2 * hj)
        Gjd = double(Gj,C[j:j+2])
        Gjdh = B[j]*B[j+1]*B[j+2]*Gjd*dag(B[j]'*B[j+1]'*B[j+2]')
        @cassert Gjdh |> imag |> norm < 1e-10
        push!(gates, real(Gjdh))
    end
    return gates
end

function apply_trotterstep_itensorexample!(gates, dmt_params, ψ)
    L = length(ψ)
    sites = siteinds(ψ)
    orthogonalize!(ψ,1)
    @cassert check_dmc(ψ,1,quiet=false)

    gc_freq = get(dmt_params, :gc_freq, 10)

    # want to truncate on the *trailing* bond
    # that is:
    #  - when going rightward, want to truncate on left bond
    #    hence apply_dmt3_left!
    #  - when going leftward, want to truncate on right bond
    #    hence apply_dmt3_right!
    for j = 1:L-2
        @cassert check_dmc(ψ,j-1,j+2)
        apply_dmt3_left!(gates[j], ψ, j, sites[j:j+2], dmt_params)
        @cassert check_dmc(ψ,j,j+3)
        if (nothing != gc_freq && 0 == j % gc_freq) Base.GC.gc(); end
    end
    for j = L-2:-1:2
        @cassert check_dmc(ψ,j+1,j+4)
        apply_dmt3_rght!(gates[j], ψ, j, sites[j:j+2], dmt_params )
        @cassert check_dmc(ψ,j,j+3)
        if (nothing != gc_freq && 0 == j % gc_freq) Base.GC.gc() end
    end

    apply_dmt3_both!(gates[1], ψ,1, sites[1:3], dmt_params, :l)
    @cassert check_dmc(ψ,1,quiet=false)
end



end # module InteractingMajoranaModel
