using DrWatson
quickactivate(pwd())
using ITensors
using Arrow
using Dates
using DataFrames
using ProgressMeter
using DMT

function ising_energy_density_tensors(L, hx, hz)
    s = siteinds("S=1/2", L; conserve_qns=false)

    C = [combiner(s[j], s[j]') for j = 1:L]

    B, sharpind = [ bchg_tensor([op("I", s[j]),
				 op("X", s[j]),
				 op("Y", s[j]),
				 op("Z", s[j])],
				C[j])
		    for j = 1:L ] |> unzip

    gates = ITensor[]
    bond_energy_ops = ITensor[]
    for j in 1:(L - 1)
	s1 = s[j]
	s2 = s[j + 1]
	hj =
	    op("Z", s1) * op("Z", s2) +
	    1 / 2 * hz * op("Z", s1) * op("I", s2) +
	    1 / 2 * hz * op("I", s1) * op("Z", s2) +
	    1 / 2 * hx * op("X", s1) * op("I", s2) +
	    1 / 2 * hx * op("I", s1) * op("X", s2)
	if j == 1
	    hj += 1 / 2 * hz * op("Z", s1) * op("I", s2) +
		1 / 2 * hx * op("X", s1) * op("I", s2)
	end

	if j == L-1
	    hj += 1 / 2 * hz * op("I", s1) * op("Z", s2) +
		1 / 2 * hx * op("I", s1) * op("X", s2)
	end

	push!(bond_energy_ops, hj)
    end
    return B, C, sharpind, bond_energy_ops
end

function energydensity_mpo(sharpind, B, C, bond_energy_ops :: Vector{ITensor}, j :: Integer)
    L = length(sharpind)
    Aψ = [onehot(sharpind[j] => 1) for j = 1:L]
    jc = Int(L/2) #center
    hjc = bond_energy_ops[jc]*C[jc]*C[jc+1]*B[jc]*B[jc+1]
    @cassert imag(hjc) |> norm < 1e-10
    hjcr = real(hjc)
    U,S,V = svd(hjcr, sharpind[jc])
    Aψ[jc] = U*S
    Aψ[jc+1] = V
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
	Gjd = double(Gj,C[j:j+1])
	Gjdh = B[j]*B[j+1]*Gjd*dag(B[j]'*B[j+1]')
	@cassert Gjdh |> imag |> norm < 1e-10
	push!(gates, real(Gjdh))
    end
    return gates
end

function apply_trotterstep_itensorexample!(gates, χmax, ψ)
    L = length(ψ)
    for j = 1:L-1
	@cassert check_dmc(ψ,j,quiet=false)
	apply_dmt!(gates[j], ψ, j, χmax, :r)
	@cassert check_dmc(ψ,j+1,quiet=false)
    end
    for j = L-1:-1:1
	@cassert check_dmc(ψ,j+1,quiet=false)
	apply_dmt!(gates[j], ψ, j, χmax, :l)
	@cassert check_dmc(ψ,j,quiet=false)
    end
end

function run_te(params)

    @assert params[:init] == :ε
    dt = params[:dt]

    fn = datadir("$jobname/$subdate/$commit/"* savename(params) * ".arrow")
    fn |> dirname |> mkpath
    #heap_fn = datadir("profiling/$jobname/$subdate/$commit/"* savename(params) * ".heapsnapshot")
    #heap_fn |> dirname |> mkpath
    @show fn
    jc = Int(params[:L]/2) #center

    B,C,sharpind,bond_energy_ops = ising_energy_density_tensors(params[:L],params[:hx],params[:hz])
    ψ = energydensity_mpo(sharpind, B, C, bond_energy_ops,jc)
    gates = trottergates_itensor_example(B,C,bond_energy_ops,params[:dt])
    df = ( params ∪ [:t  => 0,
		     :step_ctime => 0,
		     :j  => 1:params[:L]-1,
		     :χ  => getχ(ψ),
		     :norm => norm(ψ),
		     :ev => nnev_as_vector(ψ),
		     ] ) |> DataFrame

    @showprogress for t = dt:dt:T
	step_ctime = (@timed apply_trotterstep_itensorexample!(gates,params[:χmax],ψ)).time
        if t % 1 == 0
	    df = [df; ( params ∪ [:t  => t,
			          :step_ctime => step_ctime,
			          :j  => 1:params[:L]-1,
			          :χ  => getχ(ψ),
			          :norm => norm(ψ),
			          :ev => nnev_as_vector(ψ),
			          ] ) |> DataFrame ]
	    Arrow.write(fn,df)
        end
        #if t % 10 == 0
        #    Profile.take_heap_snapshot(heap_fn)
        #end
    end

    return df
end


function getχ(ψ :: MPS)
    L = length(ψ)
    χ = zeros(Int, L-1)
    for j = 1:L-1
	α = commoninds(ψ.data[j], ψ.data[j+1])
	χ[j] = dim(α)
    end
    
    return χ
end

lgLmax = 7
lgχmax = 9

Ls = 2 .^ (3:lgLmax)
χmaxs = 2 .^ (1:lgχmax)
dts = [0.125]

dumb_time_estimate(d) = d[:L]*d[:χmax]^3/d[:dt]
variable_param_list = [@dict L χmax dt for L = Ls, χmax = χmaxs, dt = dts] |> arr1d

sort!(variable_param_list, by=dumb_time_estimate)


trotter = :bous
jobname = "T104"
subdate = today()
T = 70
hx = 1.4
hz = 0.9045

using LibGit2
repo = GitRepo("/home/christopher/work/2014-12-TEBD/")
commit = "$(LibGit2.GitShortHash(repo |> LibGit2.head |> LibGit2.GitHash, 6))"

init = :ε

for vparam in variable_param_list
    params = Dict((@dict trotter jobname subdate T hx hz commit init) ∪ vparam)
    run_te(params)
end
