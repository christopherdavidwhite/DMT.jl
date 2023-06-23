using DrWatson
quickactivate(pwd())
using ArgParse
using ITensors
using Arrow
using Dates
using DataFrames
using ProgressMeter
using DMT
using Serialization
using Base.Filesystem
#using Profile

#DMT.CHECK = false
@cassert false

function getχ(ψ :: MPS)
    L = length(ψ)
    χ = zeros(Int, L-1)
    for j = 1:L-1
	α = commoninds(ψ.data[j], ψ.data[j+1])
	χ[j] = dim(α)
    end
    
    return χ
end

function majorana_energy_density_tensors(L, U)
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
    return B, C, sharpind, bond_energy_ops
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
    end
    for j = L-2:-1:2
        @cassert check_dmc(ψ,j+1,j+4)
        apply_dmt3_rght!(gates[j], ψ, j, sites[j:j+2], dmt_params )
        @cassert check_dmc(ψ,j,j+3)
    end

    apply_dmt3_both!(gates[1], ψ,1, sites[1:3], dmt_params, :l)
    @cassert check_dmc(ψ,1,quiet=false)
end


function run_te(params)

    req_sim_params = [:init, :dt, :dmt_params,:L]
    req_mod_params = [:U]
    @cassert :init ∈ keys(params)
    @cassert :dt   ∈ keys(params)
    @cassert :L    ∈ keys(params)
    @cassert :U    ∈ keys(params)
    @cassert :dmt_params ∈ keys(params)

    @cassert params[:init] == :ε

    @cassert params[:L] % 2 == 1 #want odd
    
    dmt_params = params[:dmt_params]
    delete!(params, :dmt_params)

    dt = params[:dt]
    L = params[:L]

    svnm = (params ∪ dmt_params) |> Dict |> savename
    fn = "/home/cdwhite/scratch/$jobname/$subdate/$commit/$svnm/"
    fn |> dirname |> mkpath
    @show fn
    state_fn = fn * "state.ser"
    jc = Int((L - 1)/2) #center

    B,C,sharpind,energy_density_ops = majorana_energy_density_tensors(params[:L],params[:U])
    energy_density_vecs = [real(energy_density_ops[j]*C[j]*C[j+1]*C[j+2]*B[j]*B[j+1]*B[j+2]) for j = 1:L-2]
    ψ = energydensity_mpo(sharpind, B, C, energy_density_ops)
    @cassert all([inds(v) ⊆ siteinds(ψ) for v in energy_density_vecs])

    gates = trottergates_itensor_example(B,C,energy_density_ops,params[:dt])
    df = (params ∪
          dmt_params ∪
          [:t  => 0.0,
           :step_ctime => 0.0,
           :j  => 1:(params[:L]-2),
           :χ  => getχ(ψ)[1:L-2],
           :norm => norm(ψ),
           :ε => measure_threesite_ops(ψ, energy_density_vecs)
           ]) |> DataFrame
    if L > 8 Arrow.write(fn*"t=0.0.arrow",df) end

    ε0 = measure_threesite_ops(ψ, energy_density_vecs)
    E0 = sum(ε0)
    @showprogress for t = dt:dt:T
        step_ctime = (@timed apply_trotterstep_itensorexample!(gates,dmt_params,ψ)).time
        χ = getχ(ψ)
        
        ε = measure_threesite_ops(ψ, energy_density_vecs)
        tdf = ( params ∪
                dmt_params ∪
                [:t  => t,
                 :step_ctime => step_ctime,
                 :j  => 1:(params[:L]-2),
                 :χ  => χ[1:L-2],
                 :norm => norm(ψ),
                 :ε => ε,
                 ] ) |> DataFrame
        if L > 8 Arrow.write(fn*savename(@dict t)*".arrow",tdf) end

        thresh = 1e-2/L^2
        if abs(ε[end]) > thresh*abs(E0) || abs(ε[1]) > thresh*abs(E0)
            break;
        end

        if t % 10 == 1.0
            f = open(state_fn,"w")
            serialize(f, @dict gates params dmt_params ψ)
            close(f)
        end
    end

    # if made it to the end without being killed, remove the checkpoint file
    println("removing checkpoint file")
    f = rm(state_fn, force=true)
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--length" "-L"
      arg_type = Int
      default = 17
    "--maxdim"
      arg_type = Int
      default = 16
    "--cutoff"
      arg_type = Float64
      default = 1e-15
    "--subdate"
      arg_type = String
      default = "$(today())"
    "--interaction" "-U" 
      arg_type = Float64
      default = 0.3
    "--time" "-T"
      arg_type = Int
      default = 200
    "--dt"
      arg_type = Float64
      default = 0.125
end
    
parsed_args = parse_args(ARGS, s, as_symbols=true)
Ls     = [9,parsed_args[:L]]
maxdim = parsed_args[:maxdim]
dt     = parsed_args[:dt]
U      = parsed_args[:U]
T      = parsed_args[:T]
subdate = parsed_args[:subdate]
cutoff = parsed_args[:cutoff]

trotter = :bous
jobname = "T111"

using LibGit2
repo = GitRepo(projectdir())
commit = "$(LibGit2.GitShortHash(repo |> LibGit2.head |> LibGit2.GitHash, 6))"

init = :ε

skip_identity = true
for L in Ls
    dmt_params = @dict maxdim skip_identity cutoff
    params = @dict dmt_params trotter jobname subdate T U commit init L dt
    run_te(params)
end
