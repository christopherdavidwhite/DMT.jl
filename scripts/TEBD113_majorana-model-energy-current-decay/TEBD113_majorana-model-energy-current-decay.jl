using DrWatson
quickactivate(pwd())
using ITensors
using Revise
using InteractingMajoranaModel
using ArgParse
using Arrow
using Dates
using DataFrames
using ProgressMeter
using DMT
using Serialization
using Base.Filesystem

dmt_set_check!(false)
@cassert false

s = ArgParseSettings()
@add_arg_table! s begin
    "--length" "-L"
      arg_type = Int
      default = 17
    "--maxdim"
      arg_type = Int
      default = 16
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
    "--resume"
      arg_type = String
    default = nothing
end


parsed_args = parse_args(ARGS, s, as_symbols=true)
@show parsed_args

function run_te(params, state = nothing)

    req_sim_params = [:init, :dt, :dmt_params,:L]
    req_mod_params = [:U]
    @cassert :init ∈ keys(params)
    @cassert :dt   ∈ keys(params)
    @cassert :L    ∈ keys(params)
    @cassert :U    ∈ keys(params)
    @cassert :dmt_params ∈ keys(params)

    @cassert params[:init] == :ε

    @cassert params[:L] % 2 == 1 #want odd

    if state != nothing
        println("resuming")
        dmt_params = state[:dmt_params]
        params     = state[:params]
        gates      = state[:gates]
        current_vecs = state[:current_vecs]
        ψ          = state[:ψ]
        tinit      = state[:t] + params[:dt]


        ts = tinit:params[:dt]:params[:T]
    else
        dmt_params = params[:dmt_params]
        delete!(params, :dmt_params)

        jc = params[:L] / 2 |> floor |> Int


        s = siteinds("S=1/2", params[:L])
        B,C,sharpind       = infrastructure_tensors(s)
        energy_density_ops = majorana_energy_density_tensors_paulisymmetric(U,s)
        current_ops        = majorana_energy_current_ops(params[:U],s)
        current_vecs       = majorana_energy_current_vecs(B,C,current_ops)
        ψ                  = majorana_energy_current_mpo(sharpind, current_vecs, params[:L]/2 |> floor |> Int)
        
        @cassert all([inds(v) ⊆ siteinds(ψ) for v in energy_density_vecs])
        
        gates = trottergates_itensor_example(B,C,energy_density_ops,params[:dt])
        ts = params[:dt]:params[:dt]:T
        
    end

    jobname = params[:jobname]
    commit = params[:commit]
    subdate = params[:subdate]
    dt = params[:dt]
    L = params[:L]
    
    svnm = (params ∪ dmt_params) |> Dict |> savename
    fn = datadir("$jobname/$subdate/$commit/$svnm/")
    fn |> dirname |> mkpath
    @show fn; flush(stdout)
    state_fn = fn * "state.ser"


    χ = getχ(ψ)
    Base.GC.gc()

    t = 0.0
    
    curr = measure_nsite_ops(ψ, current_vecs)
    tdf = ( params ∪
            dmt_params ∪
            [:t  => t,
             :step_ctime => 0.0,
             :j  => 1:(params[:L]-4),
             :χ  => χ[1:L-4],
             :norm => norm(ψ),
             :curr => curr,
             ] ) |> DataFrame
    if L > 9 Arrow.write(joinpath(fn,"$t.arrow"),tdf) end

    @showprogress for t = ts
        step_ctime = (@timed apply_trotterstep_itensorexample!(gates,dmt_params,ψ)).time
        χ = getχ(ψ)
	Base.GC.gc()
        
        curr = measure_nsite_ops(ψ, current_vecs)
        tdf = ( params ∪
                dmt_params ∪
                [:t  => t,
                 :step_ctime => step_ctime,
                 :j  => 1:(params[:L]-4),
                 :χ  => χ[1:L-4],
                 :norm => norm(ψ),
                 :curr => curr,
                 ] ) |> DataFrame
        if L > 9 Arrow.write(joinpath(fn,"$t.arrow"),tdf) end

        thresh = 1e-2/L^2
        if χ[1] > 1 || χ[end] > 1
            break;
        end

        if t % 10 == 1.0
            f = open(state_fn,"w")
            serialize(f, @dict params dmt_params gates current_vecs ψ t)
            close(f)
        end
	Base.GC.gc()
    end

    # if made it to the end without being killed, remove the checkpoint file
    println("removing checkpoint file")
    f = rm(state_fn, force=true)
end




if (nothing == parsed_args[:resume])
    Ls     = [9,parsed_args[:L]]
    maxdim = parsed_args[:maxdim]
    dt     = parsed_args[:dt]
    U      = parsed_args[:U]
    T      = parsed_args[:T]
    subdate = parsed_args[:subdate]

    trotter = :bous
    jobname = "T113"
    skip_identity = true


    using LibGit2
    repo = GitRepo(projectdir())
    commit = "$(LibGit2.GitShortHash(repo |> LibGit2.head |> LibGit2.GitHash, 6))"
    
    init = :curr
    
    
    for L in Ls
        dmt_params = @dict maxdim skip_identity
        params = @dict dmt_params trotter jobname subdate T U commit init L dt
        run_te(params)
    end
else
    state = deserialize(parsed_args[:resume])
    run_te(Dict(), state)
end
