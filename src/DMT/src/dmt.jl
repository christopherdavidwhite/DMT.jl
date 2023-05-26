export thick_qr_qdag
export dmc_gauge
export check_dmc
export otrace
export dmt
export apply_dmt!
export bchg_tensor
export double
export dmc!
export sweep_dmc!
export onsite_expectation_values
export nnev_as_vector

function thick_qr_qdag(A :: ITensor, Linds :: Vector{<:Index})
    Rinds = setdiff(inds(A) , Linds)
    CL = combiner(Linds); CLind = inds(CL)[1]
    CR = combiner(Rinds); CRind = inds(CR)[1]
    qthing,r = array(CL * A * CR, CLind, CRind) |> qr
    qdarr = Array(qthing')
    newα = Index(size(qdarr,1), "qr")
    @assert dim(newα) == dim(CLind)
    qd = CL*ITensor(qdarr, newα , CLind)
    return (qd,newα)
end

#"free" means---whole vector space to preserve

#free and out look the same as far as operator-trace
function otrace(A :: ITensor, αfree :: Vector{<:Index})
    @assert αfree ⊆ inds(A)
    αins = setdiff(inds(A), αfree)
    if isempty(αins)
        return A
    else
        return A * prod(onehot(αin => 1) for αin in αins)
    end
end


function dmc_gauge(A :: ITensor, αfree :: Vector{<:Index}, αout :: Index)
    free_inds = αfree ∪ [αout]
    qd,r = thick_qr_qdag(otrace(A,free_inds), [αout])
    newα = setdiff(inds(qd), inds(A))[1]
    return qd, newα
end

function check_dmc(A :: ITensor,
		   αfree :: Vector{<:Index},
		   αout :: Index;
		   quiet=true
		   )

    @assert αout ∈ inds(A)
    @assert αfree ⊆ inds(A)
    Cfree = combiner(αfree...)
    βfree = inds(Cfree)[1]
    dfree = dim(βfree)
    δ = array(otrace(A*Cfree,[βfree,αout]), βfree, αout)[:,dfree+1:end] |> norm
    tensor_norm = norm(A)
    good = δ/tensor_norm <= 1e-10
    if !quiet && !good println("check_dmc: tensor norm $tensor_norm; block norm δ = $δ") end
    return( good )
end


function dmt(A    :: ITensor,
	     σL   :: Index,
	     σR   :: Index,
	     αL   :: Vector{<:Index},
	     χmax :: Integer)

    @assert [σL, σR] ∪ αL ⊆ inds(A)

    Linds = [σL]∪ αL
    U,s,V = svd(A,Linds...) ###### SVD
    αu = commoninds(U,s)[1]
    αv = commoninds(V,s)[1]

    if dim(αu) < χmax # not doing any truncation
        return U,s,V
    end
        
    qLd,βL = dmc_gauge(U,[σL], αu)
    qRd,βR = dmc_gauge(V,[σR], αv)
    Uq = U*qLd

    Vq = qRd*V
    M = dag(qLd)*s*dag(qRd)


    Ma = array(M, βL, βR)

    Ma_sub = Ma[5:end, 5:end]

    F = svd(Ma_sub)    ###### SVD
    St = F.S
    χmax = min(χmax, size(St)[1])
    St[χmax+1:end] .= 0
    Ma_subtrunc = F.U*Diagonal(St)*F.Vt

    Ma[5:end, 5:end] = Ma_subtrunc

    F = svd(Ma)      ###### SVD
    χp1 = findfirst(F.S .< 1e-14*norm(F.S))
    if χp1 != nothing
	χ = χp1 - 1
    else
	χ = size(F.S,1)
    end

    βu = Index(χ)
    βv = Index(χ)

    Utrunc = ITensor(F.U[:,1:χ], βL, βu)
    Strunc = diagITensor(F.S[1:χ], βu, βv)
    Vtrunc = ITensor(F.Vt[1:χ,:], βv, βR)

    return Uq*Utrunc, Strunc, Vtrunc*Vq
end

function double(A :: ITensor, C :: Vector{ITensor})
    #@assert inds(A) ⊆ union([c |> inds |> collect for c in C]...)
    A = mapprime(A,1,2)
    AAt = A*dag(A')
    for c in C
	AAt *= c
	AAt *= c''
    end
    return mapprime(AAt, 2, 1)
end

function bchg_tensor(B :: Array{<:ITensor},
		     c :: ITensor)
    cind = inds(c)[1]
    d = length(B)
    Bchga = zeros(Complex{Float64}, d,d)
    for (jb,b) in B |> enumerate
	@cassert d == b |> size |> prod
	Bchga[jb, :] = (b*c |> array) / norm(b)
    end

    outind = sim(cind)
    Bchg = ITensor(Bchga, outind, cind)
    return Bchg,outind
end

function apply_dmt!(G :: ITensor,
                    ψ :: MPS,
                    j :: Integer,
                    χmax :: Integer,
                    center_to :: Symbol =:l)

    ψj   = ψ.data[j]
    ψjp1 = ψ.data[j+1]

    sl = (commoninds(ψj,G) |> scalar)
    sr = (commoninds(ψjp1,G) |> scalar)

    if j == 1
	αL = Index[]
    else
	αL = commoninds(ψj, ψ.data[j-1])
    end

    Gψψ = noprime(G*ψj*ψjp1)
    #U,S,V = svd(Gψψ, sl)
    U,S,V = dmt(Gψψ, sl, sr, αL, χmax)
    
    if center_to == :r
        α = commonind(U,S)
        qd, = dmc_gauge(U, [sl],α)
        ψ.data[j] = U*qd
        ψ.data[j+1] = dag(qd)*S*V
        # new orth center is j+1
        ψ.llim = j
        ψ.rlim = j+2
    elseif center_to == :l
        α = commonind(S,V)
        qd, = dmc_gauge(V, [sr],α)
        ψ.data[j] = U*S*dag(qd)
        ψ.data[j+1] =dag(qd)*V

        # new orth center is j+1
        ψ.llim = j-1
        ψ.rlim = j+1
    else
        error("apply_dmt: center_to=$center_to not supported")
    end
    
    return ψ
end


function check_dmc(ψ, center :: Integer;quiet=true)
    sites = siteinds(ψ)
    L = length(ψ)

    good = true
    for j = L:-1:center+1
	#println("check_dmc left $j")
	αα = commoninds(ψ.data[j-1], ψ.data[j])
        if length(αα) == 0
            thistensor_good = true
        else
	    thistensor_good = check_dmc(ψ.data[j], sites[j:j], scalar(αα),quiet=quiet)
        end
	if !quiet && !thistensor_good
	    println("[leftwards dmc tensor j = $j failed]")
	end
	good = good && thistensor_good
    end

    for j = 1:center-1
	#println("check_dmc rght $j")
	αα = commoninds(ψ.data[j], ψ.data[j+1])
        if length(αα) == 0
            thistensor_good = true
        else
	    thistensor_good = check_dmc(ψ.data[j], sites[j:j], scalar(αα),quiet=quiet)
	    if !quiet && !thistensor_good
	        println("[rghttwards dmc tensor j = $j failed]")
	    end
	    good = good && thistensor_good
        end
    end

    return good
end


# Put into DMC form with center on site 1
# nb this is independent of orthogonality center
dmc!(ψ :: MPS) = sweep_dmc!(ψ, length(ψ), 1)

function sweep_dmc!(ψ :: MPS,
		    old_center :: Integer,
		    new_center :: Integer,)
    @assert 1 <= old_center <= length(ψ)
    @assert 1 <= new_center <= length(ψ)
    sites = siteinds(ψ)
    if old_center == new_center
	return ψ
    elseif old_center < new_center #sweeping from left to right
	for j = old_center:new_center-1
	    ψj   = ψ.data[j]
	    ψjp1 = ψ.data[j+1]

	    αα = commoninds(ψj,ψjp1)
            if length(αα) == 0
                continue
            elseif length(αα) > 1
                error("sweep_dmc: tensors sites $j, $(j+1) have more than one common index")
            else #length(aa) == 1
                α = αα[1]
	        qd,newα = dmc_gauge(ψj, [sites[j]], α)
	        ψ.data[j] = ψj*qd
	        ψ.data[j+1] = ψjp1*dag(qd)
            end
	end
    elseif new_center < old_center
	for j = old_center:-1:new_center+1
	    ψj   = ψ.data[j]
	    ψjm1 = ψ.data[j-1]

	    αα = commoninds(ψj,ψjm1)
            if length(αα) == 0
                continue
            elseif length(αα) > 1
                error("sweep_dmc: tensors sites $j, $(j-1) have more than one common index")
            else #length(aa) == 1
                α = αα[1]
	        qd,newα = dmc_gauge(ψj, [sites[j]], α)
	        ψ.data[j] = ψj*qd
	        ψ.data[j-1] = ψjm1*dag(qd)
	    end
        end
    end
    return ψ
end

function onsite_expectation_values(ψ :: MPS)
    #function assumes DMC with center-site 1
    @cassert check_dmc(ψ,1)

    L = length(ψ)
    sites = siteinds(ψ)
    zL = zeros(L-1)
    zR = zeros(L-1)
    expct = zeros(dim(sites[1]), L)

    # j labels the tensor whose trace we record
    for j = L:-1:2
	ψj = ψ.data[j]
	zR[j-1] = otrace(ψj,Index[]) |> ITensors.scalar
    end

    # k labels the site where we want the expectation value
    # j labels the last tensor left of that site, whose trace we record


    for k = 1:L
	if 2 <= k
	    sweep_dmc!(ψ, k-1,k)
	end
        @cassert check_dmc(ψ, k,quiet=false)

	ψk = ψ.data[k]
	expct[:,k] = otrace(ψk, sites[k:k]) |> array

	if 2 <= k
	    j = k-1
	    ψj = ψ.data[j]
	    zL[j] = otrace(ψj, Index[]) |> ITensors.scalar
	    expct[:,k] *=  prod(zL[1:k-1])
	end

	if k <= L-1
	    expct[:,k] *= prod(zR[k:end]) # sites k+1:L correspond to indices k:L-1 = k:end
	end

    end
    dmc!(ψ)
    return expct
end

function nn_expectation_values(ψ :: MPS)
    #function assumes DMC with center-site 1
    @cassert check_dmc(ψ,1)

    L = length(ψ)
    sites = siteinds(ψ)
    zL = zeros(L-1)
    zR = zeros(L-1)
    d = dim(sites[1])
    expct = zeros(d,d,L-1)

    # j labels the tensor whose trace we record
    for j = L:-1:2
	ψj = ψ.data[j]
	zR[j-1] = otrace(ψj,Index[]) |> ITensors.scalar
    end

    # k labels the site where we want the expectation value
    # j labels the last tensor left of that site, whose trace we record


    for k = 1:L-1
	if 2 <= k
	    sweep_dmc!(ψ, k-1,k)
	end

	ψk = ψ.data[k]
        ψkp1 = ψ.data[k+1]
        #if performance an issue, can reorder these tensor multiplications
        # something like
        #    otrace(ψk, [sites[k:k]] ∪ commoninds(ψk, ψkp1)) *
        #        otrace(ψkp1, [sites[k+1:k+1]] ∪ commoninds(ψk, ψkp1))
        
	expct[:,:,k] = array(otrace(ψk*ψkp1, sites[k:k+1]), sites[k], sites[k+1])

	if 2 <= k
	    j = k-1
	    ψj = ψ.data[j]
	    zL[j] = otrace(ψj, Index[]) |> ITensors.scalar
	    expct[:,:,k] *= prod(zL[1:k-1])
	end

	if k <= L-2
	    expct[:,:,k] *= prod(zR[k+1:end]) # sites k+1:L correspond to indices k:L-1 = k:end
	end
    end

    dmc!(ψ)
    return expct
end

function nnev_as_vector(ψ)
    nnev = nn_expectation_values(ψ)
    return [nnev[:,:,j] for j = 1:size(nnev,3)]
end
