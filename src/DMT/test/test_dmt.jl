a1 = Index(2)
a2 = Index(3)
a3 = Index(5)
a4 = Index(7)


Nr = 10

@testset "thick_qr_qdag" begin
    for r = 1:Nr
	A = randomITensor(a1,a2,a3,a4)

	qd,newα = thick_qr_qdag(A,[a1])
	@test (qd * dag(qd))[1] ≈ 2
	@test qd*dag((prime(qd,"qr"))) |> array ≈ LinearAlgebra.I(dim(a1))
	@test qd*dag((prime(qd,a1))) |> array ≈ LinearAlgebra.I(dim(a1))

	local b1 = a1
	local b2 = a2
	qd,newα = thick_qr_qdag(A,[b1,b2])
	@test (qd * dag(qd))[1] ≈ dim(b1)*dim(b2)
	αqr = findinds(qd,"qr")
	@test qd*dag((prime(qd,"qr"))) ≈ δ(αqr, αqr')
	@test combiner(b1,b2)*qd*dag((prime(qd,b1,b2)))*combiner(b1',b2') |> array ≈ LinearAlgebra.I(dim(b1)*dim(b2))

	local b1 = a2
	local b2 = a4
	qd,newα = thick_qr_qdag(A,[b1,b2])
	@test (qd * dag(qd))[1] ≈ dim(b1)*dim(b2)
	local αqr = findinds(qd,"qr")
	@test qd*dag((prime(qd,"qr"))) ≈ δ(αqr, αqr')
	@test combiner(b1,b2)*qd*dag((prime(qd,b1,b2)))*combiner(b1',b2') |> array ≈ LinearAlgebra.I(dim(b1)*dim(b2))
    end
end

@testset "otrace" begin

    J = [Index(3) for a = 1:4]

    for r = 1:10
        A = randomITensor(J...)
        for sh = 1:10
            K = Random.shuffle(J)
            B = A
            Atr = otrace(A, Index[])
            for a = 1:4
                B = otrace(B, K[2:end])
                K = K[2:end]
            end

            @test B ≈ Atr
        end
    end
end


@testset "dmc_gauge" begin
    σ = Index(2)
    αin = Index(5)
    αout = Index(5)

    for r = 1:Nr
	A = randomITensor(σ,αin,αout)
	qd,newα = dmc_gauge(A,[σ],αout)
	Aq = A*qd
	@test check_dmc(Aq, [σ], newα)
    end
end

@testset "dmt" begin
    αL = Index(10)
    αR = Index(10)
    σ1 = Index(2)
    σ2 = Index(2)

    for r = 1:Nr
	A = randomITensor(αL,σ1,σ2,αR)
	A_keep = otrace(A, [σ1,σ2])
	Linds = [αL,σ1]

	χmax = dim(αL)*dim(σ1)
        Ut,St,Vt = dmt(A, σ1, σ2, [αL], Dict([:maxdim => χmax]))
	@test norm(A - Ut*St*Vt) < 1e-10


	χmax = 1
        Ut,St,Vt = dmt(A, σ1, σ2, [αL], Dict([:maxdim => χmax]))
	Atrunc_keep = otrace(Ut*St*Vt,[σ1, σ2])
	@test norm(A_keep - Atrunc_keep) < 1e-10
    end

    αL = Index(10)
    αR = Index(10)
    σ1 = Index(2)
    σ2 = Index(2)
    σ3 = Index(2)

    for r = 1:Nr
	A = randomITensor(αL,σ1,σ2,σ3,αR)
	A_keep = otrace(A, [σ1,σ2])
	Linds = [αL,σ1]

	χmax = dim(αL)*dim(σ1)
        Ut,St,Vt = dmt(A, σ2, σ3, [αL,σ1], Dict([:maxdim => χmax]))
	@test norm(A - Ut*St*Vt) < 1e-10


	χmax = 1
        Ut,St,Vt = dmt(A, σ2, σ3, [αL,σ1], Dict([:maxdim => χmax]))
	Atrunc_keep = otrace(Ut*St*Vt,[σ1, σ2])
	@test norm(A_keep - Atrunc_keep) < 1e-10
    end
end

@testset "change to hermitian basis" begin
    j = Index(2)
    c = combiner(j,j')
    cind = inds(c)[1]

    Id = ITensor([1 0 0 1], j,j')
    X = ITensor([0 1 1 0], j,j')
    Y = ITensor([0 -im im 0], j,j')
    Z = ITensor([1 0 0 -1], j,j')

    B = [Id,X,Y,Z]

    Bchg,outind = bchg_tensor(B,c)

    @test Bchg*dag(prime(Bchg,cind)) ≈ δ(cind, cind')
    @test Bchg*dag(prime(Bchg,outind)) ≈ δ(outind, outind')
    for jb = 1:length(B)
	@test onehot(outind => jb)*Bchg*c ≈ B[jb]/norm(B[jb])
    end
end

@testset "make doubled gates" begin
    N = 10
    cutoff = 1E-8
    tau = 0.1
    ttotal = 5.0

    s = siteinds("S=1/2", N; conserve_qns=false)
    C = [combiner(s[j], s[j]') for j = 1:N]
    j = 1
    B, sharpind = [ bchg_tensor([op("I", s[j]),
				 op("X", s[j]),
				 op("Y", s[j]),
				 op("Z", s[j])],
				C[j])
		    for j = 1:N ] |> unzip

    for o = ["I", "X", "Y", "Z"]
	local U = exp(-im*op(o,s[1])*0.1)
	local Ud = double(U, C[1:1])
	@test B[1]*Ud*dag(B[1]') |> imag |> norm < 1e-10
    end

    for r = 1:Nr
	H = sum(randn()*op(o, s[1]) for o in ["I", "X", "Y", "Z"])
	local U = exp(-im*H*0.2)
	local Ud = double(U, C[1:1])
	@test B[1]*Ud*dag(B[1]') |> imag |> norm < 1e-10
    end

    j = 1
    s1 = s[j]
    s2 = s[j + 1]
    hj =
	op("Sz", s1) * op("Sz", s2) +
	1 / 2 * op("S+", s1) * op("S-", s2) +
	1 / 2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-im * tau / 2 * hj)

    Gjd = double(Gj,C[1:2])
    Gjdh = B[1]*B[2]*Gjd*dag(B[1]'*B[2]')
    @test Gjdh |> imag |> norm < 1e-10
end

@testset "make dmc" begin
    ssharp = [Index(3) for j = 1:10]
    ψ = randomMPS(Complex{Float64}, ssharp, linkdims=8)
    ψold = deepcopy(ψ)

    nrm = inner(ψ, ψold)
    dmc!(ψ)

    @test check_dmc(ψ,1, quiet=true)
    @test inner(ψ, ψold) ≈ nrm
    sweep_dmc!(ψ,1,3)
    @test check_dmc(ψ,3, quiet=true)
    @test inner(ψ, ψold) ≈ nrm
    sweep_dmc!(ψ,3,5)
    @test check_dmc(ψ,5, quiet=true)
    @test inner(ψ, ψold) ≈ nrm
    sweep_dmc!(ψ,5,1)
    @test check_dmc(ψ,1, quiet=true)
    @test inner(ψ, ψold) ≈ nrm
end

@testset "onsite / nn expectation values" begin
    
    ########################## PRODUCT STATE ##########################
    L = 7
    sites = [Index(4) for j = 1:7]
    
    onsite_tensors1 = reshape(1:(4*7), 4,7) |> collect
    onsite_tensors1[1,:] .= 1

    onsite_tensors2 = reshape(1:(4*7), 4,7) |> collect
    onsite_tensors2[1,:] .= 2

    onsite_tensors3 = reshape(1:(4*7), 4,7) |> collect

    onsite_tensors4 = randn(4,7)
    for onsite_tensors = [onsite_tensors1, onsite_tensors2, onsite_tensors3, onsite_tensors4]
	ψ = MPS([ITensor(onsite_tensors[:,j], sites[j]) for j = 1:7])
	dmc!(ψ);
	nnev = nn_expectation_values(ψ)
        dmc!(ψ)
        oev = onsite_expectation_values(ψ)

	z1 = [prod([onsite_tensors[1,1:j-1] ; onsite_tensors[1,j+1:end] ]) for j = 1:L]
	z2 = [prod([onsite_tensors[1,1:j-1] ; onsite_tensors[1,j+2:end] ]) for j = 1:L-1]
        for j = 1:L
            @test oev[:,j] ≈ onsite_tensors[:,j]*z1[j]
        end
	for j = 1:L-2
	    @test nnev[1,:,j] ≈ nnev[:,1,j+1]
	    @test nnev[:,:,j] ≈ onsite_tensors[:,j] ⊗ onsite_tensors[:,j+1]' .* z2[j]
	end
    end


    ########################## MPO ##########################
    # \[ \sum c X_j + \sum_{j<k} bd yjz_k a^{k-j-1} \]
    
    A = zeros(3,3,4)
    #a = 1
    #b = 1

    a = 2
    b = 3
    c = 5
    d = 7

    A[1,1,1] = 1 # I : final loop
    A[1,2,3] = d # dY : last (left) in longrange
    A[1,3,2] = c # cX : onsite

    A[2,2,1] = a # aI : middle in longrange
    A[2,3,4] = b # bZ : first (right) in longrange
    A[3,3,1] = 1 # I : initial loop

    L = 7
    sites = [Index(4) for j = 1:L]
    links = [Index(3) for j = 1:L-1]

    At = Array{ITensor}(undef, L)
    At[1] = ITensor(A[1,:,:], links[1], sites[1])
    At[L] = ITensor(A[:,3,:], links[L-1], sites[L])
    for j = 2:L-1
        At[j] = ITensor(A, links[j-1], links[j], sites[j])
    end

    ψ = MPS(At)

    dmc!(ψ)


    apriori_oev = zeros(4,L)
    apriori_oev[2,:] .= c
    oev = onsite_expectation_values(ψ)
    dmc!(ψ)

    @test apriori_oev ≈ oev
    
    apriori_nnev = zeros(4,4,L-1)
    apriori_nnev[2,1,:] .= c   # XI
    apriori_nnev[1,2,:] .= c   # IX
    apriori_nnev[3,4,:] .= b*d # YZ
    nnev = nn_expectation_values(ψ)
    @test nnev[:,:,1] ≈ apriori_nnev[:,:,1]
    @test nnev[:,:,2] ≈ apriori_nnev[:,:,2]
    @test nnev[:,:,L-1] ≈ apriori_nnev[:,:,L-1]
    @test nnev ≈ apriori_nnev

    ########################## Ising energy density ##########################

    L = 2
    hx = 1.4
    hz = 0.9045

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
	    hz * op("Z", s1) * op("I", s2) +
	    hz * op("I", s1) * op("Z", s2) +
	    hx * op("X", s1) * op("I", s2) +
	    hx * op("I", s1) * op("X", s2)

        push!(bond_energy_ops, hj)
    end

    Aψ = [onehot(sharpind[j] => 1) for j = 1:L]
    jc = 1
    hjc = bond_energy_ops[jc]*C[jc]*C[jc+1]*B[jc]*B[jc+1]
    @assert imag(hjc) |> norm < 1e-10
    hjcr = real(hjc)

    U,S,V = svd(hjcr, sharpind[jc])

    Aψ[jc] = U*S
    Aψ[jc+1] = V
    ψ = MPS(Aψ)
    oldψT = prod(ψ.data)
    if L >= 3
        @test ( ( oldψT - prod(onehot(sharpind[j]=>1) for j = 3:L) *hjc ) |> norm ) < 1e-10
    else
        @test ( ( oldψT - hjc ) |> norm ) < 1e-10
    end

    orthogonalize!(ψ,1)
    dmc!(ψ)
    @assert check_dmc(ψ,1,quiet=false)
    @test ( (prod(ψ.data) - oldψT) |> norm ) < 1e-10

    println("expectation value:")
    @test onsite_expectation_values(ψ) ≈ 2*[0 0; hx hx; 0 0; hz hz] #factor of 2 from Hilbert space dim in trace
    @test nn_expectation_values(ψ) ≈ 2*[0  hx 0 hz;
				        hx 0  0 0 ;
				        0  0  0 0 ;
				        hz 0  0 1]
end

    


@testset "apply_dmt3_left! notrunc" begin
    L = 5
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                ψ = randomMPS(s, 3)
                G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                  s[j]', s[j+1]', s[j+2]')

                A0 = prod(ψ)*G |> noprime
                apply_dmt3_left!(G,ψ,j,s[j:j+2],Dict([:maxdim=>100]))
                A1 = prod(ψ)
                @test norm( A0 - A1) <= 1e-10
            end
        end
    end
end

@testset "apply_dmt3_left! truncate" begin
    L = 7
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                ψ = randomMPS(s, 20)
                G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                  s[j]', s[j+1]', s[j+2]')

                nnev0 = MPS(prod(ψ)*G |> noprime, s) |> dmc! |> nn_expectation_values
                dmc!(ψ)
                sweep_dmc!(ψ,1,j)
                apply_dmt3_left!(G,ψ,j,s[j:j+2],Dict([:maxdim => 2]))
                #result is not really an MPS. Do the dumb easy thing: roundtrip through just the state tensor.
                nnev1 = MPS(prod(ψ),s) |> dmc! |> nn_expectation_values
                @test norm(nnev0 - nnev1) <= 1e-10
            end
        end
    end
end

@testset "apply_dmt_left! indices / center" begin

    sites = [Index(2) for j = 1:3]

    for r = 1:Nr
        for T = [Float64, Complex{Float64}]

            ψ = randomMPS(T, sites,linkdims=3)
            G = randomITensor(T, sites..., [s' for s in sites]...)
            apply_dmt3_left!(G, ψ, 1, sites, Dict(:maxdim => 1))


            @test (ψ[1] |> inds |> Set) == (sites[1:1] ∪ commoninds(ψ[1],ψ[2]) |> Set)
            @test (ψ[2] |> inds |> Set) == (sites[2:3] ∪ commoninds(ψ[1],ψ[2]) |> Set)
            @test (ψ[3] |> inds |> Set) == Set([])

            @test check_dmc(ψ[1], sites[1:1], commoninds(ψ[1],ψ[2]) |> scalar, quiet=false)
            @test check_dmc(ψ[2], sites[2:2], commoninds(ψ[1],ψ[2]) |> scalar, quiet=false)
        end
    end
end

@testset "apply_dmt3_rght! notrunc" begin
    Nr = 10
    L = 8
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                ψ = randomMPS(s, 3)
                G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                  s[j]', s[j+1]', s[j+2]')

                A0 = prod(ψ)*G |> noprime
                apply_dmt3_rght!(G,ψ,j,s[j:j+2],Dict([:maxdim => 100]))
                A1 = prod(ψ)
                @test norm( A0 - A1) <= 1e-10
            end
        end
    end
end


@testset "apply_dmt3_rght! truncate" begin
    Nr = 10
    L = 5
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                ψ = randomMPS(s, 5)
                G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                  s[j]', s[j+1]', s[j+2]')

                nnev0 = MPS(prod(ψ)*G |> noprime, s) |> dmc! |> nn_expectation_values
                dmc!(ψ)
                sweep_dmc!(ψ,1,j)
                apply_dmt3_rght!(G,ψ,j,s[j:j+2],Dict([:maxdim =>1]))
                # result is not really an MPS, because it's collapsed two of the site tensors into one
                # these are (j+1,j+2)
                # Do the dumb easy thing: roundtrip through just the state tensor.
                nnev1 = MPS(prod(ψ),s) |> dmc! |> nn_expectation_values
                @test norm(nnev0 - nnev1) <= 1e-10
            end
        end
    end
end

@testset "apply_dmt_rght! indices / center" begin

    sites = [Index(2) for j = 1:3]

    for r = 1:Nr
        for T = [Float64, Complex{Float64}]

            ψ = randomMPS(T, sites,linkdims=3)
            G = randomITensor(T, sites..., [s' for s in sites]...)
            apply_dmt3_rght!(G, ψ, 1, sites, Dict(:maxdim => 1))


            @test (ψ[1] |> inds |> Set) == Set([])
            @test (ψ[2] |> inds |> Set) == (sites[1:2] ∪ commoninds(ψ[2],ψ[3]) |> Set)
            @test (ψ[3] |> inds |> Set) == (sites[3:3] ∪ commoninds(ψ[2],ψ[3]) |> Set)
            
            @test check_dmc(ψ[3], sites[3:3], commoninds(ψ[2],ψ[3]) |> scalar, quiet=false)
            @test check_dmc(ψ[2], sites[2:2], commoninds(ψ[2],ψ[3]) |> scalar, quiet=false)
        end
    end
end

@testset "apply_dmt3_both! notrunc" begin
    Nr = 10
    L = 8
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                for center_to = [:l, :r, :c]
                    ψ = randomMPS(s, 5)
                    G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                      s[j]', s[j+1]', s[j+2]')

                    A0 = prod(ψ)*G |> noprime

                    dmc!(ψ)
                    sweep_dmc!(ψ,1,j)

                    apply_dmt3_both!(G,ψ,j,s[j:j+2],Dict(:maxdim => 100),center_to)
                    A1 = prod(ψ)
                    @test norm( A0 - A1) <= 1e-10


                    if     center_to == :l
                        @test check_dmc(ψ, j,quiet=false)
                    elseif center_to == :c
                        @test check_dmc(ψ, j+1,quiet=false)
                    elseif center_to == :r
                        @test check_dmc(ψ, j+2,quiet=false)
                    else error()
                    end
                end
            end
        end
    end
end

@testset "apply_dmt3_both! truncate" begin
    Nr = 10
    L = 5
    s = [Index(2) for j = 1:L]
    for T = [Float64,Complex{Float64}]
        for r = 1:Nr
            for j = 1:L-2
                for center_to = [:l, :r, :c]

                    ψ = randomMPS(s, 5)
                    G = randomITensor(T, s[j] , s[j+1] , s[j+2],
                                      s[j]', s[j+1]', s[j+2]')

                    nnev0 = MPS(prod(ψ)*G |> noprime, s) |> dmc! |> nn_expectation_values
                    dmc!(ψ)
                    sweep_dmc!(ψ,1,j)
                    apply_dmt3_both!(G,ψ,j,s[j:j+2],Dict(:maxdim => 1),center_to)
                    nnev1 = MPS(prod(ψ),s) |> dmc! |> nn_expectation_values
                    @test norm(nnev0 - nnev1) <= 1e-10

                    if     center_to == :l
                        @test check_dmc(ψ, j,quiet=false)
                    elseif center_to == :c
                        @test check_dmc(ψ, j+1,quiet=false)
                    elseif center_to == :r
                        @test check_dmc(ψ, j+2,quiet=false)
                    else error()
                    end

                end
            end
        end
    end
end

@testset "apply_dmt_both! indices / center" begin
    
    sites = [Index(2) for j = 1:3]

    for r = 1:Nr
        for T = [Float64, Complex{Float64}]
            for center_to = [:l,:c,:r]
                ψ = randomMPS(T, sites,linkdims=3)
                G = randomITensor(T, sites..., [s' for s in sites]...)
                apply_dmt3_both!(G, ψ, 1, sites, Dict(:maxdim => 1), center_to)
                @test (ψ[1] |> inds |> Set) == (sites[1:1] ∪ commoninds(ψ[1],ψ[2]) |> Set)
                @test (ψ[2] |> inds |> Set) == (sites[2:2] ∪ commoninds(ψ[1],ψ[2]) ∪ commoninds(ψ[2],ψ[3]) |> Set)
                @test (ψ[3] |> inds |> Set) == (sites[3:3] ∪ commoninds(ψ[2],ψ[3]) |> Set)
                if     center_to == :l @test check_dmc(ψ, 1)
                elseif center_to == :c @test check_dmc(ψ, 2)
                elseif center_to == :r @test check_dmc(ψ, 3)
                end
            end
        end
    end
end
