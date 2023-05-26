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
