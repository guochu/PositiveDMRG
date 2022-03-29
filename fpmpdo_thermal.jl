
include("fpmpdo.jl")
using JSON


struct ThermalDMRGEnv{T}
	mpo::MPO{T}
	rho::FPMPDO{T}
	hstorage::Vector{Array{T, 3}}
end


function ThermalDMRGEnv(mpo::MPO; D::Int)
	L = length(mpo)
	physpaces = physical_dimensions(mpo)
	mps = randommps(scalar_type(mpo), physpaces, D=D)
	rho = FPMPDO(mps[1:L])

	hstorage = QuantumSpins.init_hstorage_right(mpo, MPS(rho.data))
	return ThermalDMRGEnv(mpo, rho, hstorage)
end



function stable_exp(heff, β)
	rho = exp(-heff)
	rho ./= tr(rho)
	rho_out = rho^β
	rho_out ./= tr(rho_out)
	return rho_out
end

function compute_center(hleft, hright, mpoj, mpsj, β, alg)
	s1, s2, s3 =size(mpsj)[1:3]
	L = s1 * s2 * s3
	heff = compute_heff(hleft, hright, mpoj)
	rhoj2 = stable_exp(reshape(heff, L, L), β)
	eigvalues, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, rank=alg.rank)
	D = length(eigvalues)
	return reshape(eigvectors, s1, s2, s3, D)
end

function compute_center_efficient(hleft, hright, mpoj, mpsj, β, alg)
	s1, s2, s3 = size(mpsj)[1:3]
	L = s1 * s2 * s3

	n = min(alg.rank, L)
	eigvalues, vecs = eigsolve(x->QuantumSpins.ac_prime(x, mpoj, hleft, hright), mpsj[:,:,:,1], n, :SR, Lanczos(krylovdim=2*n+2))
	# println("eigenvalues are $(eigvalues)")
	eigvalues = [exp(-item) for item in eigvalues]
	eigvalues ./= sum(eigvalues)
	Dm = length(eigvalues)
	for i in 1:Dm
		eigvalues[i] = (eigvalues[i])^β
	end
	eigvalues ./= sum(eigvalues)

	mpsj_new = similar(mpsj, (s1, s2, s3, Dm))
	for i in 1:Dm
		mpsj_new[:,:,:,i] = vecs[i]
	end

	return reshape(reshape(mpsj_new, L, Dm) * Diagonal(sqrt.(eigvalues)), size(mpsj_new))
end

function thermalize_center(m::ThermalDMRGEnv, site::Int, β::Real, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	(rho.center_pos == site) || error("center is wrong!")

	n = prod(size(rho.center)[1:3])
	if n <= min(50, 3*alg.rank)
		return compute_center(hstorage[site], hstorage[site+1], mpo[site], rho.center, β, alg)
	else
		return compute_center_efficient(hstorage[site], hstorage[site+1], mpo[site], rho.center, β, alg)
	end

	# return compute_center(hstorage[site], hstorage[site+1], mpo[site], rho.center, β, alg)
end

function left_move!(m::ThermalDMRGEnv, site::Int, β::Real, alg::FPDMRG=FPDMRG())
	println("left to right sweep on site $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	dmj = thermalize_center(m, site, β, alg)	
	@tensor twositemps[1,2,5,6,4] := dmj[1,2,3,4] * rho[site+1][3,5,6]
	mpsj, s, r, err = tsvd!(twositemps, (1,2), (3,4,5), trunc=get_trunc(alg))
	println("svd truncation D: $(size(twositemps, 1) * size(twositemps, 2))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,3,4,5] := sm[1,2] * r[2,3,4,5]
	# rhoj = kron(conj(mpsj), mpsj)
	hstorage[site+1] = QuantumSpins.updateleft(hstorage[site], mpsj, mpo[site], mpsj)
	rho[site] = mpsj
	rho.center = dmj_new
	rho.center_pos = site + 1
end

function right_move!(m::ThermalDMRGEnv, site::Int, β::Real, alg::FPDMRG=FPDMRG())
	println("right to left sweep on site $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	dmj = thermalize_center(m, site, β, alg)	
	@tensor twositemps[1,2,4,5,6] := rho[site-1][1,2,3] * dmj[3,4,5,6]
	u, s, mpsj, err = tsvd!(twositemps, (1,2,5), (3,4), trunc=get_trunc(alg))
	println("svd truncation D: $(size(twositemps, 3) * size(twositemps, 4))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,2,5,3] := u[1,2,3,4] * sm[4,5] 
	# rhoj = kron(conj(mpsj), mpsj)
	hstorage[site] = QuantumSpins.updateright(hstorage[site+1], mpsj, mpo[site], mpsj)
	rho[site] = mpsj
	rho.center = dmj_new
	rho.center_pos = site - 1	
end


function leftsweep!(m::ThermalDMRGEnv, β::Real, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage

	for site in 1:length(mpo)-1
		left_move!(m, site, β, alg)
	end
end


function rightsweep!(m::ThermalDMRGEnv, β::Real, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage

	for site in length(mpo):-1:2
		right_move!(m, site, β, alg)
	end
end

function sweep!(m::ThermalDMRGEnv, β::Real, alg::FPDMRG=FPDMRG())
	leftsweep!(m, β, alg)
	rightsweep!(m, β, alg)
end

function run_sweeps(m::ThermalDMRGEnv, n::Int, β::Real, alg::FPDMRG=FPDMRG())
	for i in 1:n
		sweep!(m, β, alg)
	end
end

function run_sweeps_2(m::ThermalDMRGEnv, n::Int, β::Real, alg::FPDMRG=FPDMRG())
	for i in 1:n
		sweep!(m, β, alg)
	end
	L = length(m.mpo)
	Lh = div(L, 2)

	for site in 1:Lh-1
		left_move!(m, site, β, alg)
	end
	rho = m.rho
	dmj = thermalize_center(m, Lh, β, alg)
	rho.center = dmj
end

function ising_chain_2(L::Int; J::Real, hz::Real)
	p = spin_half_matrices()
    x, z = p["x"], p["z"]	
    I2 = one(zeros(2,2))
    z = (z + I2) / 2
    terms = []
    for i in 1:L
    	push!(terms, QTerm(i=>z, coeff=hz))
    end
    for i in 1:L-1
    	push!(terms, QTerm(i=>x, i+1=>x, coeff=J))
    end
    return QuantumOperator([terms...])
end


function test_fpmpdo_thermal(L; D::Int=4)
	J = 1.
	hz = 1.

	ham = ising_chain(L, J=J, hz=hz)

	mpo = MPO(ham)

	# dmrg = OpenDMRG(mpo, D=4)
	# return mpo
	return ThermalDMRGEnv(mpo, D=D)
end


function test_fpmpdo_thermal_xxx(L; D::Int=4)
	J = 1.
	Jzz = 1.

	ham = heisenberg_chain(L, J=J, Jzz=1., hz=0.1)

	mpo = MPO(ham)

	# dmrg = OpenDMRG(mpo, D=4)
	# return mpo
	return ThermalDMRGEnv(mpo, D=D)
end


function main()
	Ls = vcat(4:14, 20:10:200)
	# Ls = collect(4:9)
	ts = Float64[]
	corrs = []

	p = spin_half_matrices()
	sp, sm, sz = p["+"], p["-"], p["z"]

	β = 10.
	D = 30
	rank = 50
	alg = FPDMRG(D = D, rank=rank)
	nsweeps = 2

	for L in Ls
		println("computing for L=$L")
		dmrg = test_fpmpdo_thermal(L)
		t = @elapsed run_sweeps_2(dmrg, nsweeps, β, alg)
		# println(t)
		push!(ts, t)
		# rho = DensityOperator(dmrg.rho)
		rho = dmrg.rho
		# middle_pos = div(L, 2)

		# observers = generate_obs(L)
		observers = generate_pure_obs(L)
		push!(corrs, [expectation(ob, rho) for ob in observers] )
	end

	results = Dict("Ls"=>Ls, "ts"=>ts, "obs"=>corrs)

	open("fig1data/fig1_fpdmrg_beta$(β)_D$(D)r$(rank)nsweeps$(nsweeps).json", "w") do f
		write(f, JSON.json(results))
	end
end

# function main_tmp()
# 	L = 10
# 	# Ls = collect(4:9)
# 	ts = Float64[]
# 	corrs = []


# 	β = 10.
# 	D = 30
# 	rank = 30
# 	alg = FPDMRG(D = D, rank=rank)
# 	nsweeps = 2

# 	println("computing for L=$L")
# 	dmrg = test_fpmpdo_thermal(L)
# 	t = @elapsed run_sweeps_2(dmrg, nsweeps, β, alg)
# 	# println(t)
# 	push!(ts, t)
# 	# rho = DensityOperator(dmrg.rho)
# 	rho = dmrg.rho

# 	observers = generate_pure_obs(L)

# 	return [expectation(ob, rho) for ob in observers]
# end




