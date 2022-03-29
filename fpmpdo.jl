
using LinearAlgebra, Parameters
using KrylovKit
using TensorOperations
using QuantumSpins
import QuantumSpins: expectation, DensityOperator

mutable struct FPMPDO{T}
	data::Vector{Array{T, 3}}
	center::Array{T, 4}
	center_pos::Int
end

Base.length(x::FPMPDO) = length(x.data)
Base.getindex(x::FPMPDO, i::Int) = getindex(x.data, i)
Base.setindex!(x::FPMPDO, v, i::Int) = setindex!(x.data, v, i)

function center_dm_util(rhoj)
	s1, s2, s3, s4 = size(rhoj)
	@tensor tmp[1,5,2,6,3,7] := rhoj[1,2,3,4] * conj(rhoj[5,6,7,4])
	return reshape(tmp, s1*s1, s2*s2, s3*s3)
end

function FPMPDO(data::Vector{Array{T, 3}}) where {T<:Number}
	mps = MPS(copy(data))
	canonicalize!(mps, normalize=true)
	s1, s2, s3 = size(mps[1])
	tmp = reshape(mps[1], s1, s2, s3, 1)
	return FPMPDO(mps[1:length(data)], copy(tmp), 1)
end



center_dm(x::FPMPDO) = center_dm_util(x.center)

function local_dm(x::FPMPDO, site::Int)
	if site == x.center_pos
		return center_dm(x)
	else
		rhoj = x[site]
		return kron(conj(rhoj), rhoj)
	end
end

function DensityOperator(x::FPMPDO)
	ds = [size(x[i], 2) for i in 1:length(x)]
	rho = MPS([local_dm(x, i) for i in 1:length(x)])
	T = scalar_type(rho)
	return DensityOperatorMPS(rho, QuantumSpins.default_fusers(T, ds), QuantumSpins.identity_mps(T, ds))
end

# struct FPExpectCache{T}
# 	rho::FPMPDO{T}
# 	cstorage::Vector{Array{T, 2}}
# end


function update_center_left(hold, center)
	@tensor tmp[6,4] := hold[1,2] * center[2,3,4,5] * conj(center[1,3,6,5])
	return tmp
end

function update_center_left(hold, op, center)
	@tensor tmp[6,4] := hold[1,2] * center[2,3,4,5] * op[7,3] * conj(center[1,7,6,5])
	return tmp
end

# # from left to right
# function FPExpectCache(rho::FPMPDO{T}) where {T<:Number}
# 	(rho.center_pos == 1) || error("center must be 1.")
# 	L = length(rho)
# 	cstorage = Vector{Array{T, 2}}(undef, L+1)
# 	cstorage[1] = ones(1, 1)
# 	cstorage[2] = update_left_first(cstorage[1], rho.center)
# 	for i in 2:L
# 		cstorage[i+1] = QuantumSpins.updateleft(cstorage[i], conj(rho[i]), rho[i])
# 	end
# 	nrm = only(cstorage[L+1])
# 	# println("norm is $nrm")
# 	(abs(nrm - 1.) <= 1.0e-8) || error("not trace preserving.")
# 	return FPExpectCache(rho, cstorage)
# end

function expectation(m::QTerm, rho::FPMPDO)
	# cstorage = env.cstorage
	QuantumSpins.is_zero(m) && return 0.
	L = length(rho)
	pos = positions(m)
	ops = QuantumSpins.op(m)
	pos_end = pos[end]
	pos_begin = min(pos[1], rho.center_pos)
	(pos_end <= L) || throw(BoundsError())
	ss = (pos_begin == rho.center_pos) ? size(rho.center, 1) : size(rho[pos_begin], 1)
	hold = one(zeros(ss, ss))
	for j in pos_begin:max(pos_end, rho.center_pos)
		pj = findfirst(x->x==j, pos)
		if j == rho.center_pos
			if isnothing(pj)
				hold = update_center_left(hold, rho.center)
			else
				hold = update_center_left(hold, ops[pj], rho.center)
			end			
		else
			if isnothing(pj)
				hold = QuantumSpins.updateleft(hold, rho[j], pj, rho[j])
			else
				hold = QuantumSpins.updateleft(hold, rho[j], ops[pj], rho[j])
			end						
		end
	end

	return tr(hold) * QuantumSpins.value(QuantumSpins.coeff(m))
end


# function expectation(m::QTerm, rho::FPMPDO)
# 	# cstorage = env.cstorage
# 	QuantumSpins.is_zero(m) && return 0.
# 	L = length(rho)
# 	pos = positions(m)
# 	ops = QuantumSpins.op(m)
# 	pos_end = pos[end]
# 	(pos_end <= L) || throw(BoundsError())
# 	hold = ones(1, 1)
# 	if pos[1] == 1
# 		hold = update_left_first(hold, ops[1], rho.center)
# 	else
# 		hold = update_left_first(hold, rho.center)
# 	end
# 	for j in 2:pos_end
# 		pj = findfirst(x->x==j, pos)
# 		if isnothing(pj)
# 			hold = QuantumSpins.updateleft(hold, rho[j], pj, rho[j])
# 		else
# 			hold = QuantumSpins.updateleft(hold, rho[j], ops[pj], rho[j])
# 		end
# 	end

# 	return tr(hold) * QuantumSpins.value(QuantumSpins.coeff(m))
# end

@with_kw struct FPDMRG
	D::Int = 5
	maxiter::Int = 20
	tol::Float64 = 1.0e-6
	# verbosity::Int = Defaults.verbosity
	rank::Int = 20
	mixed_tol::Float64 = 1.0e-8
end

get_trunc(x::FPDMRG) = MPSTruncation(D=x.D, ϵ=x.tol)



struct OpenDMRGEnv
	mpo::MPO{ComplexF64}
	rho::FPMPDO{ComplexF64}
	hstorage::Vector{Array{ComplexF64, 3}}
end


function OpenDMRGEnv(mpo::MPO; D::Int)
	L = length(mpo)
	physpaces = [round(Int, sqrt(d)) for d in physical_dimensions(mpo)]
	mps = randommps(ComplexF64, physpaces, D=D)
	rho = FPMPDO(mps[1:L])
	# dm = DensityOperator(rho, trunc=NoTruncation())

	hstorage = Vector{Array{ComplexF64, 3}}(undef, L+1)
	hstorage[1] = ones(1,1,1)
	hstorage[L+1] = ones(1,1,1)
	for i in L:-1:2
		rhoj = local_dm(rho, i)
		hstorage[i] = QuantumSpins.updateright(hstorage[i+1], rhoj, mpo[i], rhoj)
	end
	return OpenDMRGEnv(mpo, rho, hstorage)
end

function eigen_decomp(rhoj; tol::Real=1.0e-10, rank::Int=100)
	eigvalues, eigvectors = eigen(Hermitian(rhoj))
	# println(eigvalues)
	println("Hermitian check: $(maximum(abs.(rhoj - rhoj')))")
	pos = 1
	for i in 1:length(eigvalues) 
		if eigvalues[i] >= tol
			pos = i
			break
		end
	end
	pos_required = pos
	if length(eigvalues) - pos + 1 >= rank
		pos = length(eigvalues) - rank + 1
	end
	# err = (pos == 1) ? 0. : eigvalues[pos-1]
	err = (pos == 1) ? 0. : sum(eigvalues[1:pos-1])
	println("smallest Schmidt number $(eigvalues[1]), largest $(eigvalues[end])")
	println("mixed truncation D required: $(length(eigvalues) - pos_required + 1), actual: $(length(eigvalues))->$(length(eigvalues) - pos + 1), error: $err")

	eigvalues_r = eigvalues[pos:end]
	eigvectors_r = eigvectors[:, pos:end]

	eigvalues_r ./= sum(eigvalues_r)

	return eigvalues_r, eigvectors_r * Diagonal(sqrt.(eigvalues_r))
end

# const EIG_TOL = 1.0e-5

function compute_heff(hleft, hright, mpoj)
	@tensor heff[1,4,7,3,6,8] := hleft[1,2,3] * mpoj[2,4,5,6] * hright[7,5,8]
	return heff
end

function find_abs_smallest(v)
	abs_v = abs.(v)
	pos = argmin(abs_v)
	return v[pos]
end

function minimize_center(m::OpenDMRGEnv, site::Int, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	(rho.center_pos == site) || error("center is wrong!")
	s1, s2, s3 =size(rho.center)[1:3]
	L = s1 * s2 * s3

	# heff = compute_heff(hstorage[site], hstorage[site+1], mpo[site])
	# println("L is $L")
	# println("smallest eigenvalues $(find_abs_smallest(eigvals(reshape(heff, L^2, L^2))))")

	eigvalues, vecs = eigsolve(x->QuantumSpins.ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), center_dm(rho), 1, EigSorter(abs; rev = false), Arnoldi())
	println("Energy after optimization on site $site is $(eigvalues[1]).")
	rhoj = reshape(vecs[1], (s1, s1, s2, s2, s3, s3))
	rhoj2 = reshape(permute(rhoj, (1,3,5,2,4,6)), L, L)
	rhoj2 ./= tr(rhoj2)
	eigvalues_2, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, rank=alg.rank)
	D = length(eigvalues_2)
	# println("number of nonzero Schmidt values $D")
	dmj = reshape(eigvectors, s1, s2, s3, D)
	return eigvalues[1], dmj
end

function left_move!(m::OpenDMRGEnv, site::Int, alg::FPDMRG=FPDMRG())
	println("left to right sweep on site $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	eigvalue, dmj = minimize_center(m, site, alg)	
	@tensor twositemps[1,2,5,6,4] := dmj[1,2,3,4] * rho[site+1][3,5,6]
	mpsj, s, r, err = tsvd!(twositemps, (1,2), (3,4,5), trunc=get_trunc(alg))
	println("svd truncation D: $(size(twositemps, 1) * size(twositemps, 2))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,3,4,5] := sm[1,2] * r[2,3,4,5]
	rhoj = kron(conj(mpsj), mpsj)
	hstorage[site+1] = QuantumSpins.updateleft(hstorage[site], rhoj, mpo[site], rhoj)
	rho[site] = mpsj
	rho.center = dmj_new
	rho.center_pos = site + 1
	return eigvalue
end

function right_move!(m::OpenDMRGEnv, site::Int, alg::FPDMRG=FPDMRG())
	println("right to left sweep on site $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	eigvalue, dmj = minimize_center(m, site, alg)	
	@tensor twositemps[1,2,4,5,6] := rho[site-1][1,2,3] * dmj[3,4,5,6]
	u, s, mpsj, err = tsvd!(twositemps, (1,2,5), (3,4), trunc=get_trunc(alg))
	println("svd truncation D: $(size(twositemps, 3) * size(twositemps, 4))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,2,5,3] := u[1,2,3,4] * sm[4,5] 
	rhoj = kron(conj(mpsj), mpsj)
	hstorage[site] = QuantumSpins.updateright(hstorage[site+1], rhoj, mpo[site], rhoj)
	rho[site] = mpsj
	rho.center = dmj_new
	rho.center_pos = site - 1	
	return eigvalue
end

function leftsweep!(m::OpenDMRGEnv, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage

	Energies = ComplexF64[]
	for site in 1:length(mpo)-1
		eigvalue = left_move!(m, site, alg)
		push!(Energies, eigvalue)
		# (rho.center_pos == site) || error("center is wrong!")
		# s1, s2, s3 =size(rho.center)[1:3]
		# # s1, s2, s3 = size(rho[site])
		# L = s1 * s2 * s3
		# eigvals, vecs = eigsolve(x->QuantumSpins.ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), center_dm(rho), 1, EigSorter(abs; rev = false), Arnoldi())
		# push!(Energies, eigvals[1])
		# println("Energy after optimization on site $site is $(Energies[end]).")
		# rhoj = reshape(vecs[1], (s1, s1, s2, s2, s3, s3))
		# rhoj2 = reshape(permute(rhoj, (1,3,5,2,4,6)), L, L)
		# rhoj2 ./= tr(rhoj2)
		# eigvalues, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, rank = alg.rank)
		# D = length(eigvalues)
		# println("number of nonzero Schmidt values $D")
		# dmj = reshape(eigvectors, s1, s2, s3, D)
		# mpsj, s, r, err = tsvd!(dmj, (1,2), (3,4), trunc=get_trunc(alg))
		# println("svd truncation error is $err")
		# sm = QuantumSpins.diag(s)
		# @tensor dmj_new[0,4,5,3] := sm[0, 1] * r[1,2,3] * rho[site+1][2,4,5]
		# rhoj = kron(conj(mpsj), mpsj)
		# hstorage[site+1] = QuantumSpins.updateleft(hstorage[site], rhoj, mpo[site], rhoj)
		# rho[site] = mpsj
		# rho.center = dmj_new
		# rho.center_pos = site + 1
	end

	return Energies
end


function rightsweep!(m::OpenDMRGEnv, alg::FPDMRG=FPDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage

	Energies = ComplexF64[]
	for site in length(mpo):-1:2
		eigvalue = right_move!(m, site, alg)
		push!(Energies, eigvalue)		
		# (rho.center_pos == site) || error("center is wrong!")
		# s1, s2, s3 =size(rho.center)[1:3]
		# # s1, s2, s3 = size(rho[site])
		# L = s1 * s2 * s3
		# eigvals, vecs = eigsolve(x->QuantumSpins.ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), center_dm(rho), 1, EigSorter(abs; rev = false), Arnoldi())
		# push!(Energies, eigvals[1])
		# println("Energy after optimization on site $site is $(Energies[end]).")
		# rhoj = reshape(vecs[1], (s1, s1, s2, s2, s3, s3))
		# rhoj2 = reshape(permute(rhoj, (1,3,5,2,4,6)), L, L)
		# rhoj2 ./= tr(rhoj2)
		# eigvalues, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, rank = alg.rank)
		# D = length(eigvalues)
		# println("number of nonzero Schmidt values $D")
		# dmj = reshape(eigvectors, s1, s2, s3, D)
		# r, s, mpsj, err = tsvd!(dmj, (1,4), (2,3), trunc=get_trunc(alg))
		# println("svd truncation error is $err")
		# sm = QuantumSpins.diag(s)
		# @tensor dmj_new[1,2,6,4] := rho[site-1][1,2,3] * r[3,4,5] * sm[5, 6]
		# rhoj = kron(conj(mpsj), mpsj)
		# hstorage[site] = QuantumSpins.updateright(hstorage[site+1], rhoj, mpo[site], rhoj)
		# rho[site] = mpsj
		# rho.center = dmj_new
		# rho.center_pos = site - 1
	end

	return Energies	
end

function sweep!(m::OpenDMRGEnv, alg::FPDMRG=FPDMRG())
	return vcat(leftsweep!(m, alg), rightsweep!(m, alg))
end

function run_sweeps(m::OpenDMRGEnv, n::Int, alg::FPDMRG=FPDMRG())
	Energies = ComplexF64[]
	for i in 1:n
		append!(Energies, sweep!(m, alg))
	end
	return Energies
end

function run_sweeps_2(m::OpenDMRGEnv, n::Int, alg::FPDMRG=FPDMRG())
	Energies = ComplexF64[]
	for i in 1:n
		append!(Energies, sweep!(m, alg))
	end
	L = length(m.mpo)
	Lh = div(L, 2)

	for site in 1:Lh-1
		eigvalue = left_move!(m, site, alg)
		push!(Energies, eigvalue)
	end
	rho = m.rho
	eigvalue, dmj = minimize_center(m, Lh, FPDMRG(D=alg.D, rank=10000))	
	rho.center = dmj
	push!(Energies, eigvalue)
	return Energies
end

gamma_plus_minus_from_gamma_n(Γ, n) = Γ*n, Γ*(1-n)

function dissipative_ising(L::Int; J::Real, hz::Real, n::Real, Γ::Real)
	p = spin_half_matrices()
	(n >=0 && n <=1) || error("n must be between 0 and 1.")
	(Γ >=0) || error("Γ should not be negative.")
	sp, sm, sz = p["+"], p["-"], p["z"]
	lindblad = superoperator(-im * ising_chain(L; J=J, hz=hz))
	gamma_plus, gamma_minus = gamma_plus_minus_from_gamma_n(Γ, n)
	for i in 1:L
		add_dissipation!(lindblad, QTerm(1=>sp, coeff=sqrt(gamma_plus)))
		add_dissipation!(lindblad, QTerm(1=>sm, coeff=sqrt(gamma_minus)))		
	end
	return lindblad
end


function generate_pure_obs(L::Int)
	p = spin_half_matrices()
	sp, sm, sz = p["+"], p["-"], p["z"]
	observers = [QTerm(i=>sp, i+1=>sm) for i in 1:L-1]
	return observers
end


function generate_obs(L::Int)
	p = spin_half_matrices()
	sp, sm, sz = p["+"], p["-"], p["z"]
	observers = [QTerm(i=>sp, i+1=>sm) for i in 1:L-1]
	return [superoperator(item, id(item)) for item in observers]
end



function test_fpmpdo_2(L; D::Int=4)
	J = 1.
	hz = 0.

	n = 0.
	Γ = 5.


	p = spin_half_matrices()

	lindblad1 = dissipative_ising(L, J=J, hz=hz, n=n, Γ=Γ)

	mpo = MPO(lindblad1)

	# dmrg = OpenDMRG(mpo, D=4)
	# return mpo
	return OpenDMRGEnv(mpo, D=D)
end



