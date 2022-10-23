
@with_kw struct PDMRG
	D::Int = 5
	maxiter::Int = 20
	tol::Float64 = 1.0e-6
	verbosity::Int = 1
	R::Int = 20
	mixed_tol::Float64 = 1.0e-8
end

get_trunc(x::PDMRG) = MPSTruncation(D=x.D, ϵ=x.tol)

ThermalDMRG(mpo::MPO; kwargs...) = ThermalDMRGEnv(mpo; kwargs...)
ThermalDMRG(mpo::MPO, alg::PDMRG) = ThermalDMRG(mpo, D=alg.D, R=alg.R)


QuantumSpins.sweep!(m::ThermalDMRGEnv, alg::PDMRG=PDMRG(); β::Real) = vcat(leftsweep!(m, alg, β=β), rightsweep!(m, alg, β=β))

leftsweep!(m::ThermalDMRGEnv, alg::PDMRG=PDMRG(); β::Real) = [left_move!(m, site, β, alg) for site in 1:length(m)-1]
rightsweep!(m::ThermalDMRGEnv, alg::PDMRG=PDMRG(); β::Real) = [right_move!(m, site, β, alg) for site in length(m):-1:2]


function left_move!(m::ThermalDMRGEnv, site::Int, β::Real, alg::PDMRG=PDMRG())
	(alg.verbosity > 2) && println("sweeping from left to right at site: $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	dmj, free_energy = thermalize_center(m, site, β, alg)	
	(alg.verbosity > 2) && println("free energy is $free_energy")
	@tensor twositemps[1,2,5,6,4] := dmj[1,2,3,4] * rho[site+1][3,5,6]
	mpsj, s, r, err = tsvd!(twositemps, (1,2), (3,4,5), trunc=get_trunc(alg))
	(alg.verbosity > 2) && println("svd truncation D: $(size(twositemps, 1) * size(twositemps, 2))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,3,4,5] := sm[1,2] * r[2,3,4,5]
	# rhoj = kron(conj(mpsj), mpsj)
	hstorage[site+1] = QuantumSpins.updateleft(hstorage[site], mpsj, mpo[site], mpsj)
	rho.mcenter = dmj_new
	rho.center = site + 1
	rho[site] = mpsj
	return free_energy
end

function right_move!(m::ThermalDMRGEnv, site::Int, β::Real, alg::PDMRG=PDMRG())
	(alg.verbosity > 2) && println("sweeping from right to left at site: $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	dmj, free_energy = thermalize_center(m, site, β, alg)	
	(alg.verbosity > 2) && println("free energy is $free_energy")
	@tensor twositemps[1,2,4,5,6] := rho[site-1][1,2,3] * dmj[3,4,5,6]
	u, s, mpsj, err = tsvd!(twositemps, (1,2,5), (3,4), trunc=get_trunc(alg))
	(alg.verbosity > 2) && println("svd truncation D: $(size(twositemps, 3) * size(twositemps, 4))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,2,5,3] := u[1,2,3,4] * sm[4,5] 
	# rhoj = kron(conj(mpsj), mpsj)
	hstorage[site] = QuantumSpins.updateright(hstorage[site+1], mpsj, mpo[site], mpsj)
	rho.mcenter = dmj_new
	rho.center = site - 1	
	rho[site] = mpsj
	return free_energy
end

function thermalize_center(m::ThermalDMRGEnv, site::Int, β::Real, alg::PDMRG=PDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	@assert rho.center == site
	n = prod(size(rho.mcenter)[1:3])
	if n <= min(50, 3*alg.R)
		return compute_center(hstorage[site], hstorage[site+1], mpo[site], rho.mcenter, β, alg)
	else
		return compute_center_efficient(hstorage[site], hstorage[site+1], mpo[site], rho.mcenter, β, alg)
	end
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
	eigvalues, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, R=alg.R, verbosity=alg.verbosity)
	D = length(eigvalues)
	r = reshape(eigvectors, s1, s2, s3, D)

	return r, compute_free_energy(hleft, hright, mpoj, r, eigvalues, β)
end

function compute_center_efficient(hleft, hright, mpoj, mpsj, β, alg)
	s1, s2, s3 = size(mpsj)[1:3]
	L = s1 * s2 * s3

	n = min(alg.R, L)
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

	r = reshape(reshape(mpsj_new, L, Dm) * Diagonal(sqrt.(eigvalues)), size(mpsj_new))

	return r, compute_free_energy(hleft, hright, mpoj, r, eigvalues, β)
end

function compute_free_energy(hleft, hright, mpoj, eigvectors, eigvalues, β)
	s1, s2, s3, s4 = size(eigvectors)
	loss = -β * entropy(eigvalues)
	for i in 1:s4
		x = eigvectors[:,:,:,i]
		y = QuantumSpins.ac_prime(x, mpoj, hleft, hright)
		loss += real(dot(x, y))
	end
	return loss
end

function compute_heff(hleft, hright, mpoj)
	@tensor heff[1,4,7,3,6,8] := hleft[1,2,3] * mpoj[2,4,5,6] * hright[7,5,8]
	return heff
end
function eigen_decomp(rhoj; tol::Real=1.0e-10, R::Int=100, verbosity::Int=1)
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
	if length(eigvalues) - pos + 1 >= R
		pos = length(eigvalues) - R + 1
	end
	# err = (pos == 1) ? 0. : eigvalues[pos-1]
	err = (pos == 1) ? 0. : sum(eigvalues[1:pos-1])
	(verbosity > 2) && println("smallest Schmidt number $(eigvalues[1]), largest $(eigvalues[end])")
	(verbosity > 2) && println("mixed truncation D required: $(length(eigvalues) - pos_required + 1), actual: $(length(eigvalues))->$(length(eigvalues) - pos + 1), error: $err")

	eigvalues_r = eigvalues[pos:end]
	eigvectors_r = eigvectors[:, pos:end]

	eigvalues_r ./= sum(eigvalues_r)

	return eigvalues_r, eigvectors_r * Diagonal(sqrt.(eigvalues_r))
end
