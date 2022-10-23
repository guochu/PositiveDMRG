
OpenDMRG(mpo::MPO; kwargs...) = OpenDMRGEnv(mpo; kwargs...)
OpenDMRG(mpo::MPO, alg::PDMRG) = OpenDMRG(mpo, D=alg.D, R=alg.R)

QuantumSpins.sweep!(m::OpenDMRGEnv, alg::PDMRG=PDMRG()) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

leftsweep!(m::OpenDMRGEnv, alg::PDMRG=PDMRG()) = [left_move!(m, site, alg) for site in 1:length(m)-1]
rightsweep!(m::OpenDMRGEnv, alg::PDMRG=PDMRG()) = [right_move!(m, site, alg) for site in length(m):-1:2]


function minimize_center(m::OpenDMRGEnv, site::Int, alg::PDMRG=PDMRG())
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	@assert rho.center == site
	s1, s2, s3 =size(rho[site])[1:3]
	L = s1 * s2 * s3


	eigvalues, vecs = eigsolve(x->QuantumSpins.ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), center_dm(rho), 1, EigSorter(abs; rev = false), Arnoldi())
	(alg.verbosity > 2) && println("Energy after optimization on site $site is $(eigvalues[1]).")
	rhoj = reshape(vecs[1], (s1, s1, s2, s2, s3, s3))
	rhoj2 = reshape(permute(rhoj, (1,3,5,2,4,6)), L, L)
	rhoj2 ./= tr(rhoj2)
	eigvalues_2, eigvectors = eigen_decomp(rhoj2, tol=alg.mixed_tol, R=alg.R)
	D = length(eigvalues_2)
	# println("number of nonzero Schmidt values $D")
	dmj = reshape(eigvectors, s1, s2, s3, D)
	return eigvalues[1], dmj
end

function left_move!(m::OpenDMRGEnv, site::Int, alg::PDMRG=PDMRG())
	(alg.verbosity > 2) && println("sweeping from left to right at site: $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	eigvalue, dmj = minimize_center(m, site, alg)	
	@tensor twositemps[1,2,5,6,4] := dmj[1,2,3,4] * rho[site+1][3,5,6]
	mpsj, s, r, err = tsvd!(twositemps, (1,2), (3,4,5), trunc=get_trunc(alg))
	(alg.verbosity > 2) && println("svd truncation D: $(size(twositemps, 1) * size(twositemps, 2))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,3,4,5] := sm[1,2] * r[2,3,4,5]
	rhoj = kron(conj(mpsj), mpsj)
	hstorage[site+1] = QuantumSpins.updateleft(hstorage[site], rhoj, mpo[site], rhoj)
	rho.mcenter = dmj_new
	rho.center = site + 1
	rho[site] = mpsj
	return eigvalue
end

function right_move!(m::OpenDMRGEnv, site::Int, alg::PDMRG=PDMRG())
	(alg.verbosity > 2) && println("sweeping from right to left at site: $site")
	mpo = m.mpo
	rho = m.rho
	hstorage = m.hstorage
	eigvalue, dmj = minimize_center(m, site, alg)	
	@tensor twositemps[1,2,4,5,6] := rho[site-1][1,2,3] * dmj[3,4,5,6]
	u, s, mpsj, err = tsvd!(twositemps, (1,2,5), (3,4), trunc=get_trunc(alg))
	(alg.verbosity > 2) && println("svd truncation D: $(size(twositemps, 3) * size(twositemps, 4))->$(length(s)), error: $err")
	sm = QuantumSpins.diag(s)
	@tensor dmj_new[1,2,5,3] := u[1,2,3,4] * sm[4,5] 
	rhoj = kron(conj(mpsj), mpsj)
	hstorage[site] = QuantumSpins.updateright(hstorage[site+1], rhoj, mpo[site], rhoj)
	rho.mcenter = dmj_new
	rho.center = site - 1	
	rho[site] = mpsj
	return eigvalue
end

