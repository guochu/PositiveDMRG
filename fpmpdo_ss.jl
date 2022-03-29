
include("fpmpdo.jl")

using JSON

function test_fpmpdo(L, gammaphase=0.; D::Int=5)
	J = 1.
	Jzz = 1.5
	hz = 0.

	nl = 1.
	gammal = 1.

	nr = 0.0
	gammar = 1.


	p = spin_half_matrices()

	lindblad1 = boundary_driven_xxz(L, J=J, Jzz=Jzz, hz=hz, nl=nl, Λl=gammal, nr=nr, Λr=gammar, Λp=gammaphase)

	mpo = MPO(lindblad1)

	# dmrg = OpenDMRG(mpo, D=4)
	# return mpo
	return OpenDMRGEnv(mpo, D=D)
end

function nonlocal_dissipative_ising(L; J, hz, Γ, D=4)
	p = spin_half_matrices()
	(Γ >=0) || error("Γ should not be negative.")
	sp, sm, sz = p["+"], p["-"], p["z"]
	lindblad = superoperator(-im * ising_chain(L; J=J, hz=hz))
	for i in 1:L-1
		terms = [QTerm(i=>sp*sm), QTerm(i=>sp, i+1=>sm, coeff=-1), QTerm(i=>sm, i+1=>sp), QTerm(i+1=>sp*sm, coeff=-1)]
		add_dissipation!(lindblad, QuantumOperator(terms) * sqrt(Γ) )
	end
	mpo = MPO(lindblad)
	return OpenDMRGEnv(mpo, D=D)
end

function main_ising()
	# Ls = collect(4:10)
	Ls = vcat(4:12, 15:5:25)

	J = 1.
	hz = 0.1
	Γ = 1.

	ts = Float64[]
	corrs = []

	D = 30
	rank = 30
	alg = FPDMRG(D = D, rank=rank)

	nsweeps = 2
	all_energies = []

	for L in Ls
		println("computing for L=$L")
		dmrg = nonlocal_dissipative_ising(L, J=J, hz=hz, Γ=Γ)
		t = @elapsed energies = run_sweeps_2(dmrg, nsweeps, alg)
		# println(t)
		rho = DensityOperator(dmrg.rho)
		# middle_pos = div(L, 2)

		push!(ts, t)
		push!(all_energies, energies)
		observers = generate_obs(L)
		Js = [expectation(ob, rho) for ob in observers]
		println(Js)
		push!(corrs,  Js)
	end

	results = Dict("Ls"=>Ls, "obs"=>corrs, "energies"=>all_energies, "ts"=>ts)

	open("fig2data/fig2_fpdmrg_D$(D)r$(rank)nsweeps$(nsweeps).json", "w") do f
		write(f, JSON.json(results))
	end

end

function main()
	# Ls = collect(4:10)
	Ls = vcat(3:12, 15:5:25)

	ts = Float64[]
	corrs = []

	D = 30
	rank = 30
	alg = FPDMRG(D = D, rank=rank)

	nsweeps = 2
	all_energies = []

	for L in Ls
		println("computing for L=$L")
		dmrg = test_fpmpdo(L)
		t = @elapsed energies = run_sweeps_2(dmrg, nsweeps, alg)
		# println(t)
		rho = DensityOperator(dmrg.rho)
		# middle_pos = div(L, 2)

		push!(ts, t)
		push!(all_energies, energies)
		observers = generate_obs(L)
		Js = [expectation(ob, rho) for ob in observers]
		println(Js)
		push!(corrs,  Js)

		results = Dict("Ls"=>Ls, "obs"=>corrs, "energies"=>all_energies, "ts"=>ts)

		open("fig2datanew/fig2_fpdmrg_D$(D)r$(rank)nsweeps$(nsweeps).json", "w") do f
			write(f, JSON.json(results))
		end
		
	end


end




