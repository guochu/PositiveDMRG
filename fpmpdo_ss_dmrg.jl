push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/Meteor.jl/src")


using JSON
using Meteor.Utility
using Meteor.TensorNetwork
using Meteor.Model

function run_xxz(L; J, Jzz, h, gammal, nl, gammar, nr, gammaphase, D, nsweeps)

	model = boundary_driven_xxz(L, J=J, Jzz=Jzz, h=h, gammal=gammal, nl=nl, gammar=gammar, nr=nr, gammaphase=gammaphase, symm=nothing)

	# for i in 1:L
 #    	add_observer!(model, (i,), ("sz",), name="z")
	# end
	for i in 1:L-1
    	add_observer!(model, (i, i+1), ("sp", "sm"), name="j")
	end

	initial_state = [0 for i in 1:L]
	for i in 1:2:L
    	initial_state[i] = 1
	end
    product_state!(model, initial_state)

 #    basis = generate_basis(model.h, model.state)
 #    ham = immrep(basis, model.h)
 #    observer = immrep(basis, model.observer)

 #    @time delta = relaxation_time(ham, model.state)

	# println("relaxation time is $delta.")

	observer = model.observer

    mpo = tompo(model.h)
    mpo = mpo' * mpo
    compress!(mpo)

    dmrg = DMRG(mpo, maxbonddimension=D, subexp=0.)

    energies = dosweeps!(dmrg, nsweeps, single_site=true)

    state = dmrg.mps
    renormalize_open!(state, identity_state.(particles(model)))

	return Observables(measure(observer, state)), energies
end


function main()
	# Ls = collect(4:12)
	Ls = vcat(3:12, 15:5:25)

	J = 1.
	Jzz = 1.5
	hz = 0.

	nl = 1.
	gammal = 1.

	nr = 0.0
	gammar = 1.


	D = 25
	nsweeps = 5000


	ts = Float64[]	
	corrs = []

	all_energies = []

	for L in Ls
		println("computing for L=$L")
		t = @elapsed obs, energies = run_xxz(L, J=J, Jzz=Jzz, h=hz, gammal=gammal, nl=nl, gammar=gammar, nr=nr, gammaphase=0., D=D, nsweeps=nsweeps)

		Js = [obs["j[$i,$(i+1)]"][1] for i in 1:L-1]
		println(Js)

		push!(ts, t)
		push!(corrs, Js)
		push!(all_energies, energies)

		results = Dict("Ls"=>Ls, "obs"=>corrs, "energies"=>all_energies, "ts"=>ts)

		open("fig2datanew/fig2_dmrgD$(D)nsweeps$(nsweeps).json", "w") do f
			write(f, JSON.json(results))
		end

	end


end

# main(paras)
