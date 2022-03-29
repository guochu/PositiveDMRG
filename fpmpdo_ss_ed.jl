push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/Meteor.jl/src")


using JSON
using Meteor.Utility
using Meteor.ExactDiag

function run_xxz(L; J, Jzz, h, gammal, nl, gammar, nr, gammaphase)

	model = boundary_driven_xxz(L, J=J, Jzz=Jzz, h=h, gammal=gammal, nl=nl, gammar=gammar, nr=nr, gammaphase=gammaphase, issymmetric=true)

	for i in 1:L
    	add_observer!(model, (i,), ("sz",), name="z")
	end
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
	results = steady_state!(model)

	return results
end


function main()
	Ls = collect(3:12)

	J = 1.
	Jzz = 1.5
	hz = 0.

	nl = 1.
	gammal = 1.

	nr = 0.0
	gammar = 1.

	
	ts = Float64[]	
	corrs = []

	for L in Ls
		println("computing for L=$L")
		t = @elapsed r = run_xxz(L, J=J, Jzz=Jzz, h=hz, gammal=gammal, nl=nl, gammar=gammar, nr=nr, gammaphase=0.)

		obs = r.observables
		Js = [obs["j[$i,$(i+1)]"][1] for i in 1:L-1]

		println(Js)
		push!(corrs, Js)
		push!(ts, t)

		results = Dict("Ls"=>Ls, "obs"=>corrs, "ts"=>ts)

		open("fig2datanew/fig2_ed.json", "w") do f
			write(f, JSON.json(results))
		end

	end


end

# main(paras)
