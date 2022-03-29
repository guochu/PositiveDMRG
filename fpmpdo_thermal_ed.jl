push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/Meteor.jl/src")


using JSON
using Meteor.Utility
using Meteor.ExactDiag


function ising_chain(L, J, hz)
	model = spin_chain(L, issymmetric=false)
	for i in 1:L
		add!(model, (i,), ("sz",), coeff=hz)
	end
	for i in 1:L-1
		add!(model, (i, i+1), ("sx", "sx"), coeff=J)
	end
	# for i in 1:L
 #    	add_observer!(model, (i,), ("sz",), name="z")
	# end
	for i in 1:L-1
    	add_observer!(model, (i, i+1), ("sp", "sm"), name="j")
	end
	return model
end

function run_ising(L; β, J, hz)

	model = ising_chain(L, J, hz)

	results = thermal_state(model, beta=β)

	return results
end

function run_xxz(L; β, J, Jzz)

	model = xxz(L, J=J, Jzz=Jzz, h=0.1)

	for i in 1:L-1
    	add_observer!(model, (i, i+1), ("sp", "sm"), name="j")
	end	

	results = thermal_state(model, beta=β)

	return results
end

function main()

	J = 1.
	# Jzz = 1.
	hz = 1.
	β = 10.

	Ls = collect(4:14)
	ts = Float64[]
	corrs = []


	for L in Ls
		println("computing for L=$L")
		t = @elapsed r = run_ising(L, β=β, J=J, hz=hz)
		push!(ts, t)
		obs = r.observables
		Js = [obs["j[$i,$(i+1)]"][1] for i in 1:L-1]
		println(Js)
		push!(corrs, Js)
	end

	results = Dict("Ls"=>Ls, "ts"=>ts, "obs"=>corrs)

	open("fig1data/fig1_ed_beta$(β).json", "w") do f
		write(f, JSON.json(results))
	end

end

