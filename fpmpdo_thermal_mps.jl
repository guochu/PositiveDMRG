push!(LOAD_PATH, "/home/guochu/Documents/QuantumSimulator/Meteor.jl/src")


using JSON
using Meteor.Utility
using Meteor.TensorNetwork
using Meteor.Model


function ising_chain(L, J, hz)
	model = spin_chain(L, symm=nothing)
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

function run_ising(L; β, J, hz, dt, D)

	model = ising_chain(L, J, hz)

	results = thermal_state(model, mode=:purified, beta=β, dt=dt, maxbonddimension=D, order=4)

	return results
end

function run_xxz(L; β, J, Jzz, dt, D)

	model = xxz(L, J=J, Jzz=Jzz, h=0.1)

	for i in 1:L-1
    	add_observer!(model, (i, i+1), ("sp", "sm"), name="j")
	end
	
	# results = thermal_state(model, beta=β)

	results = thermal_state(model, mode=:purified, beta=β, dt=dt, maxbonddimension=D, order=4)

	return results
end

function main()

	J = 1.
	# Jzz = 1.
	hz = 1.
	β = 20.

	Ls = vcat(4:14, 20:10:200)
	# Ls = collect(4:9)
	D = 30
	dt = 0.05

	ts = Float64[]
	corrs = []


	for L in Ls
		println("computing for L=$L")
		# t = @elapsed r = run_xxz(L, β=β, J=J, Jzz=Jzz, dt=dt, D=D)
		t = @elapsed r = run_ising(L, β=β, J=J, hz=hz, dt=dt, D=D)
		push!(ts, t)
		obs = r.observables
		Js = [obs["j[$i,$(i+1)]"][end] for i in 1:L-1]
		println(real(Js))
		push!(corrs, Js)
	end

	results = Dict("Ls"=>Ls, "ts"=>ts, "obs"=>corrs)


	open("fig1data/fig1_mps_beta$(β)_D$(D)dt$(dt).json", "w") do f
		write(f, JSON.json(results))
	end
end

# function main_tmp()

# 	J = 1.
# 	# Jzz = 1.
# 	hz = 1.
# 	β = 20.

# 	L = 100
# 	# Ls = collect(4:9)
# 	D = 50
# 	dt = 0.05

# 	ts = Float64[]
# 	corrs = []


# 	println("computing for L=$L")
# 	t = @elapsed r = run_ising(L, β=β, J=J, hz=hz, dt=dt, D=D)
# 	push!(ts, t)
# 	obs = r.observables
# 	Js = [obs["j[$i,$(i+1)]"][end] for i in 1:L-1]

# 	return Js
# end

