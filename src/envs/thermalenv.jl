


struct ThermalDMRGEnv{T} 
	mpo::MPO{T}
	rho::PositiveMPA{T}
	hstorage::Vector{Array{T, 3}}
end

function ThermalDMRGEnv(mpo::MPO; D::Int, R::Int)
	@assert QuantumSpins.iphysical_dimensions(mpo) == QuantumSpins.ophysical_dimensions(mpo)
	ds = QuantumSpins.iphysical_dimensions(mpo)
	inject = Injection(ds, R)
	mpo = make_injective(mpo, inject)
	rho = randompmpa(scalar_type(mpo), inject, D=D)
	hstorage = QuantumSpins.init_hstorage_right(mpo, MPS(rho.data))
	return ThermalDMRGEnv(mpo, rho, hstorage)
end

Base.length(x::ThermalDMRGEnv) = length(x.mpo)

