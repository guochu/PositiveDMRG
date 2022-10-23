

struct OpenDMRGEnv
	mpo::MPO{ComplexF64}
	rho::PositiveMPA{ComplexF64}
	hstorage::Vector{Array{ComplexF64, 3}}
end


function OpenDMRGEnv(mpo::MPO; D::Int, R::Int)
	@assert QuantumSpins.iphysical_dimensions(mpo) == QuantumSpins.ophysical_dimensions(mpo)
	ds = safe_sqrt.(QuantumSpins.iphysical_dimensions(mpo))
	inject = Injection(ds, R)
	mpo = make_injective_open(mpo, inject)
	rho = randompmpa(scalar_type(mpo), inject, D=D)
	L = length(mpo)

	hstorage = Vector{Array{ComplexF64, 3}}(undef, L+1)
	hstorage[1] = ones(1,1,1)
	hstorage[L+1] = ones(1,1,1)
	for i in L:-1:2
		rhoj = local_dm(rho, i)
		hstorage[i] = QuantumSpins.updateright(hstorage[i+1], rhoj, mpo[i], rhoj)
	end
	return OpenDMRGEnv(mpo, rho, hstorage)
end

Base.length(x::OpenDMRGEnv) = length(x.mpo)



function center_dm_util(rhoj)
	s1, s2, s3, s4 = size(rhoj)
	@tensor tmp[1,5,2,6,3,7] := rhoj[1,2,3,4] * conj(rhoj[5,6,7,4])
	return reshape(tmp, s1*s1, s2*s2, s3*s3)
end

center_dm(x::PositiveMPA) = center_dm_util(x.mcenter)

function local_dm(x::PositiveMPA, site::Int)
	if site == x.center
		return center_dm(x)
	else
		rhoj = x[site]
		return kron(conj(rhoj), rhoj)
	end
end


function make_injective_open(m::MPO, inject::Injection)
	original_ds = inject.original_ds
	tmp = original_ds .^ 2
	@assert (QuantumSpins.iphysical_dimensions(m) == tmp) && (QuantumSpins.ophysical_dimensions(m) == tmp)

	mleft = _group_mpotensors_open([m[i] for i in 1:inject.left])
	mright = _group_mpotensors_open([m[i] for i in inject.right:length(m)])
	sitetensors = [mleft, [m[i] for i in inject.left+1:inject.right-1]..., mright]
	return MPO(sitetensors)

end

function _group_mpotensors_open(v::Vector)
	m = v[1]
	for i in 2:length(v)
		d1 = safe_sqrt(size(m, 2))
		s1, s2, s3, s4 = size(m)
		m6 = reshape(m, (s1, d1, d1, s3, d1, d1))
		s1, s2, s3, s4 = size(v[i])
		d2 = safe_sqrt(s2)
		v6 = reshape(v[i], (s1, d2, d2, s3, d2, d2))
		@tensor tmp[1,2,7,3,8,9,5,10,6,11] := m6[1,2,3,4,5,6] * v6[4,7,8,9,10,11]
		s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = size(tmp)
		m = reshape(tmp, (s1, s2*s3*s4*s5, s6, s7*s8*s9*s10))
	end
	return m
end


function safe_sqrt(D::Int)
	Dh = round(Int, sqrt(D))
	@assert Dh^2 == D
	return Dh
end