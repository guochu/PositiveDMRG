

mutable struct PositiveMPA{T<:Number}
	data::Vector{Array{T, 3}}
	mcenter::Array{T, 4}
	center::Int
	inject::Injection
end

Base.length(x::PositiveMPA) = length(x.data)
Base.getindex(x::PositiveMPA, i::Int) = (i == x.center) ? x.mcenter : getindex(x.data, i)
function Base.setindex!(x::PositiveMPA, v, i::Int)
	if i == x.center
		x.mcenter = v
	else
		setindex!(x.data, v, i)
	end
end

function PositiveMPA(mps::MPS; iscanonical::Bool=false, R::Int)
	inject = Injection(physical_dimensions(mps), R)
	mps = make_injective(mps, inject)
	cmps = copy(mps)
	if !iscanonical
		rightorth!(cmps, alg = QRFact())
	end
	center = 1
	return PositiveMPA(cmps.data, reshape(cmps[center], size(cmps[center])..., 1), center, inject )
end

function randompmpa(::Type{T}, inject::Injection; D::Int) where {T <: Number}
	ds = inject.ds
	mps = randommps(T, ds, D=D)
	rightorth!(mps, alg = QRFact())
	center = 1
	mcenter = randn(T, size(mps[center])..., inject.R)
	return PositiveMPA(mps.data, mcenter, center, inject)
end
randompmpa(::Type{T}, ds::Vector{Int}; R::Int, D::Int) where {T <: Number} = randompmpa(T, Injection(ds, R), D=D)

QuantumSpins.scalar_type(x::PositiveMPA{T}) where T = T
function QuantumSpins.tr(x::PositiveMPA)
	mcenter = x.mcenter
	@tensor tmp[1,2,3,5,6,7] := mcenter[1,2,3,4] * conj(mcenter[5,6,7,4])
	L = prod(size(tmp)[1:3])
	return tr(reshape(tmp, (L, L)))
end

QuantumSpins.physical_dimensions(x::PositiveMPA) = x.inject.ds
QuantumSpins.bond_dimensions(x::PositiveMPA) = [size(x[i], 3) for i in 1:length(x)-1]

# function local_dm(x::PositiveMPA, site::Int)
# 	if site == x.center
# 		return center_dm(x)
# 	else
# 		rhoj = x[site]
# 		return kron(conj(rhoj), rhoj)
# 	end
# end


# function center_dm_util(rhoj)
# 	s1, s2, s3, s4 = size(rhoj)
# 	@tensor tmp[1,5,2,6,3,7] := rhoj[1,2,3,4] * conj(rhoj[5,6,7,4])
# 	return reshape(tmp, s1*s1, s2*s2, s3*s3)
# end

# center_dm(x::PositiveMPA) = center_dm_util(x.center)

