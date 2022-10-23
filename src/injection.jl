

struct Injection
	R::Int
	original_ds::Vector{Int}
	ds::Vector{Int}
	left::Int
	right::Int
end

function Injection(ds::Vector{Int}, R::Int)
	@assert (R >= 1) && (R < prod(ds))
	left = 1
	left_pos = 0
	for pos in 1:length(ds)
		left *= ds[pos]
		if left >= R
			left_pos = pos
			break
		end
	end
	right = 1
	right_pos = left_pos + 1
	for pos in length(ds):-1:left_pos+1
		right *= ds[pos]
		if right >= R
			right_pos = pos
			break
		end
	end
	rds = [prod(ds[1:left_pos]), ds[left_pos+1:right_pos-1]..., prod(ds[right_pos:end])]
	return Injection(R, ds, rds, left_pos, right_pos)
end

isinjective(ds::Vector{Int}, R::Int) = (ds[1] >= R && ds[end] >= R)

function make_injective(m::QTerm{T}, inject::Injection) where T
	pos = positions(m)
	ops = QuantumSpins.op(m)
	original_ds = inject.original_ds
	L = length(original_ds)
	new_pos = Int[]
	new_ops = Matrix{T}[]

	left_pos = findlast(x->x<=inject.left, pos)
	if !isnothing(left_pos)
		mat = Matrix(QuantumSpins._prodham_util(original_ds[1:inject.left], Dict(pos[i]=>ops[i] for i in 1:left_pos)))
		push!(new_pos, 1)
		push!(new_ops, mat)
	else
		 left_pos = 0
	end
	right_pos = findfirst(x->x>=inject.right, pos)
	if !isnothing(right_pos)
		for i in left_pos+1:right_pos-1
			push!(new_pos, pos[i] - inject.left + 1 )
			push!(new_ops, ops[i])
		end
		mat = Matrix(QuantumSpins._prodham_util(original_ds[(inject.right):L], Dict((pos[i] - inject.right+1) =>ops[i] for i in right_pos:length(pos))))
		push!(new_pos, length(inject.ds))
		push!(new_ops, mat)
	else
		for i in left_pos+1:length(pos)
			push!(new_pos, pos[i] - inject.left + 1 )
			push!(new_ops, ops[i])
		end		
	end
	return QTerm(new_pos, new_ops, QuantumSpins.coeff(m))
end

make_injective(m::QuantumOperator, inject::Injection) = QuantumOperator([make_injective(t, inject) for t in qterms(m)])

function make_injective(m::MPO, inject::Injection)
	@assert (QuantumSpins.iphysical_dimensions(m) == inject.original_ds) && (QuantumSpins.ophysical_dimensions(m) == inject.original_ds)
	mleft = _group_mpotensors([m[i] for i in 1:inject.left])
	mright = _group_mpotensors([m[i] for i in inject.right:length(m)])
	sitetensors = [mleft, [m[i] for i in inject.left+1:inject.right-1]..., mright]
	return MPO(sitetensors)
end

function make_injective(m::MPS, inject::Injection)
	@assert physical_dimensions(m) == inject.original_ds
	mleft = _group_mpstensors([m[i] for i in 1:inject.left])
	mright = _group_mpstensors([m[i] for i in inject.right:length(m)])	
	sitetensors = [mleft, [m[i] for i in inject.left+1:inject.right-1]..., mright]
	return MPS(sitetensors)
end

function _group_mpotensors(v::Vector)
	m = v[1]
	for i in 2:length(v)
		@tensor tmp[1,2,5,6,4,7] := m[1,2,3,4] * v[i][3,5,6,7]
		s1, s2, s3, s4, s5, s6 = size(tmp)
		m = reshape(tmp, (s1, s2*s3, s4, s5*s6))
	end
	return m
end

function _group_mpstensors(v::Vector)
	m = v[1]
	for i in 2:length(v)
		@tensor tmp[1,2,4,5] := m[1,2,3] * v[i][3,4,5]
		s1, s2, s3, s4 = size(tmp)
		m = reshape(tmp, (s1, s2*s3, s4))
	end
	return m
end




