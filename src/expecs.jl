

QuantumSpins.expectation(m::QTerm, rho::PositiveMPA) = _expectation(make_injective(m, rho.inject), rho)
QuantumSpins.expectation(m::QuantumOperator, rho::PositiveMPA) = _expectation(make_injective(m, rho.inject), rho)

function _expectation(m::QTerm, rho::PositiveMPA)
	# cstorage = env.cstorage
	QuantumSpins.is_zero(m) && return 0.
	L = length(rho)
	pos = positions(m)
	ops = QuantumSpins.op(m)
	pos_end = pos[end]
	pos_begin = min(pos[1], rho.center)
	(pos_end <= L) || throw(BoundsError())
	ss = size(rho[pos_begin], 1)
	hold = one(zeros(ss, ss))
	for j in pos_begin:max(pos_end, rho.center)
		pj = findfirst(x->x==j, pos)
		if j == rho.center
			if isnothing(pj)
				hold = update_center_left(hold, rho[j])
			else
				hold = update_center_left(hold, ops[pj], rho[j])
			end			
		else
			if isnothing(pj)
				hold = QuantumSpins.updateleft(hold, rho[j], pj, rho[j])
			else
				hold = QuantumSpins.updateleft(hold, rho[j], ops[pj], rho[j])
			end						
		end
	end

	return tr(hold) * QuantumSpins.value(QuantumSpins.coeff(m))
end

function _expectation(m::QuantumOperator, rho::PositiveMPA)
	r = 0.
	for t in qterms(m)
		r += _expectation(t, rho)
	end
	return r
end


function update_center_left(hold, center)
	@tensor tmp[6,4] := hold[1,2] * center[2,3,4,5] * conj(center[1,3,6,5])
	return tmp
end

function update_center_left(hold, op, center)
	@tensor tmp[6,4] := hold[1,2] * center[2,3,4,5] * op[7,3] * conj(center[1,7,6,5])
	return tmp
end

