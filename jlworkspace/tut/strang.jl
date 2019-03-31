function strang(N::Int64)
	A = zeros(Int64, N, N)
	for i in 1:N
		if i > 1
			A[i,i-1] = 1
		end
		A[i,i] = -2
		if i < N
			A[i,i+1] = 1
		end
	end
	return A
end
