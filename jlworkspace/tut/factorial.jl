function factorial(n)
	fac = one(n)
	for i in 2:n
		fac *= i
	end
	return fac
end
