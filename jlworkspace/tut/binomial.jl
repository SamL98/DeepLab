function binomial_rv(n, p)
	n = 0
	bin = rand(n)
	for i in 1:n
		if bin[i] >= p
			n += 1
		end
	end
	return n
end
