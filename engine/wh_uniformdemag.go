package engine

func UniformDemag() (B [3][3]float32) {
	//we want to normalise here.

	/*
		msat := Msat.Average()

		B_un := demagConv().UniformDemag2()

		for c := 0; c < 3; c++ {
			for c_ := 0; c_ < 3; c_++ {
				B[c][c_] = B_un[c][c_] * float32(msat)
			}
		}

		return B
	*/

	msat := Msat.MSlice()
	defer msat.Recycle()

	B = demagConv().UniformDemag(msat, geometry.Gpu())
	return B

}
