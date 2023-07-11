package engine

func ExchangeAtCell(i, j, k, i_, j_, k_ int) float32 {
	if !lex2.cpu_ok {
		lex2.update()
	}

	// do out of range checks in here.
	if i < 0 || j < 0 || k < 0 || i >= MeshSize()[0] || j >= MeshSize()[1] || k >= MeshSize()[2] {
		return 0
	}

	r1 := regions.GetCell(i, j, k)
	r2 := regions.GetCell(i, j, k)

	return lex2.lut[symmidx(r1, r2)]
}
