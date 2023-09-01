package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func demagProp(lm, lb *data.Slice) {
	if EnableDemag {
		msat := Msat.MSlice()
		defer msat.Recycle()
		if NoDemagSpins.isZero() {
			// Normal demag, everywhere
			demagConv().Exec(lm, lb, geometry.Gpu(), msat)
		} else {
			panic("fuck haven't implemented setMaskedDemagField yet")

		}
	} else {
		cuda.Zero(lm) // will ADD other terms to it
	}
}
