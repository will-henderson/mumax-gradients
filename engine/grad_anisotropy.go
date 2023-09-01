package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func addAnisotropyProp(lm, lb *data.Slice) {

	ms := Msat.MSlice()
	defer ms.Recycle()

	if Ku1.nonZero() || Ku2.nonZero() {
		ku1 := Ku1.MSlice()
		defer ku1.Recycle()
		ku2 := Ku2.MSlice()
		defer ku2.Recycle()
		u := AnisU.MSlice()
		defer u.Recycle()

		cuda.AddUniaxialAnisotropy2(lm, lb, ms, ku1, ku2, u)
	}

	if Kc1.nonZero() || Kc2.nonZero() || Kc3.nonZero() {
		ms := Msat.MSlice()
		defer ms.Recycle()

		kc1 := Kc1.MSlice()
		defer kc1.Recycle()

		kc2 := Kc2.MSlice()
		defer kc2.Recycle()

		kc3 := Kc3.MSlice()
		defer kc3.Recycle()

		c1 := AnisC1.MSlice()
		defer c1.Recycle()

		c2 := AnisC2.MSlice()
		defer c2.Recycle()
		cuda.AddCubicAnisotropy2(lm, lb, ms, kc1, kc2, kc3, c1, c2)
	}
}
