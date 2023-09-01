package cuda

import (
	"math"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func FourierMode(dstReal, dstImag *data.Slice, idx [3]int32, cellsize [3]float64) {

	size := dstReal.Size()

	util.Assert(dstReal.NComp() == 1 && dstImag.NComp() == 1 && dstImag.Size() == size)

	//all components are the

	var f [3]float32
	for c := 0; c < 3; c++ {
		f[c] = 2 * math.Pi * float32(idx[c]) / (float32(cellsize[c]) * float32(size[c]))
	}

	cfg := make3DConf(size)
	k_fourierMode_async(dstReal.DevPtr(0), dstImag.DevPtr(0), f[0], f[1], f[2], size[0], size[1], size[2], cfg)

}
