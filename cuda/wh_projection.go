package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func InitRotation(gsMag *data.Slice, R [3][3]*data.Slice) {
	N := gsMag.Len()

	for c := 0; c < 3; c++ {
		for c_ := 0; c_ < 3; c_++ {
			util.Assert(R[c][c_].NComp() == 1 && R[c][c_].Len() == N)
		}
	}

	cfg := make1DConf(N)
	k_initRotation_async(gsMag.DevPtr(X), gsMag.DevPtr(Y), gsMag.DevPtr(Z),
		R[X][X].DevPtr(0), R[X][Y].DevPtr(0), R[X][Z].DevPtr(0),
		R[Y][X].DevPtr(0), R[Y][Y].DevPtr(0), R[Y][Z].DevPtr(0),
		R[Z][X].DevPtr(0), R[Z][Y].DevPtr(0), R[Z][Z].DevPtr(0),
		N, cfg)
}

func RotateMode(dst *data.Slice, mode *data.Slice, R [3][3]*data.Slice) {
	N := mode.Len()

	util.Assert(dst.NComp() == 2 && mode.NComp() == 3 && dst.Len() == N)

	cfg := make1DConf(N)
	k_rotateMode_async(dst.DevPtr(X), dst.DevPtr(Y),
		mode.DevPtr(X), mode.DevPtr(Y), mode.DevPtr(Z),
		R[X][X].DevPtr(0), R[X][Y].DevPtr(0), R[X][Z].DevPtr(0),
		R[Y][X].DevPtr(0), R[Y][Y].DevPtr(0), R[Y][Z].DevPtr(0),
		N, cfg)
}

func DerotateMode(dst *data.Slice, mode *data.Slice, R [3][3]*data.Slice) {
	N := mode.Len()

	util.Assert(dst.NComp() == 3 && mode.NComp() == 2 && dst.Len() == N)

	cfg := make1DConf(N)
	k_derotateMode_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		mode.DevPtr(X), mode.DevPtr(Y),
		R[X][X].DevPtr(0), R[X][Y].DevPtr(0), R[X][Z].DevPtr(0),
		R[Y][X].DevPtr(0), R[Y][Y].DevPtr(0), R[Y][Z].DevPtr(0),
		N, cfg)
}
