package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func TensorFieldFactor(dst, ms *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(ms.Len() == N && ms.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_tensorFieldFactor_async(dst.DevPtr(c), ms.DevPtr(c), N, cfg)
	}
}
