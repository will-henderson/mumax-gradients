package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func MScale(dst *data.Slice, in MSlice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(in.arr.Len() == N && in.arr.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_scale_async(dst.DevPtr(c), in.DevPtr(c), in.Mul(c), N, cfg)
	}
}

func Scale(dst, src *data.Slice, factor float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_scale_async(dst.DevPtr(c), src.DevPtr(c), factor, N, cfg)
	}
}

func AddMul1D(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == 1 && src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_addMul_async(dst.DevPtr(c), src1.DevPtr(0), src2.DevPtr(c), N, cfg)
	}

}

func Mul1D(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == 1 && src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_mul_async(dst.DevPtr(c), src1.DevPtr(0), src2.DevPtr(c), N, cfg)
	}

}
