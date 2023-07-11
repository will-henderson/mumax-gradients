package cuda

import (
	"github.com/mumax/3/data"
)

// UniformDemag returns a matrix which, for uniform magnetisation, gives the overall demag field.
func (dc *DemagConvolution) UniformDemag(Msat MSlice, vol *data.Slice) (mat [3][3]float32) {

	dc.setUniformInput(Msat, vol)
	dc.fwPlan.ExecAsync(dc.fftRBuf[0], dc.fftCBuf[0])

	if dc.is2D() {
		return dc.uniformDemag2D()
	} else {
		return dc.uniformDemag3D()
	}

}

func (dc *DemagConvolution) setUniformInput(Msat MSlice, vol *data.Slice) {
	zero1_async(dc.fftRBuf[0])
	cfg := make3DConf(dc.inputSize)
	k_unipadmul_async(dc.fftRBuf[0].DevPtr(0), dc.realKernSize[X], dc.realKernSize[Y], dc.realKernSize[Z],
		dc.inputSize[X], dc.inputSize[Y], dc.inputSize[Z],
		Msat.DevPtr(0), Msat.Mul(0), vol.DevPtr(0), cfg)
}

func (dc *DemagConvolution) uniformDemag2D() (mat [3][3]float32) {

	var Nc [4]*data.Slice
	for i := 0; i < 4; i++ {
		Nc[i] = NewSlice(1, fftR2COutputSizeFloats(dc.realKernSize))
	}

	cfg := make3DConf(dc.fftKernLogicSize)
	k_kernmulUniform2D_async(Nc[0].DevPtr(0), Nc[1].DevPtr(0), Nc[2].DevPtr(0), Nc[3].DevPtr(0),
		dc.kern[X][X].DevPtr(0), dc.kern[Y][Y].DevPtr(0), dc.kern[Z][Z].DevPtr(0), dc.kern[X][Y].DevPtr(0),
		dc.fftCBuf[0].DevPtr(0), dc.fftKernLogicSize[X], dc.fftKernLogicSize[Y], dc.fftKernLogicSize[Z], cfg)

	Nr := NewSlice(1, dc.realKernSize)
	Nup := NewSlice(1, dc.inputSize)
	var entry [4]float32
	for i := 0; i < 4; i++ {
		dc.bwPlan.ExecAsync(Nc[i], Nr)
		Nc[i].Free()

		copyUnPad(Nup, Nr, dc.inputSize, dc.realKernSize)

		entry[i] = Sum(Nup)
	}

	Nr.Free()
	Nup.Free()

	mat[X][X] = entry[0]
	mat[Y][Y] = entry[1]
	mat[Z][Z] = entry[2]
	mat[X][Y] = entry[3]

	mat[Y][X] = mat[X][Y]

	return mat
}

func (dc *DemagConvolution) uniformDemag3D() (mat [3][3]float32) {

	var Nc [6]*data.Slice
	for i := 0; i < 6; i++ {
		Nc[i] = NewSlice(1, fftR2COutputSizeFloats(dc.realKernSize))
	}

	cfg := make3DConf(dc.fftKernLogicSize)

	k_kernmulUniform3D_async(Nc[0].DevPtr(0), Nc[1].DevPtr(0), Nc[2].DevPtr(0),
		Nc[3].DevPtr(0), Nc[4].DevPtr(0), Nc[5].DevPtr(0),
		dc.kern[X][X].DevPtr(0), dc.kern[Y][Y].DevPtr(0), dc.kern[Z][Z].DevPtr(0),
		dc.kern[Y][Z].DevPtr(0), dc.kern[X][Z].DevPtr(0), dc.kern[X][Y].DevPtr(0),
		dc.fftCBuf[0].DevPtr(0),
		dc.fftKernLogicSize[X], dc.fftKernLogicSize[Y], dc.fftKernLogicSize[Z], cfg)

	Nr := NewSlice(1, dc.realKernSize)
	Nup := NewSlice(1, dc.inputSize)
	var entry [6]float32
	for i := 0; i < 6; i++ {
		dc.bwPlan.ExecAsync(Nc[i], Nr)
		Nc[i].Free()

		copyUnPad(Nup, Nr, dc.inputSize, dc.realKernSize)

		entry[i] = Sum(Nup)
	}

	Nr.Free()
	Nup.Free()

	mat[X][X] = entry[0]
	mat[Y][Y] = entry[1]
	mat[Z][Z] = entry[2]
	mat[Y][Z] = entry[3]
	mat[X][Z] = entry[4]
	mat[X][Y] = entry[5]

	mat[Y][X] = mat[X][Y]
	mat[Z][X] = mat[X][Z]
	mat[Z][Y] = mat[Y][Z]

	return mat
}
