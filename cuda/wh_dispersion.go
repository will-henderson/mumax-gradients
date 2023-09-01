package cuda

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/cufft"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

const SIZEOF_INT32 int64 = 4

type DispersionFFT struct {
	inputSize       [3]int
	fftLogicSize    [3]int
	nSamples        int
	samplesPtrs     [3]unsafe.Pointer
	askedToComputed []int
	fftPlan         fft3DR2CPlan
	fftCBuf         *data.Slice
	t               int
	timeSteps       int
	SpaceTrans      *data.Slice
}

func NewDispersionFFT(inputSize [3]int, samples [3][]int32, timeSteps int) (dispFFT *DispersionFFT) {
	dispFFT = new(DispersionFFT)
	dispFFT.inputSize = inputSize
	dispFFT.fftLogicSize = fftR2COutputSizeFloats(inputSize)
	dispFFT.fftCBuf = NewSlice(3, dispFFT.fftLogicSize)

	dispFFT.setSamples(samples)

	dispFFT.fftPlan = newFFT3DR2C(inputSize[X], inputSize[Y], inputSize[Z])

	dispFFT.timeSteps = timeSteps
	dispFFT.SpaceTrans = NewSlice(3, [3]int{2 * dispFFT.nSamples, timeSteps, 1}) //2* nSamples because complex comps

	return dispFFT
}

func (dc *DispersionFFT) setSamples(samples [3][]int32) {

	nAsked := len(samples[0])
	util.Assert(nAsked == len(samples[1]) && nAsked == len(samples[2]))

	size := dc.inputSize

	sampleMesh := make([][][]int, dc.inputSize[X]/2+1)
	for i := 0; i < size[X]/2+1; i++ {
		sampleMesh[i] = make([][]int, dc.inputSize[Y])
		for j := 0; j < size[Y]; j++ {
			sampleMesh[i][j] = make([]int, dc.inputSize[Z])
		}
	}

	for i := 0; i < nAsked; i++ {
		ix, iy, iz := kRequestedToMeshIdx(samples[0][i], samples[1][i], samples[2][i], dc.inputSize)
		sampleMesh[ix][iy][iz] = 1
	}

	var checkedSamples [3][]int32
	csCount := 0
	for i := 0; i < size[X]/2+1; i++ {
		for j := 0; j < size[Y]; j++ {
			for k := 0; k < size[Z]; k++ {
				if sampleMesh[i][j][k] > 0 {
					checkedSamples[0] = append(checkedSamples[0], int32(i))
					checkedSamples[1] = append(checkedSamples[1], int32(j))
					checkedSamples[2] = append(checkedSamples[2], int32(k))

					sampleMesh[i][j][k] = csCount
					csCount++
				}
			}
		}
	}
	dc.nSamples = len(checkedSamples[0])

	//store array telling how asked maps to computed
	dc.askedToComputed = make([]int, nAsked)
	for i := 0; i < nAsked; i++ {
		ix, iy, iz := kRequestedToMeshIdx(samples[0][i], samples[1][i], samples[2][i], dc.inputSize)
		dc.askedToComputed[i] = sampleMesh[ix][iy][iz]
	}

	for c := 0; c < 3; c++ {
		bytes := int64(dc.nSamples) * SIZEOF_INT32
		dc.samplesPtrs[c] = MemAlloc(bytes)
		MemCpyHtoD(dc.samplesPtrs[c], unsafe.Pointer(&checkedSamples[c][0]), bytes)
	}

}

func kRequestedToMeshIdx(px, py, pz int32, size [3]int) (ix, iy, iz int32) {

	iy = mod(py, int32(size[Y]))
	iz = mod(pz, int32(size[Z]))

	ix = mod(px, int32(size[X]))
	if ix > int32(size[X]/2) {
		ix = int32(size[X]) - ix
	}

	return ix, iy, iz
}

func mod(a, b int32) int32 {
	return ((a % b) + b) % b
}

func (dc *DispersionFFT) FFT(m *data.Slice) {

	for c := 0; c < 3; c++ {
		dc.fftPlan.handle.ExecR2C(cu.DevicePtr(uintptr(m.DevPtr(c))), cu.DevicePtr(uintptr(dc.fftCBuf.DevPtr(c))))
	}

	cfg := make1DConf(dc.nSamples)
	for c := 0; c < 3; c++ {
		k_selectPoints_async(dc.getPtrForTime(c), dc.nSamples, dc.fftCBuf.DevPtr(c),
			dc.fftLogicSize[X], dc.fftLogicSize[Y], dc.fftLogicSize[Z],
			dc.samplesPtrs[X], dc.samplesPtrs[Y], dc.samplesPtrs[Z], cfg)
	}

	dc.t++

}

func (dc *DispersionFFT) getPtrForTime(comp int) unsafe.Pointer {
	shift := uintptr(dc.SpaceTrans.Index(0, dc.t, 0)) * cu.SIZEOF_FLOAT32
	return unsafe.Pointer(uintptr(dc.SpaceTrans.DevPtr(comp)) + shift)
}

func (dc *DispersionFFT) OutputSize() [3]int {
	return fftR2COutputSizeFloats(dc.inputSize)
}

// should only be called once, because the spatially transformed modes are destroyed in this operation.
func (dc *DispersionFFT) TimeFFT() [3][][]float32 {

	//I guess this should return the space and time transformed fourier modes.
	//And probably extract the amplitude

	n := dc.timeSteps    //transform size
	batch := dc.nSamples //batch size (no 2* here because this factor is just to pretend complex in data storage)

	//handle for column-wise 1D transform
	handle := cufft.ProperPlanMany([]int{n}, []int{n}, batch, 1, []int{n}, batch, 1, cufft.C2C, batch)
	handle.SetStream(stream0)
	plan := fftplan{handle}

	cresult := NewSlice(3, [3]int{1, dc.timeSteps, 2 * dc.nSamples})

	for c := 0; c < 3; c++ {
		plan.handle.ExecC2C(cu.DevicePtr(uintptr(dc.SpaceTrans.DevPtr(0))),
			cu.DevicePtr(uintptr(cresult.DevPtr(c))), cufft.FORWARD)
	}

	//free things now as we potentially might like to save space
	dc.free(plan)

	result := NewSlice(3, [3]int{dc.nSamples, dc.timeSteps, 1})
	cfg := make1DConf(result.Len())
	for c := 0; c < 3; c++ {
		k_complexMagnitude_async(result.DevPtr(c), cresult.DevPtr(c),
			result.Len(), cfg)
	}

	resultVectors := result.HostCopy().Vectors()
	result.Free()
	cresult.Free()

	//ok so now data is stored in a timesteps by nSamples slice.
	//we want to return this to the user 1) on cpu, 2) convert back to the order of requested samples.

	var resultOrdered [3][][]float32
	for c := 0; c < 3; c++ {

		resultOrdered[c] = make([][]float32, len(dc.askedToComputed))
		for i := 0; i < len(dc.askedToComputed); i++ {
			resultOrdered[c][i] = make([]float32, dc.timeSteps)
			for f := 0; f < dc.timeSteps; f++ {
				resultOrdered[c][i][f] = resultVectors[c][0][f][dc.askedToComputed[i]]
			}
		}
	}

	return resultOrdered

}

func (dc *DispersionFFT) free(timePlan fftplan) {
	dc.fftCBuf.Free()
	dc.SpaceTrans.Free()
	for c := 0; c < 3; c++ {
		memFree(dc.samplesPtrs[c])
	}

	dc.fftPlan.Free()
	timePlan.Free()
	cudaCtx.SetCurrent() //need to do this otherwise the will get context destroyed errors next time something tries to work with cuda
}
