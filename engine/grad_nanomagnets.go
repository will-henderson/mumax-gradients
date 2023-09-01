package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// we assume we have nanomagnets with have some distribution over the material, and have magnetisation vector lying in plane.
type location struct {
	x float64
	y float64
}

type Nanomagnets struct {
	magnetSize [3]float64
	offset     float64
	n          int
	pos        []location
}

// Sets a uniform grid of nanomagnets on a mesh.
func (nm Nanomagnets) UniformSpace(c1, c2, separation location) {

}

func (nm Nanomagnets) RandomSpace(c1, c2 location, number, seed int) {

}

// assumes we have magnetic cubes sitting above the sample with magnetisation pointing in a planar direction.
type PlanarCubes struct {
	Nanomagnets
	pbc           [3]int
	demagAccuracy float64
	cacheDir      string
	θ             []float64
	msat          float64
}

func (pc PlanarCubes) B() *data.Slice {

	var magnetCells [3]int
	for c := 0; c < 3; c++ {
		magnetCells[c] = int(math.Ceil(pc.magnetSize[c] / Mesh().CellSize()[c]))
	}

	//we need a new kernel that includes some mesh above to hold these magnets in.
	var bigMeshSize [3]int
	bigMeshSize[0] = Mesh().Size()[0]
	bigMeshSize[1] = Mesh().Size()[1]

	//we round offset down and magnetSize up
	magnetLayer := int(math.Floor((pc.offset) / Mesh().CellSize()[2]))
	nmLayers := magnetLayer + magnetCells[2]
	bigMeshSize[2] = Mesh().Size()[2] + nmLayers

	//we round magnet size up to be in units of cellsize.
	SetBusy(true)
	defer SetBusy(false)
	kernel := mag.DemagKernel(bigMeshSize, Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, *Flag_cachedir)
	conv := cuda.NewDemag(Mesh().Size(), Mesh().PBC(), kernel, *Flag_selftest)
	defer conv.Free() //we might actually want to keep this for reuse everytime this is called?

	//make the magnet array on the cpu then copy to the gpu
	magnetsCPU := data.NewSlice(3, bigMeshSize)
	vectors := magnetsCPU.Vectors()
	for c := 0; c < 3; c++ {
		for k := 0; k < magnetCells[2]; k++ {
			for j := 0; j < magnetCells[1]; j++ {
				for i := 0; i < magnetCells[2]; i++ {
					vectors[c][k][j][i] = 0
				}
			}
		}
	}

	for idx, point := range pc.pos {

		pointCellX := int(math.Floor(point.x / Mesh().CellSize()[0]))
		pointCellY := int(math.Floor(point.y / Mesh().CellSize()[1]))

		magX := float32(math.Cos(pc.θ[idx]))
		magY := float32(math.Sin(pc.θ[idx]))

		for i := 0; i < magnetCells[0]; i++ {
			for j := 0; j < magnetCells[1]; j++ {
				for k := 0; k < magnetCells[2]; k++ {
					//do a plus equal to deal with overlaps sensibly
					vectors[0][nmLayers+k][pointCellY+j][pointCellX+i] += magX
					vectors[1][nmLayers+k][pointCellY+j][pointCellX+i] += magY

				}
			}
		}
	}

	magnets := cuda.NewSlice(3, bigMeshSize)
	defer magnets.Free()
	data.Copy(magnets, magnetsCPU)

	bigB := cuda.NewSlice(3, bigMeshSize)
	defer bigB.Free()

	//need to mask out
	msat := cuda.MakeMSlice(data.NilSlice(1, Mesh().Size()), []float64{pc.msat})
	defer msat.Recycle()

	conv.Exec(bigB, magnets, data.NilSlice(1, Mesh().Size()), msat)

	//should actually use the cuda.Crop function here.
	b := cuda.NewSlice(3, MeshSize())
	for c := 0; c < 3; c++ {
		cuda.MemCpy(b.DevPtr(c), bigB.DevPtr(c), data.SIZEOF_FLOAT32*int64(prod(MeshSize())))
	}

	//note, because of sparseness it might actually be more efficient to calculate this directly rather than via fourier transform.

	return b

}
