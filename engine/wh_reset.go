package engine

import (
	"github.com/mumax/3/cuda"
)

func init() {
	DeclFunc("Reset", Reset, "Forgets all the defined regions")
}

func Reset() {
	if !(globalmesh_.Size() == [3]int{0, 0, 0}) {
		ResetRegions()
		ResetGeom()
		ResetCentering()
	}
}

func ResetRegions() {
	regions.hist = nil
	regions.gpuCache.Free()
	mesh := regions.Mesh()
	regions.gpuCache = cuda.NewBytes(mesh.NCell())
	DefRegion(0, universe)
}

func ResetCentering() {
	postStep = nil
	lastShift = 0
	lastT = 0
	lastV = 0
	TotalShift = 0
	TotalYShift = 0
	Time = 0
}

func ResetGeom() {
	SetGeom(universe)
}
