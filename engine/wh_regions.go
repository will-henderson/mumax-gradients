package engine

func RegionCounts() [NREGION]int {

	var rc [NREGION]int
	arr := regions.HostList()
	for i := 0; i < Mesh().NCell(); i++ {
		rc[arr[i]]++
	}
	return rc
}

func UniformGeometry() bool {

	return spaceFill() > 0.99 //not one because potential numerical error.

}

func Geometry() *geom {
	return &geometry
}
