package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func addExchangeProp(lm, lb *data.Slice) {
	inter := !Dind.isZero()
	bulk := !Dbulk.isZero()
	ms := Msat.MSlice()
	defer ms.Recycle()
	switch {
	case !inter && !bulk:
		cuda.AddExchange(lm, lb, lex2.Gpu(), ms, regions.Gpu(), M.Mesh())
	case inter && !bulk:
		panic("addDMI not implemented")
	case bulk && !inter:
		panic("addDMIbulk not implemented")
		// TODO: add ScaleInterDbulk and InterDbulk
	case inter && bulk:
		util.Fatal("Cannot have interfacial-induced DMI and bulk DMI at the same time")
	}
}
