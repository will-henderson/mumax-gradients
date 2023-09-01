package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/gradients"
)

type Loss interface {
	// Returns the derivative of the loss with respect to the magnetisation at timestep n.
	// i.e. how the loss directly depends on the value of the magnetisation at timestep n.
	DirectLoss(n int, m *data.Slice) *data.Slice
}

func LossDerivatives(lossObj Loss, loader data.Loader, w gradients.Weights) {

	var timesteps int

	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	lm := cuda.NewSlice(3, Mesh().Size())

	b2m := cuda.NewSlice(3, Mesh().Size())
	cuda.Zero(b2m)

	m2m := cuda.NewSlice(3, Mesh().Size())
	cuda.Zero(m2m)

	m2b := cuda.NewSlice(3, Mesh().Size())
	cuda.Zero(m2b)

	for n := timesteps - 1; n >= 0; n-- {

		m := loader.M(n)
		b := loader.B(n)

		directLoss := lossObj.DirectLoss(n, m)
		cuda.Madd3(lm, directLoss, m2m, b2m, 1., 1., 1.)

		// relate Wn to Mn and Bn
		w.Prop(lm, b2m)

		//find Bn-1 ready for next timestep.
		cuda.MProp(m2m, m2b, m, b, lm, alpha)
		BProp(b2m, m2b)

	}
}
