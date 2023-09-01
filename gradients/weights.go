package gradients

import (
	"github.com/mumax/3/data"
)

type Weights interface {
	W() []float32                    //returns the values of the weights
	Grad() []float32                 //returns the value of the loss with respect to the vector of weights
	Prop(dLdm_n, dLdb_n *data.Slice) //adds the contribution from timestep n to the weights
}
