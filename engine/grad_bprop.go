package engine

import (
	"github.com/mumax/3/data"
)

func BProp(lm, lb *data.Slice) {
	demagProp(lm, lb)
	addExchangeProp(lm, lb)
	addAnisotropyProp(lm, lb)
}
