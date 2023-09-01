package cuda

import (
	"github.com/mumax/3/data"
)

func MProp(new_lm, new_lb, m, B, old_lm *data.Slice, alpha MSlice) {
	N := new_lm.Len()
	cfg := make1DConf(N)

	k_mProp_async(new_lm.DevPtr(0), new_lm.DevPtr(1), new_lm.DevPtr(2),
		new_lb.DevPtr(0), new_lb.DevPtr(1), new_lb.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		B.DevPtr(0), B.DevPtr(1), B.DevPtr(2),
		old_lm.DevPtr(0), old_lm.DevPtr(1), old_lm.DevPtr(2),
		alpha.DevPtr(0), alpha.Mul(0), N, cfg)

}
