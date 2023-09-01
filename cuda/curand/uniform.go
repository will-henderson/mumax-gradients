package curand

//#include <curand.h>
import "C"

import (
	"unsafe"
)

func (g Generator) GenerateUniform(output uintptr, n int64) {
	err := Status(C.curandGenerateUniform(
		C.curandGenerator_t(unsafe.Pointer(uintptr(g))),
		(*C.float)(unsafe.Pointer(output)),
		C.size_t(n)))
	if err != SUCCESS {
		panic(err)
	}
}
