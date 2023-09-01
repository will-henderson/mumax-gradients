package cufft

import (
	"unsafe"
)

//#include <cufft.h>
import "C"

//need a planmany that will allow different idist

func ProperPlanMany(n, inembed []int, istride, idist int, oembed []int, ostride, odist int, typ Type, batch int) Handle {
	var handle C.cufftHandle

	NULL := (*C.int)(unsafe.Pointer(uintptr(0)))

	inembedptr := NULL
	if inembed != nil {
		inembedptr = (*C.int)(unsafe.Pointer(&inembed[0]))
	}

	oembedptr := NULL
	if oembed != nil {
		oembedptr = (*C.int)(unsafe.Pointer(&oembed[0]))
	}

	err := Result(C.cufftPlanMany(
		&handle,
		C.int(len(n)),                   // rank
		(*C.int)(unsafe.Pointer(&n[0])), // n
		inembedptr,
		C.int(istride),
		C.int(idist),
		oembedptr,
		C.int(ostride),
		C.int(odist),
		C.cufftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}
