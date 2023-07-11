package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for addMul kernel
var addMul_code cu.Function

// Stores the arguments for addMul kernel invocation
type addMul_args_t struct {
	arg_dst  unsafe.Pointer
	arg_src1 unsafe.Pointer
	arg_src2 unsafe.Pointer
	arg_N    int
	argptr   [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addMul kernel invocation
var addMul_args addMul_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addMul_args.argptr[0] = unsafe.Pointer(&addMul_args.arg_dst)
	addMul_args.argptr[1] = unsafe.Pointer(&addMul_args.arg_src1)
	addMul_args.argptr[2] = unsafe.Pointer(&addMul_args.arg_src2)
	addMul_args.argptr[3] = unsafe.Pointer(&addMul_args.arg_N)
}

// Wrapper for addMul CUDA kernel, asynchronous.
func k_addMul_async(dst unsafe.Pointer, src1 unsafe.Pointer, src2 unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addMul")
	}

	addMul_args.Lock()
	defer addMul_args.Unlock()

	if addMul_code == 0 {
		addMul_code = fatbinLoad(addMul_map, "addMul")
	}

	addMul_args.arg_dst = dst
	addMul_args.arg_src1 = src1
	addMul_args.arg_src2 = src2
	addMul_args.arg_N = N

	args := addMul_args.argptr[:]
	cu.LaunchKernel(addMul_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addMul")
	}
}

// maps compute capability on PTX code for addMul kernel.
var addMul_map = map[int]string{0: "",
	50: addMul_ptx_50}

// addMul PTX code for various compute capabilities.
const (
	addMul_ptx_50 = `
.version 7.8
.target sm_50
.address_size 64

	// .globl	addMul

.visible .entry addMul(
	.param .u64 addMul_param_0,
	.param .u64 addMul_param_1,
	.param .u64 addMul_param_2,
	.param .u32 addMul_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd1, [addMul_param_0];
	ld.param.u64 	%rd2, [addMul_param_1];
	ld.param.u64 	%rd3, [addMul_param_2];
	ld.param.u32 	%r2, [addMul_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.nc.f32 	%f1, [%rd8];
	ld.global.nc.f32 	%f2, [%rd6];
	cvta.to.global.u64 	%rd9, %rd1;
	add.s64 	%rd10, %rd9, %rd5;
	ld.global.f32 	%f3, [%rd10];
	fma.rn.f32 	%f4, %f2, %f1, %f3;
	st.global.f32 	[%rd10], %f4;

$L__BB0_2:
	ret;

}

`
)
