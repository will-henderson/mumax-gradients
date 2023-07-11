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

// CUDA handle for rotateMode kernel
var rotateMode_code cu.Function

// Stores the arguments for rotateMode kernel invocation
type rotateMode_args_t struct {
	arg_dstx unsafe.Pointer
	arg_dsty unsafe.Pointer
	arg_mx   unsafe.Pointer
	arg_my   unsafe.Pointer
	arg_mz   unsafe.Pointer
	arg_Rxx  unsafe.Pointer
	arg_Rxy  unsafe.Pointer
	arg_Rxz  unsafe.Pointer
	arg_Ryx  unsafe.Pointer
	arg_Ryy  unsafe.Pointer
	arg_Ryz  unsafe.Pointer
	arg_N    int
	argptr   [12]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for rotateMode kernel invocation
var rotateMode_args rotateMode_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	rotateMode_args.argptr[0] = unsafe.Pointer(&rotateMode_args.arg_dstx)
	rotateMode_args.argptr[1] = unsafe.Pointer(&rotateMode_args.arg_dsty)
	rotateMode_args.argptr[2] = unsafe.Pointer(&rotateMode_args.arg_mx)
	rotateMode_args.argptr[3] = unsafe.Pointer(&rotateMode_args.arg_my)
	rotateMode_args.argptr[4] = unsafe.Pointer(&rotateMode_args.arg_mz)
	rotateMode_args.argptr[5] = unsafe.Pointer(&rotateMode_args.arg_Rxx)
	rotateMode_args.argptr[6] = unsafe.Pointer(&rotateMode_args.arg_Rxy)
	rotateMode_args.argptr[7] = unsafe.Pointer(&rotateMode_args.arg_Rxz)
	rotateMode_args.argptr[8] = unsafe.Pointer(&rotateMode_args.arg_Ryx)
	rotateMode_args.argptr[9] = unsafe.Pointer(&rotateMode_args.arg_Ryy)
	rotateMode_args.argptr[10] = unsafe.Pointer(&rotateMode_args.arg_Ryz)
	rotateMode_args.argptr[11] = unsafe.Pointer(&rotateMode_args.arg_N)
}

// Wrapper for rotateMode CUDA kernel, asynchronous.
func k_rotateMode_async(dstx unsafe.Pointer, dsty unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Rxx unsafe.Pointer, Rxy unsafe.Pointer, Rxz unsafe.Pointer, Ryx unsafe.Pointer, Ryy unsafe.Pointer, Ryz unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("rotateMode")
	}

	rotateMode_args.Lock()
	defer rotateMode_args.Unlock()

	if rotateMode_code == 0 {
		rotateMode_code = fatbinLoad(rotateMode_map, "rotateMode")
	}

	rotateMode_args.arg_dstx = dstx
	rotateMode_args.arg_dsty = dsty
	rotateMode_args.arg_mx = mx
	rotateMode_args.arg_my = my
	rotateMode_args.arg_mz = mz
	rotateMode_args.arg_Rxx = Rxx
	rotateMode_args.arg_Rxy = Rxy
	rotateMode_args.arg_Rxz = Rxz
	rotateMode_args.arg_Ryx = Ryx
	rotateMode_args.arg_Ryy = Ryy
	rotateMode_args.arg_Ryz = Ryz
	rotateMode_args.arg_N = N

	args := rotateMode_args.argptr[:]
	cu.LaunchKernel(rotateMode_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("rotateMode")
	}
}

// maps compute capability on PTX code for rotateMode kernel.
var rotateMode_map = map[int]string{0: "",
	50: rotateMode_ptx_50}

// rotateMode PTX code for various compute capabilities.
const (
	rotateMode_ptx_50 = `
.version 7.8
.target sm_50
.address_size 64

	// .globl	rotateMode

.visible .entry rotateMode(
	.param .u64 rotateMode_param_0,
	.param .u64 rotateMode_param_1,
	.param .u64 rotateMode_param_2,
	.param .u64 rotateMode_param_3,
	.param .u64 rotateMode_param_4,
	.param .u64 rotateMode_param_5,
	.param .u64 rotateMode_param_6,
	.param .u64 rotateMode_param_7,
	.param .u64 rotateMode_param_8,
	.param .u64 rotateMode_param_9,
	.param .u64 rotateMode_param_10,
	.param .u32 rotateMode_param_11
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd1, [rotateMode_param_0];
	ld.param.u64 	%rd2, [rotateMode_param_1];
	ld.param.u64 	%rd3, [rotateMode_param_2];
	ld.param.u64 	%rd4, [rotateMode_param_3];
	ld.param.u64 	%rd5, [rotateMode_param_4];
	ld.param.u64 	%rd6, [rotateMode_param_5];
	ld.param.u64 	%rd7, [rotateMode_param_6];
	ld.param.u64 	%rd8, [rotateMode_param_7];
	ld.param.u64 	%rd9, [rotateMode_param_8];
	ld.param.u64 	%rd10, [rotateMode_param_9];
	ld.param.u64 	%rd11, [rotateMode_param_10];
	ld.param.u32 	%r2, [rotateMode_param_11];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd12, %rd3;
	cvta.to.global.u64 	%rd13, %rd6;
	mul.wide.s32 	%rd14, %r1, 4;
	add.s64 	%rd15, %rd13, %rd14;
	add.s64 	%rd16, %rd12, %rd14;
	ld.global.nc.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd18, %rd17, %rd14;
	cvta.to.global.u64 	%rd19, %rd4;
	add.s64 	%rd20, %rd19, %rd14;
	ld.global.nc.f32 	%f3, [%rd20];
	ld.global.nc.f32 	%f4, [%rd18];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd21, %rd8;
	add.s64 	%rd22, %rd21, %rd14;
	cvta.to.global.u64 	%rd23, %rd5;
	add.s64 	%rd24, %rd23, %rd14;
	ld.global.nc.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd25, %rd1;
	add.s64 	%rd26, %rd25, %rd14;
	st.global.f32 	[%rd26], %f9;
	cvta.to.global.u64 	%rd27, %rd9;
	add.s64 	%rd28, %rd27, %rd14;
	ld.global.nc.f32 	%f10, [%rd28];
	cvta.to.global.u64 	%rd29, %rd10;
	add.s64 	%rd30, %rd29, %rd14;
	ld.global.nc.f32 	%f11, [%rd30];
	mul.f32 	%f12, %f11, %f3;
	fma.rn.f32 	%f13, %f10, %f1, %f12;
	cvta.to.global.u64 	%rd31, %rd11;
	add.s64 	%rd32, %rd31, %rd14;
	ld.global.nc.f32 	%f14, [%rd32];
	fma.rn.f32 	%f15, %f14, %f7, %f13;
	cvta.to.global.u64 	%rd33, %rd2;
	add.s64 	%rd34, %rd33, %rd14;
	st.global.f32 	[%rd34], %f15;

$L__BB0_2:
	ret;

}

`
)