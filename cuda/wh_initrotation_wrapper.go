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

// CUDA handle for initRotation kernel
var initRotation_code cu.Function

// Stores the arguments for initRotation kernel invocation
type initRotation_args_t struct {
	arg_mx  unsafe.Pointer
	arg_my  unsafe.Pointer
	arg_mz  unsafe.Pointer
	arg_Rxx unsafe.Pointer
	arg_Rxy unsafe.Pointer
	arg_Rxz unsafe.Pointer
	arg_Ryx unsafe.Pointer
	arg_Ryy unsafe.Pointer
	arg_Ryz unsafe.Pointer
	arg_Rzx unsafe.Pointer
	arg_Rzy unsafe.Pointer
	arg_Rzz unsafe.Pointer
	arg_N   int
	argptr  [13]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for initRotation kernel invocation
var initRotation_args initRotation_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	initRotation_args.argptr[0] = unsafe.Pointer(&initRotation_args.arg_mx)
	initRotation_args.argptr[1] = unsafe.Pointer(&initRotation_args.arg_my)
	initRotation_args.argptr[2] = unsafe.Pointer(&initRotation_args.arg_mz)
	initRotation_args.argptr[3] = unsafe.Pointer(&initRotation_args.arg_Rxx)
	initRotation_args.argptr[4] = unsafe.Pointer(&initRotation_args.arg_Rxy)
	initRotation_args.argptr[5] = unsafe.Pointer(&initRotation_args.arg_Rxz)
	initRotation_args.argptr[6] = unsafe.Pointer(&initRotation_args.arg_Ryx)
	initRotation_args.argptr[7] = unsafe.Pointer(&initRotation_args.arg_Ryy)
	initRotation_args.argptr[8] = unsafe.Pointer(&initRotation_args.arg_Ryz)
	initRotation_args.argptr[9] = unsafe.Pointer(&initRotation_args.arg_Rzx)
	initRotation_args.argptr[10] = unsafe.Pointer(&initRotation_args.arg_Rzy)
	initRotation_args.argptr[11] = unsafe.Pointer(&initRotation_args.arg_Rzz)
	initRotation_args.argptr[12] = unsafe.Pointer(&initRotation_args.arg_N)
}

// Wrapper for initRotation CUDA kernel, asynchronous.
func k_initRotation_async(mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Rxx unsafe.Pointer, Rxy unsafe.Pointer, Rxz unsafe.Pointer, Ryx unsafe.Pointer, Ryy unsafe.Pointer, Ryz unsafe.Pointer, Rzx unsafe.Pointer, Rzy unsafe.Pointer, Rzz unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("initRotation")
	}

	initRotation_args.Lock()
	defer initRotation_args.Unlock()

	if initRotation_code == 0 {
		initRotation_code = fatbinLoad(initRotation_map, "initRotation")
	}

	initRotation_args.arg_mx = mx
	initRotation_args.arg_my = my
	initRotation_args.arg_mz = mz
	initRotation_args.arg_Rxx = Rxx
	initRotation_args.arg_Rxy = Rxy
	initRotation_args.arg_Rxz = Rxz
	initRotation_args.arg_Ryx = Ryx
	initRotation_args.arg_Ryy = Ryy
	initRotation_args.arg_Ryz = Ryz
	initRotation_args.arg_Rzx = Rzx
	initRotation_args.arg_Rzy = Rzy
	initRotation_args.arg_Rzz = Rzz
	initRotation_args.arg_N = N

	args := initRotation_args.argptr[:]
	cu.LaunchKernel(initRotation_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("initRotation")
	}
}

// maps compute capability on PTX code for initRotation kernel.
var initRotation_map = map[int]string{0: "",
	80: initRotation_ptx_80}

// initRotation PTX code for various compute capabilities.
const (
	initRotation_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	initRotation

.visible .entry initRotation(
	.param .u64 initRotation_param_0,
	.param .u64 initRotation_param_1,
	.param .u64 initRotation_param_2,
	.param .u64 initRotation_param_3,
	.param .u64 initRotation_param_4,
	.param .u64 initRotation_param_5,
	.param .u64 initRotation_param_6,
	.param .u64 initRotation_param_7,
	.param .u64 initRotation_param_8,
	.param .u64 initRotation_param_9,
	.param .u64 initRotation_param_10,
	.param .u64 initRotation_param_11,
	.param .u32 initRotation_param_12
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<38>;


	ld.param.u64 	%rd1, [initRotation_param_0];
	ld.param.u64 	%rd2, [initRotation_param_1];
	ld.param.u64 	%rd3, [initRotation_param_2];
	ld.param.u64 	%rd4, [initRotation_param_3];
	ld.param.u64 	%rd5, [initRotation_param_4];
	ld.param.u64 	%rd6, [initRotation_param_5];
	ld.param.u64 	%rd7, [initRotation_param_6];
	ld.param.u64 	%rd8, [initRotation_param_7];
	ld.param.u64 	%rd9, [initRotation_param_8];
	ld.param.u64 	%rd10, [initRotation_param_9];
	ld.param.u64 	%rd11, [initRotation_param_10];
	ld.param.u64 	%rd12, [initRotation_param_11];
	ld.param.u32 	%r2, [initRotation_param_12];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd13, %rd3;
	mul.wide.s32 	%rd14, %r1, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f1, [%rd15];
	mul.f32 	%f2, %f1, %f1;
	mov.f32 	%f3, 0f3F800000;
	sub.f32 	%f4, %f3, %f2;
	sqrt.rn.f32 	%f5, %f4;
	cvta.to.global.u64 	%rd16, %rd1;
	add.s64 	%rd17, %rd16, %rd14;
	ld.global.nc.f32 	%f6, [%rd17];
	div.rn.f32 	%f7, %f6, %f5;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd14;
	ld.global.nc.f32 	%f8, [%rd19];
	div.rn.f32 	%f9, %f8, %f5;
	mul.f32 	%f10, %f1, %f7;
	cvta.to.global.u64 	%rd20, %rd4;
	add.s64 	%rd21, %rd20, %rd14;
	st.global.f32 	[%rd21], %f10;
	mul.f32 	%f11, %f1, %f9;
	cvta.to.global.u64 	%rd22, %rd5;
	add.s64 	%rd23, %rd22, %rd14;
	st.global.f32 	[%rd23], %f11;
	neg.f32 	%f12, %f5;
	cvta.to.global.u64 	%rd24, %rd6;
	add.s64 	%rd25, %rd24, %rd14;
	st.global.f32 	[%rd25], %f12;
	neg.f32 	%f13, %f9;
	cvta.to.global.u64 	%rd26, %rd7;
	add.s64 	%rd27, %rd26, %rd14;
	st.global.f32 	[%rd27], %f13;
	cvta.to.global.u64 	%rd28, %rd8;
	add.s64 	%rd29, %rd28, %rd14;
	st.global.f32 	[%rd29], %f7;
	cvta.to.global.u64 	%rd30, %rd9;
	add.s64 	%rd31, %rd30, %rd14;
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd31], %r9;
	mul.f32 	%f14, %f5, %f7;
	cvta.to.global.u64 	%rd32, %rd10;
	add.s64 	%rd33, %rd32, %rd14;
	st.global.f32 	[%rd33], %f14;
	mul.f32 	%f15, %f5, %f9;
	cvta.to.global.u64 	%rd34, %rd11;
	add.s64 	%rd35, %rd34, %rd14;
	st.global.f32 	[%rd35], %f15;
	cvta.to.global.u64 	%rd36, %rd12;
	add.s64 	%rd37, %rd36, %rd14;
	st.global.f32 	[%rd37], %f1;

$L__BB0_2:
	ret;

}

`
)
