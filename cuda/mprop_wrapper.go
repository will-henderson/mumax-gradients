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

// CUDA handle for mProp kernel
var mProp_code cu.Function

// Stores the arguments for mProp kernel invocation
type mProp_args_t struct {
	arg_new_lmx   unsafe.Pointer
	arg_new_lmy   unsafe.Pointer
	arg_new_lmz   unsafe.Pointer
	arg_new_lbx   unsafe.Pointer
	arg_new_lby   unsafe.Pointer
	arg_new_lbz   unsafe.Pointer
	arg_mx        unsafe.Pointer
	arg_my        unsafe.Pointer
	arg_mz        unsafe.Pointer
	arg_bx        unsafe.Pointer
	arg_by        unsafe.Pointer
	arg_bz        unsafe.Pointer
	arg_old_lmx   unsafe.Pointer
	arg_old_lmy   unsafe.Pointer
	arg_old_lmz   unsafe.Pointer
	arg_alpha_    unsafe.Pointer
	arg_alpha_mul float32
	arg_N         int
	argptr        [18]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for mProp kernel invocation
var mProp_args mProp_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	mProp_args.argptr[0] = unsafe.Pointer(&mProp_args.arg_new_lmx)
	mProp_args.argptr[1] = unsafe.Pointer(&mProp_args.arg_new_lmy)
	mProp_args.argptr[2] = unsafe.Pointer(&mProp_args.arg_new_lmz)
	mProp_args.argptr[3] = unsafe.Pointer(&mProp_args.arg_new_lbx)
	mProp_args.argptr[4] = unsafe.Pointer(&mProp_args.arg_new_lby)
	mProp_args.argptr[5] = unsafe.Pointer(&mProp_args.arg_new_lbz)
	mProp_args.argptr[6] = unsafe.Pointer(&mProp_args.arg_mx)
	mProp_args.argptr[7] = unsafe.Pointer(&mProp_args.arg_my)
	mProp_args.argptr[8] = unsafe.Pointer(&mProp_args.arg_mz)
	mProp_args.argptr[9] = unsafe.Pointer(&mProp_args.arg_bx)
	mProp_args.argptr[10] = unsafe.Pointer(&mProp_args.arg_by)
	mProp_args.argptr[11] = unsafe.Pointer(&mProp_args.arg_bz)
	mProp_args.argptr[12] = unsafe.Pointer(&mProp_args.arg_old_lmx)
	mProp_args.argptr[13] = unsafe.Pointer(&mProp_args.arg_old_lmy)
	mProp_args.argptr[14] = unsafe.Pointer(&mProp_args.arg_old_lmz)
	mProp_args.argptr[15] = unsafe.Pointer(&mProp_args.arg_alpha_)
	mProp_args.argptr[16] = unsafe.Pointer(&mProp_args.arg_alpha_mul)
	mProp_args.argptr[17] = unsafe.Pointer(&mProp_args.arg_N)
}

// Wrapper for mProp CUDA kernel, asynchronous.
func k_mProp_async(new_lmx unsafe.Pointer, new_lmy unsafe.Pointer, new_lmz unsafe.Pointer, new_lbx unsafe.Pointer, new_lby unsafe.Pointer, new_lbz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, bx unsafe.Pointer, by unsafe.Pointer, bz unsafe.Pointer, old_lmx unsafe.Pointer, old_lmy unsafe.Pointer, old_lmz unsafe.Pointer, alpha_ unsafe.Pointer, alpha_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("mProp")
	}

	mProp_args.Lock()
	defer mProp_args.Unlock()

	if mProp_code == 0 {
		mProp_code = fatbinLoad(mProp_map, "mProp")
	}

	mProp_args.arg_new_lmx = new_lmx
	mProp_args.arg_new_lmy = new_lmy
	mProp_args.arg_new_lmz = new_lmz
	mProp_args.arg_new_lbx = new_lbx
	mProp_args.arg_new_lby = new_lby
	mProp_args.arg_new_lbz = new_lbz
	mProp_args.arg_mx = mx
	mProp_args.arg_my = my
	mProp_args.arg_mz = mz
	mProp_args.arg_bx = bx
	mProp_args.arg_by = by
	mProp_args.arg_bz = bz
	mProp_args.arg_old_lmx = old_lmx
	mProp_args.arg_old_lmy = old_lmy
	mProp_args.arg_old_lmz = old_lmz
	mProp_args.arg_alpha_ = alpha_
	mProp_args.arg_alpha_mul = alpha_mul
	mProp_args.arg_N = N

	args := mProp_args.argptr[:]
	cu.LaunchKernel(mProp_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("mProp")
	}
}

// maps compute capability on PTX code for mProp kernel.
var mProp_map = map[int]string{0: "",
	80: mProp_ptx_80}

// mProp PTX code for various compute capabilities.
const (
	mProp_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	mProp

.visible .entry mProp(
	.param .u64 mProp_param_0,
	.param .u64 mProp_param_1,
	.param .u64 mProp_param_2,
	.param .u64 mProp_param_3,
	.param .u64 mProp_param_4,
	.param .u64 mProp_param_5,
	.param .u64 mProp_param_6,
	.param .u64 mProp_param_7,
	.param .u64 mProp_param_8,
	.param .u64 mProp_param_9,
	.param .u64 mProp_param_10,
	.param .u64 mProp_param_11,
	.param .u64 mProp_param_12,
	.param .u64 mProp_param_13,
	.param .u64 mProp_param_14,
	.param .u64 mProp_param_15,
	.param .f32 mProp_param_16,
	.param .u32 mProp_param_17
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<87>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<53>;


	ld.param.u64 	%rd2, [mProp_param_0];
	ld.param.u64 	%rd3, [mProp_param_1];
	ld.param.u64 	%rd4, [mProp_param_2];
	ld.param.u64 	%rd5, [mProp_param_3];
	ld.param.u64 	%rd6, [mProp_param_4];
	ld.param.u64 	%rd7, [mProp_param_5];
	ld.param.u64 	%rd8, [mProp_param_6];
	ld.param.u64 	%rd9, [mProp_param_7];
	ld.param.u64 	%rd10, [mProp_param_8];
	ld.param.u64 	%rd11, [mProp_param_9];
	ld.param.u64 	%rd12, [mProp_param_10];
	ld.param.u64 	%rd13, [mProp_param_11];
	ld.param.u64 	%rd14, [mProp_param_12];
	ld.param.u64 	%rd15, [mProp_param_13];
	ld.param.u64 	%rd16, [mProp_param_14];
	ld.param.u64 	%rd17, [mProp_param_15];
	ld.param.f32 	%f86, [mProp_param_16];
	ld.param.u32 	%r2, [mProp_param_17];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd18, %rd8;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd19, %r1, 4;
	add.s64 	%rd20, %rd18, %rd19;
	cvta.to.global.u64 	%rd21, %rd9;
	add.s64 	%rd22, %rd21, %rd19;
	cvta.to.global.u64 	%rd23, %rd10;
	add.s64 	%rd24, %rd23, %rd19;
	cvta.to.global.u64 	%rd25, %rd11;
	add.s64 	%rd26, %rd25, %rd19;
	cvta.to.global.u64 	%rd27, %rd12;
	add.s64 	%rd28, %rd27, %rd19;
	cvta.to.global.u64 	%rd29, %rd13;
	add.s64 	%rd30, %rd29, %rd19;
	cvta.to.global.u64 	%rd31, %rd14;
	add.s64 	%rd32, %rd31, %rd19;
	cvta.to.global.u64 	%rd33, %rd15;
	add.s64 	%rd34, %rd33, %rd19;
	cvta.to.global.u64 	%rd35, %rd16;
	add.s64 	%rd36, %rd35, %rd19;
	ld.global.nc.f32 	%f1, [%rd34];
	ld.global.nc.f32 	%f2, [%rd30];
	ld.global.nc.f32 	%f3, [%rd36];
	ld.global.nc.f32 	%f4, [%rd28];
	ld.global.nc.f32 	%f5, [%rd26];
	ld.global.nc.f32 	%f6, [%rd32];
	ld.global.nc.f32 	%f7, [%rd22];
	ld.global.nc.f32 	%f8, [%rd24];
	ld.global.nc.f32 	%f9, [%rd20];
	setp.eq.s64 	%p2, %rd17, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd37, %rd17;
	shl.b64 	%rd38, %rd1, 2;
	add.s64 	%rd39, %rd37, %rd38;
	ld.global.nc.f32 	%f13, [%rd39];
	mul.f32 	%f86, %f13, %f86;

$L__BB0_3:
	mul.f32 	%f14, %f8, %f4;
	mul.f32 	%f15, %f7, %f2;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f9, %f2;
	mul.f32 	%f18, %f8, %f5;
	sub.f32 	%f19, %f18, %f17;
	mul.f32 	%f20, %f7, %f5;
	mul.f32 	%f21, %f9, %f4;
	sub.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f7, %f3;
	mul.f32 	%f24, %f8, %f1;
	sub.f32 	%f25, %f24, %f23;
	mul.f32 	%f26, %f8, %f6;
	mul.f32 	%f27, %f9, %f3;
	sub.f32 	%f28, %f27, %f26;
	mul.f32 	%f29, %f9, %f1;
	mul.f32 	%f30, %f7, %f6;
	sub.f32 	%f31, %f30, %f29;
	fma.rn.f32 	%f32, %f86, %f86, 0f3F800000;
	rcp.rn.f32 	%f33, %f32;
	mul.f32 	%f34, %f19, %f3;
	mul.f32 	%f35, %f22, %f1;
	sub.f32 	%f36, %f35, %f34;
	mul.f32 	%f37, %f6, %f22;
	mul.f32 	%f38, %f16, %f3;
	sub.f32 	%f39, %f38, %f37;
	mul.f32 	%f40, %f1, %f16;
	mul.f32 	%f41, %f6, %f19;
	sub.f32 	%f42, %f41, %f40;
	mul.f32 	%f43, %f4, %f31;
	mul.f32 	%f44, %f2, %f28;
	sub.f32 	%f45, %f44, %f43;
	mul.f32 	%f46, %f2, %f25;
	mul.f32 	%f47, %f5, %f31;
	sub.f32 	%f48, %f47, %f46;
	mul.f32 	%f49, %f5, %f28;
	mul.f32 	%f50, %f4, %f25;
	sub.f32 	%f51, %f50, %f49;
	add.f32 	%f52, %f36, %f45;
	add.f32 	%f53, %f39, %f48;
	add.f32 	%f54, %f42, %f51;
	mul.f32 	%f55, %f4, %f3;
	mul.f32 	%f56, %f2, %f1;
	sub.f32 	%f57, %f56, %f55;
	fma.rn.f32 	%f58, %f52, %f86, %f57;
	mul.f32 	%f59, %f2, %f6;
	mul.f32 	%f60, %f5, %f3;
	sub.f32 	%f61, %f60, %f59;
	fma.rn.f32 	%f62, %f53, %f86, %f61;
	mul.f32 	%f63, %f5, %f1;
	mul.f32 	%f64, %f4, %f6;
	sub.f32 	%f65, %f64, %f63;
	fma.rn.f32 	%f66, %f54, %f86, %f65;
	neg.f32 	%f67, %f33;
	mul.f32 	%f68, %f7, %f31;
	mul.f32 	%f69, %f8, %f28;
	sub.f32 	%f70, %f69, %f68;
	mul.f32 	%f71, %f8, %f25;
	mul.f32 	%f72, %f9, %f31;
	sub.f32 	%f73, %f72, %f71;
	mul.f32 	%f74, %f9, %f28;
	mul.f32 	%f75, %f7, %f25;
	sub.f32 	%f76, %f75, %f74;
	fma.rn.f32 	%f77, %f70, %f86, %f25;
	fma.rn.f32 	%f78, %f73, %f86, %f28;
	fma.rn.f32 	%f79, %f76, %f86, %f31;
	mul.f32 	%f80, %f77, %f67;
	mul.f32 	%f81, %f78, %f67;
	mul.f32 	%f82, %f79, %f67;
	fma.rn.f32 	%f83, %f58, %f33, %f6;
	fma.rn.f32 	%f84, %f62, %f33, %f1;
	fma.rn.f32 	%f85, %f66, %f33, %f3;
	cvta.to.global.u64 	%rd40, %rd2;
	shl.b64 	%rd41, %rd1, 2;
	add.s64 	%rd42, %rd40, %rd41;
	st.global.f32 	[%rd42], %f83;
	cvta.to.global.u64 	%rd43, %rd3;
	add.s64 	%rd44, %rd43, %rd41;
	st.global.f32 	[%rd44], %f84;
	cvta.to.global.u64 	%rd45, %rd4;
	add.s64 	%rd46, %rd45, %rd41;
	st.global.f32 	[%rd46], %f85;
	cvta.to.global.u64 	%rd47, %rd5;
	add.s64 	%rd48, %rd47, %rd41;
	st.global.f32 	[%rd48], %f80;
	cvta.to.global.u64 	%rd49, %rd6;
	add.s64 	%rd50, %rd49, %rd41;
	st.global.f32 	[%rd50], %f81;
	cvta.to.global.u64 	%rd51, %rd7;
	add.s64 	%rd52, %rd51, %rd41;
	st.global.f32 	[%rd52], %f82;

$L__BB0_4:
	ret;

}

`
)
