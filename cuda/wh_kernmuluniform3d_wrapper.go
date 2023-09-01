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

// CUDA handle for kernmulUniform3D kernel
var kernmulUniform3D_code cu.Function

// Stores the arguments for kernmulUniform3D kernel invocation
type kernmulUniform3D_args_t struct {
	arg_Fxx    unsafe.Pointer
	arg_Fyy    unsafe.Pointer
	arg_Fzz    unsafe.Pointer
	arg_Fyz    unsafe.Pointer
	arg_Fxz    unsafe.Pointer
	arg_Fxy    unsafe.Pointer
	arg_fftKxx unsafe.Pointer
	arg_fftKyy unsafe.Pointer
	arg_fftKzz unsafe.Pointer
	arg_fftKyz unsafe.Pointer
	arg_fftKxz unsafe.Pointer
	arg_fftKxy unsafe.Pointer
	arg_ftu    unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	argptr     [16]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for kernmulUniform3D kernel invocation
var kernmulUniform3D_args kernmulUniform3D_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	kernmulUniform3D_args.argptr[0] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fxx)
	kernmulUniform3D_args.argptr[1] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fyy)
	kernmulUniform3D_args.argptr[2] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fzz)
	kernmulUniform3D_args.argptr[3] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fyz)
	kernmulUniform3D_args.argptr[4] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fxz)
	kernmulUniform3D_args.argptr[5] = unsafe.Pointer(&kernmulUniform3D_args.arg_Fxy)
	kernmulUniform3D_args.argptr[6] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKxx)
	kernmulUniform3D_args.argptr[7] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKyy)
	kernmulUniform3D_args.argptr[8] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKzz)
	kernmulUniform3D_args.argptr[9] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKyz)
	kernmulUniform3D_args.argptr[10] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKxz)
	kernmulUniform3D_args.argptr[11] = unsafe.Pointer(&kernmulUniform3D_args.arg_fftKxy)
	kernmulUniform3D_args.argptr[12] = unsafe.Pointer(&kernmulUniform3D_args.arg_ftu)
	kernmulUniform3D_args.argptr[13] = unsafe.Pointer(&kernmulUniform3D_args.arg_Nx)
	kernmulUniform3D_args.argptr[14] = unsafe.Pointer(&kernmulUniform3D_args.arg_Ny)
	kernmulUniform3D_args.argptr[15] = unsafe.Pointer(&kernmulUniform3D_args.arg_Nz)
}

// Wrapper for kernmulUniform3D CUDA kernel, asynchronous.
func k_kernmulUniform3D_async(Fxx unsafe.Pointer, Fyy unsafe.Pointer, Fzz unsafe.Pointer, Fyz unsafe.Pointer, Fxz unsafe.Pointer, Fxy unsafe.Pointer, fftKxx unsafe.Pointer, fftKyy unsafe.Pointer, fftKzz unsafe.Pointer, fftKyz unsafe.Pointer, fftKxz unsafe.Pointer, fftKxy unsafe.Pointer, ftu unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("kernmulUniform3D")
	}

	kernmulUniform3D_args.Lock()
	defer kernmulUniform3D_args.Unlock()

	if kernmulUniform3D_code == 0 {
		kernmulUniform3D_code = fatbinLoad(kernmulUniform3D_map, "kernmulUniform3D")
	}

	kernmulUniform3D_args.arg_Fxx = Fxx
	kernmulUniform3D_args.arg_Fyy = Fyy
	kernmulUniform3D_args.arg_Fzz = Fzz
	kernmulUniform3D_args.arg_Fyz = Fyz
	kernmulUniform3D_args.arg_Fxz = Fxz
	kernmulUniform3D_args.arg_Fxy = Fxy
	kernmulUniform3D_args.arg_fftKxx = fftKxx
	kernmulUniform3D_args.arg_fftKyy = fftKyy
	kernmulUniform3D_args.arg_fftKzz = fftKzz
	kernmulUniform3D_args.arg_fftKyz = fftKyz
	kernmulUniform3D_args.arg_fftKxz = fftKxz
	kernmulUniform3D_args.arg_fftKxy = fftKxy
	kernmulUniform3D_args.arg_ftu = ftu
	kernmulUniform3D_args.arg_Nx = Nx
	kernmulUniform3D_args.arg_Ny = Ny
	kernmulUniform3D_args.arg_Nz = Nz

	args := kernmulUniform3D_args.argptr[:]
	cu.LaunchKernel(kernmulUniform3D_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("kernmulUniform3D")
	}
}

// maps compute capability on PTX code for kernmulUniform3D kernel.
var kernmulUniform3D_map = map[int]string{0: "",
	80: kernmulUniform3D_ptx_80}

// kernmulUniform3D PTX code for various compute capabilities.
const (
	kernmulUniform3D_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	kernmulUniform3D

.visible .entry kernmulUniform3D(
	.param .u64 kernmulUniform3D_param_0,
	.param .u64 kernmulUniform3D_param_1,
	.param .u64 kernmulUniform3D_param_2,
	.param .u64 kernmulUniform3D_param_3,
	.param .u64 kernmulUniform3D_param_4,
	.param .u64 kernmulUniform3D_param_5,
	.param .u64 kernmulUniform3D_param_6,
	.param .u64 kernmulUniform3D_param_7,
	.param .u64 kernmulUniform3D_param_8,
	.param .u64 kernmulUniform3D_param_9,
	.param .u64 kernmulUniform3D_param_10,
	.param .u64 kernmulUniform3D_param_11,
	.param .u64 kernmulUniform3D_param_12,
	.param .u32 kernmulUniform3D_param_13,
	.param .u32 kernmulUniform3D_param_14,
	.param .u32 kernmulUniform3D_param_15
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<28>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<42>;


	ld.param.u64 	%rd1, [kernmulUniform3D_param_0];
	ld.param.u64 	%rd2, [kernmulUniform3D_param_1];
	ld.param.u64 	%rd3, [kernmulUniform3D_param_2];
	ld.param.u64 	%rd4, [kernmulUniform3D_param_3];
	ld.param.u64 	%rd5, [kernmulUniform3D_param_4];
	ld.param.u64 	%rd6, [kernmulUniform3D_param_5];
	ld.param.u64 	%rd7, [kernmulUniform3D_param_6];
	ld.param.u64 	%rd8, [kernmulUniform3D_param_7];
	ld.param.u64 	%rd9, [kernmulUniform3D_param_8];
	ld.param.u64 	%rd10, [kernmulUniform3D_param_9];
	ld.param.u64 	%rd11, [kernmulUniform3D_param_10];
	ld.param.u64 	%rd12, [kernmulUniform3D_param_11];
	ld.param.u64 	%rd13, [kernmulUniform3D_param_12];
	ld.param.u32 	%r4, [kernmulUniform3D_param_13];
	ld.param.u32 	%r5, [kernmulUniform3D_param_14];
	ld.param.u32 	%r6, [kernmulUniform3D_param_15];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r8, %r7, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd14, %rd7;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	shl.b32 	%r18, %r17, 1;
	cvta.to.global.u64 	%rd15, %rd13;
	mul.wide.s32 	%rd16, %r18, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f1, [%rd17+4];
	shr.u32 	%r19, %r5, 31;
	add.s32 	%r20, %r5, %r19;
	shr.s32 	%r21, %r20, 1;
	setp.gt.s32 	%p6, %r2, %r21;
	sub.s32 	%r22, %r5, %r2;
	selp.b32 	%r23, %r22, %r2, %p6;
	selp.f32 	%f2, 0fBF800000, 0f3F800000, %p6;
	shr.u32 	%r24, %r6, 31;
	add.s32 	%r25, %r6, %r24;
	shr.s32 	%r26, %r25, 1;
	setp.gt.s32 	%p7, %r3, %r26;
	neg.f32 	%f3, %f2;
	sub.s32 	%r27, %r6, %r3;
	selp.b32 	%r28, %r27, %r3, %p7;
	selp.f32 	%f4, %f3, %f2, %p7;
	selp.f32 	%f5, 0fBF800000, 0f3F800000, %p7;
	add.s32 	%r29, %r21, 1;
	mad.lo.s32 	%r30, %r28, %r29, %r23;
	mad.lo.s32 	%r31, %r30, %r4, %r1;
	mul.wide.s32 	%rd18, %r31, 4;
	add.s64 	%rd19, %rd14, %rd18;
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd18;
	ld.global.nc.f32 	%f6, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd18;
	ld.global.nc.f32 	%f7, [%rd23];
	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd25, %rd24, %rd18;
	ld.global.nc.f32 	%f8, [%rd25];
	mul.f32 	%f9, %f4, %f8;
	cvta.to.global.u64 	%rd26, %rd11;
	add.s64 	%rd27, %rd26, %rd18;
	ld.global.nc.f32 	%f10, [%rd27];
	mul.f32 	%f11, %f5, %f10;
	cvta.to.global.u64 	%rd28, %rd12;
	add.s64 	%rd29, %rd28, %rd18;
	ld.global.nc.f32 	%f12, [%rd29];
	mul.f32 	%f13, %f2, %f12;
	ld.global.nc.f32 	%f14, [%rd19];
	ld.global.nc.f32 	%f15, [%rd17];
	mul.f32 	%f16, %f15, %f14;
	cvta.to.global.u64 	%rd30, %rd1;
	add.s64 	%rd31, %rd30, %rd16;
	st.global.f32 	[%rd31], %f16;
	mul.f32 	%f17, %f1, %f14;
	st.global.f32 	[%rd31+4], %f17;
	mul.f32 	%f18, %f15, %f13;
	cvta.to.global.u64 	%rd32, %rd6;
	add.s64 	%rd33, %rd32, %rd16;
	st.global.f32 	[%rd33], %f18;
	mul.f32 	%f19, %f1, %f13;
	st.global.f32 	[%rd33+4], %f19;
	mul.f32 	%f20, %f15, %f11;
	cvta.to.global.u64 	%rd34, %rd5;
	add.s64 	%rd35, %rd34, %rd16;
	st.global.f32 	[%rd35], %f20;
	mul.f32 	%f21, %f1, %f11;
	st.global.f32 	[%rd35+4], %f21;
	mul.f32 	%f22, %f15, %f6;
	cvta.to.global.u64 	%rd36, %rd2;
	add.s64 	%rd37, %rd36, %rd16;
	st.global.f32 	[%rd37], %f22;
	mul.f32 	%f23, %f1, %f6;
	st.global.f32 	[%rd37+4], %f23;
	mul.f32 	%f24, %f15, %f9;
	cvta.to.global.u64 	%rd38, %rd4;
	add.s64 	%rd39, %rd38, %rd16;
	st.global.f32 	[%rd39], %f24;
	mul.f32 	%f25, %f1, %f9;
	st.global.f32 	[%rd39+4], %f25;
	mul.f32 	%f26, %f15, %f7;
	cvta.to.global.u64 	%rd40, %rd3;
	add.s64 	%rd41, %rd40, %rd16;
	st.global.f32 	[%rd41], %f26;
	mul.f32 	%f27, %f1, %f7;
	st.global.f32 	[%rd41+4], %f27;

$L__BB0_2:
	ret;

}

`
)
