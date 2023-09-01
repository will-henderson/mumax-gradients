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

// CUDA handle for settopologicalchargelattice kernel
var settopologicalchargelattice_code cu.Function

// Stores the arguments for settopologicalchargelattice kernel invocation
type settopologicalchargelattice_args_t struct {
	arg_s     unsafe.Pointer
	arg_mx    unsafe.Pointer
	arg_my    unsafe.Pointer
	arg_mz    unsafe.Pointer
	arg_icxcy float32
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	arg_PBC   byte
	argptr    [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for settopologicalchargelattice kernel invocation
var settopologicalchargelattice_args settopologicalchargelattice_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	settopologicalchargelattice_args.argptr[0] = unsafe.Pointer(&settopologicalchargelattice_args.arg_s)
	settopologicalchargelattice_args.argptr[1] = unsafe.Pointer(&settopologicalchargelattice_args.arg_mx)
	settopologicalchargelattice_args.argptr[2] = unsafe.Pointer(&settopologicalchargelattice_args.arg_my)
	settopologicalchargelattice_args.argptr[3] = unsafe.Pointer(&settopologicalchargelattice_args.arg_mz)
	settopologicalchargelattice_args.argptr[4] = unsafe.Pointer(&settopologicalchargelattice_args.arg_icxcy)
	settopologicalchargelattice_args.argptr[5] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Nx)
	settopologicalchargelattice_args.argptr[6] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Ny)
	settopologicalchargelattice_args.argptr[7] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Nz)
	settopologicalchargelattice_args.argptr[8] = unsafe.Pointer(&settopologicalchargelattice_args.arg_PBC)
}

// Wrapper for settopologicalchargelattice CUDA kernel, asynchronous.
func k_settopologicalchargelattice_async(s unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, icxcy float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("settopologicalchargelattice")
	}

	settopologicalchargelattice_args.Lock()
	defer settopologicalchargelattice_args.Unlock()

	if settopologicalchargelattice_code == 0 {
		settopologicalchargelattice_code = fatbinLoad(settopologicalchargelattice_map, "settopologicalchargelattice")
	}

	settopologicalchargelattice_args.arg_s = s
	settopologicalchargelattice_args.arg_mx = mx
	settopologicalchargelattice_args.arg_my = my
	settopologicalchargelattice_args.arg_mz = mz
	settopologicalchargelattice_args.arg_icxcy = icxcy
	settopologicalchargelattice_args.arg_Nx = Nx
	settopologicalchargelattice_args.arg_Ny = Ny
	settopologicalchargelattice_args.arg_Nz = Nz
	settopologicalchargelattice_args.arg_PBC = PBC

	args := settopologicalchargelattice_args.argptr[:]
	cu.LaunchKernel(settopologicalchargelattice_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("settopologicalchargelattice")
	}
}

// maps compute capability on PTX code for settopologicalchargelattice kernel.
var settopologicalchargelattice_map = map[int]string{0: "",
	80: settopologicalchargelattice_ptx_80}

// settopologicalchargelattice PTX code for various compute capabilities.
const (
	settopologicalchargelattice_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	settopologicalchargelattice

.visible .entry settopologicalchargelattice(
	.param .u64 settopologicalchargelattice_param_0,
	.param .u64 settopologicalchargelattice_param_1,
	.param .u64 settopologicalchargelattice_param_2,
	.param .u64 settopologicalchargelattice_param_3,
	.param .f32 settopologicalchargelattice_param_4,
	.param .u32 settopologicalchargelattice_param_5,
	.param .u32 settopologicalchargelattice_param_6,
	.param .u32 settopologicalchargelattice_param_7,
	.param .u8 settopologicalchargelattice_param_8
)
{
	.reg .pred 	%p<85>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<296>;
	.reg .b32 	%r<181>;
	.reg .b64 	%rd<46>;


	ld.param.u8 	%rs3, [settopologicalchargelattice_param_8];
	ld.param.u64 	%rd5, [settopologicalchargelattice_param_0];
	ld.param.u64 	%rd6, [settopologicalchargelattice_param_1];
	ld.param.u64 	%rd7, [settopologicalchargelattice_param_2];
	ld.param.u64 	%rd8, [settopologicalchargelattice_param_3];
	ld.param.f32 	%f60, [settopologicalchargelattice_param_4];
	ld.param.u32 	%r49, [settopologicalchargelattice_param_5];
	ld.param.u32 	%r50, [settopologicalchargelattice_param_6];
	ld.param.u32 	%r51, [settopologicalchargelattice_param_7];
	cvta.to.global.u64 	%rd1, %rd8;
	cvta.to.global.u64 	%rd2, %rd7;
	cvta.to.global.u64 	%rd3, %rd6;
	mov.u32 	%r52, %ntid.x;
	mov.u32 	%r53, %ctaid.x;
	mov.u32 	%r54, %tid.x;
	mad.lo.s32 	%r1, %r53, %r52, %r54;
	mov.u32 	%r55, %ntid.y;
	mov.u32 	%r56, %ctaid.y;
	mov.u32 	%r57, %tid.y;
	mad.lo.s32 	%r2, %r56, %r55, %r57;
	mov.u32 	%r58, %ntid.z;
	mov.u32 	%r59, %ctaid.z;
	mov.u32 	%r60, %tid.z;
	mad.lo.s32 	%r3, %r59, %r58, %r60;
	setp.ge.s32 	%p3, %r1, %r49;
	setp.ge.s32 	%p4, %r2, %r50;
	or.pred  	%p5, %p3, %p4;
	setp.ge.s32 	%p6, %r3, %r51;
	or.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_72;

	mul.lo.s32 	%r4, %r3, %r50;
	add.s32 	%r61, %r4, %r2;
	mul.lo.s32 	%r5, %r61, %r49;
	add.s32 	%r62, %r5, %r1;
	mul.wide.s32 	%rd9, %r62, 4;
	add.s64 	%rd10, %rd3, %rd9;
	add.s64 	%rd11, %rd2, %rd9;
	add.s64 	%rd12, %rd1, %rd9;
	ld.global.nc.f32 	%f1, [%rd10];
	ld.global.nc.f32 	%f2, [%rd11];
	mul.f32 	%f61, %f2, %f2;
	fma.rn.f32 	%f62, %f1, %f1, %f61;
	ld.global.nc.f32 	%f3, [%rd12];
	fma.rn.f32 	%f63, %f3, %f3, %f62;
	setp.eq.f32 	%p8, %f63, 0f00000000;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd4, %rd13, %rd9;
	@%p8 bra 	$L__BB0_71;
	bra.uni 	$L__BB0_2;

$L__BB0_71:
	mov.u32 	%r168, 0;
	st.global.u32 	[%rd4], %r168;
	bra.uni 	$L__BB0_72;

$L__BB0_2:
	and.b16  	%rs1, %rs3, 1;
	setp.eq.s16 	%p9, %rs1, 0;
	add.s32 	%r6, %r1, 1;
	@%p9 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	add.s32 	%r65, %r49, -1;
	min.s32 	%r169, %r6, %r65;
	bra.uni 	$L__BB0_5;

$L__BB0_3:
	rem.s32 	%r63, %r6, %r49;
	add.s32 	%r64, %r63, %r49;
	rem.s32 	%r169, %r64, %r49;

$L__BB0_5:
	and.b16  	%rs2, %rs3, 2;
	setp.eq.s16 	%p10, %rs2, 0;
	add.s32 	%r10, %r2, 1;
	@%p10 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_6;

$L__BB0_7:
	add.s32 	%r68, %r50, -1;
	min.s32 	%r170, %r10, %r68;
	bra.uni 	$L__BB0_8;

$L__BB0_6:
	rem.s32 	%r66, %r10, %r50;
	add.s32 	%r67, %r66, %r50;
	rem.s32 	%r170, %r67, %r50;

$L__BB0_8:
	add.s32 	%r14, %r1, -1;
	@%p9 bra 	$L__BB0_10;
	bra.uni 	$L__BB0_9;

$L__BB0_10:
	max.s32 	%r171, %r14, 0;
	bra.uni 	$L__BB0_11;

$L__BB0_9:
	rem.s32 	%r69, %r14, %r49;
	add.s32 	%r70, %r69, %r49;
	rem.s32 	%r171, %r70, %r49;

$L__BB0_11:
	add.s32 	%r18, %r171, %r5;
	add.s32 	%r19, %r2, -1;
	add.s32 	%r20, %r169, %r5;
	@%p10 bra 	$L__BB0_13;
	bra.uni 	$L__BB0_12;

$L__BB0_13:
	max.s32 	%r172, %r19, 0;
	bra.uni 	$L__BB0_14;

$L__BB0_12:
	rem.s32 	%r71, %r19, %r50;
	add.s32 	%r72, %r71, %r50;
	rem.s32 	%r172, %r72, %r50;

$L__BB0_14:
	add.s32 	%r73, %r172, %r4;
	mad.lo.s32 	%r74, %r73, %r49, %r1;
	mul.wide.s32 	%rd14, %r20, 4;
	add.s64 	%rd15, %rd3, %rd14;
	ld.global.nc.f32 	%f4, [%rd15];
	add.s64 	%rd16, %rd2, %rd14;
	ld.global.nc.f32 	%f5, [%rd16];
	add.s64 	%rd17, %rd1, %rd14;
	ld.global.nc.f32 	%f6, [%rd17];
	add.s32 	%r75, %r170, %r4;
	mad.lo.s32 	%r76, %r75, %r49, %r1;
	mul.wide.s32 	%rd18, %r76, 4;
	add.s64 	%rd19, %rd3, %rd18;
	ld.global.nc.f32 	%f7, [%rd19];
	add.s64 	%rd20, %rd2, %rd18;
	ld.global.nc.f32 	%f8, [%rd20];
	add.s64 	%rd21, %rd1, %rd18;
	ld.global.nc.f32 	%f9, [%rd21];
	mul.wide.s32 	%rd22, %r18, 4;
	add.s64 	%rd23, %rd3, %rd22;
	ld.global.nc.f32 	%f10, [%rd23];
	add.s64 	%rd24, %rd2, %rd22;
	ld.global.nc.f32 	%f11, [%rd24];
	add.s64 	%rd25, %rd1, %rd22;
	ld.global.nc.f32 	%f12, [%rd25];
	mul.wide.s32 	%rd26, %r74, 4;
	add.s64 	%rd27, %rd3, %rd26;
	ld.global.nc.f32 	%f13, [%rd27];
	add.s64 	%rd28, %rd2, %rd26;
	ld.global.nc.f32 	%f14, [%rd28];
	add.s64 	%rd29, %rd1, %rd26;
	ld.global.nc.f32 	%f15, [%rd29];
	setp.ne.s16 	%p13, %rs1, 0;
	setp.lt.s32 	%p14, %r6, %r49;
	or.pred  	%p1, %p14, %p13;
	not.pred 	%p15, %p1;
	mov.f32 	%f291, 0f00000000;
	@%p15 bra 	$L__BB0_28;

	setp.ge.s32 	%p16, %r10, %r50;
	and.pred  	%p18, %p16, %p10;
	@%p18 bra 	$L__BB0_28;

	@%p10 bra 	$L__BB0_18;
	bra.uni 	$L__BB0_17;

$L__BB0_18:
	add.s32 	%r79, %r50, -1;
	min.s32 	%r173, %r10, %r79;
	bra.uni 	$L__BB0_19;

$L__BB0_17:
	rem.s32 	%r77, %r10, %r50;
	add.s32 	%r78, %r77, %r50;
	rem.s32 	%r173, %r78, %r50;

$L__BB0_19:
	@%p9 bra 	$L__BB0_21;
	bra.uni 	$L__BB0_20;

$L__BB0_21:
	add.s32 	%r82, %r49, -1;
	min.s32 	%r174, %r6, %r82;
	bra.uni 	$L__BB0_22;

$L__BB0_20:
	rem.s32 	%r80, %r6, %r49;
	add.s32 	%r81, %r80, %r49;
	rem.s32 	%r174, %r81, %r49;

$L__BB0_22:
	add.s32 	%r83, %r173, %r4;
	mad.lo.s32 	%r84, %r83, %r49, %r174;
	mul.wide.s32 	%rd30, %r84, 4;
	add.s64 	%rd31, %rd3, %rd30;
	add.s64 	%rd32, %rd2, %rd30;
	add.s64 	%rd33, %rd1, %rd30;
	ld.global.nc.f32 	%f66, [%rd31];
	ld.global.nc.f32 	%f67, [%rd32];
	mul.f32 	%f68, %f67, %f67;
	fma.rn.f32 	%f69, %f66, %f66, %f68;
	ld.global.nc.f32 	%f70, [%rd33];
	fma.rn.f32 	%f16, %f70, %f70, %f69;
	mul.f32 	%f71, %f6, %f8;
	mul.f32 	%f72, %f5, %f9;
	sub.f32 	%f73, %f72, %f71;
	mul.f32 	%f74, %f4, %f9;
	mul.f32 	%f75, %f6, %f7;
	sub.f32 	%f76, %f75, %f74;
	mul.f32 	%f77, %f5, %f7;
	mul.f32 	%f78, %f4, %f8;
	sub.f32 	%f79, %f78, %f77;
	mul.f32 	%f80, %f2, %f76;
	fma.rn.f32 	%f81, %f1, %f73, %f80;
	fma.rn.f32 	%f17, %f3, %f79, %f81;
	mul.f32 	%f82, %f2, %f5;
	fma.rn.f32 	%f83, %f1, %f4, %f82;
	fma.rn.f32 	%f84, %f3, %f6, %f83;
	add.f32 	%f85, %f84, 0f3F800000;
	mul.f32 	%f86, %f2, %f8;
	fma.rn.f32 	%f87, %f1, %f7, %f86;
	fma.rn.f32 	%f88, %f3, %f9, %f87;
	add.f32 	%f89, %f85, %f88;
	mul.f32 	%f90, %f5, %f8;
	fma.rn.f32 	%f91, %f4, %f7, %f90;
	fma.rn.f32 	%f92, %f6, %f9, %f91;
	add.f32 	%f18, %f92, %f89;
	abs.f32 	%f19, %f18;
	abs.f32 	%f20, %f17;
	setp.eq.f32 	%p21, %f19, 0f00000000;
	setp.eq.f32 	%p22, %f20, 0f00000000;
	and.pred  	%p23, %p21, %p22;
	@%p23 bra 	$L__BB0_26;
	bra.uni 	$L__BB0_23;

$L__BB0_26:
	mov.b32 	%r95, %f18;
	shr.s32 	%r96, %r95, 31;
	and.b32  	%r97, %r96, 1078530011;
	mov.b32 	%r98, %f17;
	and.b32  	%r99, %r98, -2147483648;
	or.b32  	%r100, %r99, %r97;
	mov.b32 	%f288, %r100;
	bra.uni 	$L__BB0_27;

$L__BB0_23:
	setp.eq.f32 	%p24, %f19, 0f7F800000;
	setp.eq.f32 	%p25, %f20, 0f7F800000;
	and.pred  	%p26, %p24, %p25;
	@%p26 bra 	$L__BB0_25;
	bra.uni 	$L__BB0_24;

$L__BB0_25:
	mov.b32 	%r90, %f18;
	setp.lt.s32 	%p30, %r90, 0;
	selp.b32 	%r91, 1075235812, 1061752795, %p30;
	mov.b32 	%r92, %f17;
	and.b32  	%r93, %r92, -2147483648;
	or.b32  	%r94, %r93, %r91;
	mov.b32 	%f288, %r94;
	bra.uni 	$L__BB0_27;

$L__BB0_24:
	max.f32 	%f93, %f20, %f19;
	min.f32 	%f94, %f20, %f19;
	div.rn.f32 	%f95, %f94, %f93;
	mul.rn.f32 	%f96, %f95, %f95;
	mov.f32 	%f97, 0fC0B59883;
	mov.f32 	%f98, 0fBF52C7EA;
	fma.rn.f32 	%f99, %f96, %f98, %f97;
	mov.f32 	%f100, 0fC0D21907;
	fma.rn.f32 	%f101, %f99, %f96, %f100;
	mul.f32 	%f102, %f96, %f101;
	mul.f32 	%f103, %f95, %f102;
	add.f32 	%f104, %f96, 0f41355DC0;
	mov.f32 	%f105, 0f41E6BD60;
	fma.rn.f32 	%f106, %f104, %f96, %f105;
	mov.f32 	%f107, 0f419D92C8;
	fma.rn.f32 	%f108, %f106, %f96, %f107;
	rcp.rn.f32 	%f109, %f108;
	fma.rn.f32 	%f110, %f103, %f109, %f95;
	mov.f32 	%f111, 0f3FC90FDB;
	sub.f32 	%f112, %f111, %f110;
	setp.gt.f32 	%p27, %f20, %f19;
	selp.f32 	%f113, %f112, %f110, %p27;
	mov.b32 	%r85, %f18;
	setp.lt.s32 	%p28, %r85, 0;
	mov.f32 	%f114, 0f40490FDB;
	sub.f32 	%f115, %f114, %f113;
	selp.f32 	%f116, %f115, %f113, %p28;
	mov.b32 	%r86, %f116;
	mov.b32 	%r87, %f17;
	and.b32  	%r88, %r87, -2147483648;
	or.b32  	%r89, %r88, %r86;
	mov.b32 	%f117, %r89;
	add.f32 	%f118, %f19, %f20;
	setp.le.f32 	%p29, %f118, 0f7F800000;
	selp.f32 	%f288, %f117, %f118, %p29;

$L__BB0_27:
	add.f32 	%f119, %f288, %f288;
	setp.eq.f32 	%p31, %f16, 0f00000000;
	selp.f32 	%f120, 0f3F800000, 0f3F000000, %p31;
	fma.rn.f32 	%f291, %f120, %f119, 0f00000000;

$L__BB0_28:
	setp.gt.s32 	%p32, %r1, 0;
	or.pred  	%p2, %p32, %p13;
	not.pred 	%p34, %p2;
	@%p34 bra 	$L__BB0_42;

	setp.ge.s32 	%p35, %r10, %r50;
	and.pred  	%p37, %p35, %p10;
	@%p37 bra 	$L__BB0_42;

	@%p10 bra 	$L__BB0_32;
	bra.uni 	$L__BB0_31;

$L__BB0_32:
	add.s32 	%r103, %r50, -1;
	min.s32 	%r175, %r10, %r103;
	bra.uni 	$L__BB0_33;

$L__BB0_31:
	rem.s32 	%r101, %r10, %r50;
	add.s32 	%r102, %r101, %r50;
	rem.s32 	%r175, %r102, %r50;

$L__BB0_33:
	@%p9 bra 	$L__BB0_35;
	bra.uni 	$L__BB0_34;

$L__BB0_35:
	max.s32 	%r176, %r14, 0;
	bra.uni 	$L__BB0_36;

$L__BB0_34:
	rem.s32 	%r104, %r14, %r49;
	add.s32 	%r105, %r104, %r49;
	rem.s32 	%r176, %r105, %r49;

$L__BB0_36:
	add.s32 	%r106, %r175, %r4;
	mad.lo.s32 	%r107, %r106, %r49, %r176;
	mul.wide.s32 	%rd34, %r107, 4;
	add.s64 	%rd35, %rd3, %rd34;
	add.s64 	%rd36, %rd2, %rd34;
	add.s64 	%rd37, %rd1, %rd34;
	ld.global.nc.f32 	%f121, [%rd35];
	ld.global.nc.f32 	%f122, [%rd36];
	mul.f32 	%f123, %f122, %f122;
	fma.rn.f32 	%f124, %f121, %f121, %f123;
	ld.global.nc.f32 	%f125, [%rd37];
	fma.rn.f32 	%f27, %f125, %f125, %f124;
	mul.f32 	%f126, %f9, %f11;
	mul.f32 	%f127, %f8, %f12;
	sub.f32 	%f128, %f127, %f126;
	mul.f32 	%f129, %f7, %f12;
	mul.f32 	%f130, %f9, %f10;
	sub.f32 	%f131, %f130, %f129;
	mul.f32 	%f132, %f8, %f10;
	mul.f32 	%f133, %f7, %f11;
	sub.f32 	%f134, %f133, %f132;
	mul.f32 	%f135, %f2, %f131;
	fma.rn.f32 	%f136, %f1, %f128, %f135;
	fma.rn.f32 	%f28, %f3, %f134, %f136;
	mul.f32 	%f137, %f2, %f8;
	fma.rn.f32 	%f138, %f1, %f7, %f137;
	fma.rn.f32 	%f139, %f3, %f9, %f138;
	add.f32 	%f140, %f139, 0f3F800000;
	mul.f32 	%f141, %f2, %f11;
	fma.rn.f32 	%f142, %f1, %f10, %f141;
	fma.rn.f32 	%f143, %f3, %f12, %f142;
	add.f32 	%f144, %f140, %f143;
	mul.f32 	%f145, %f8, %f11;
	fma.rn.f32 	%f146, %f7, %f10, %f145;
	fma.rn.f32 	%f147, %f9, %f12, %f146;
	add.f32 	%f29, %f147, %f144;
	abs.f32 	%f30, %f29;
	abs.f32 	%f31, %f28;
	setp.eq.f32 	%p40, %f30, 0f00000000;
	setp.eq.f32 	%p41, %f31, 0f00000000;
	and.pred  	%p42, %p40, %p41;
	@%p42 bra 	$L__BB0_40;
	bra.uni 	$L__BB0_37;

$L__BB0_40:
	mov.b32 	%r118, %f29;
	shr.s32 	%r119, %r118, 31;
	and.b32  	%r120, %r119, 1078530011;
	mov.b32 	%r121, %f28;
	and.b32  	%r122, %r121, -2147483648;
	or.b32  	%r123, %r122, %r120;
	mov.b32 	%f290, %r123;
	bra.uni 	$L__BB0_41;

$L__BB0_37:
	setp.eq.f32 	%p43, %f30, 0f7F800000;
	setp.eq.f32 	%p44, %f31, 0f7F800000;
	and.pred  	%p45, %p43, %p44;
	@%p45 bra 	$L__BB0_39;
	bra.uni 	$L__BB0_38;

$L__BB0_39:
	mov.b32 	%r113, %f29;
	setp.lt.s32 	%p49, %r113, 0;
	selp.b32 	%r114, 1075235812, 1061752795, %p49;
	mov.b32 	%r115, %f28;
	and.b32  	%r116, %r115, -2147483648;
	or.b32  	%r117, %r116, %r114;
	mov.b32 	%f290, %r117;
	bra.uni 	$L__BB0_41;

$L__BB0_38:
	max.f32 	%f148, %f31, %f30;
	min.f32 	%f149, %f31, %f30;
	div.rn.f32 	%f150, %f149, %f148;
	mul.rn.f32 	%f151, %f150, %f150;
	mov.f32 	%f152, 0fC0B59883;
	mov.f32 	%f153, 0fBF52C7EA;
	fma.rn.f32 	%f154, %f151, %f153, %f152;
	mov.f32 	%f155, 0fC0D21907;
	fma.rn.f32 	%f156, %f154, %f151, %f155;
	mul.f32 	%f157, %f151, %f156;
	mul.f32 	%f158, %f150, %f157;
	add.f32 	%f159, %f151, 0f41355DC0;
	mov.f32 	%f160, 0f41E6BD60;
	fma.rn.f32 	%f161, %f159, %f151, %f160;
	mov.f32 	%f162, 0f419D92C8;
	fma.rn.f32 	%f163, %f161, %f151, %f162;
	rcp.rn.f32 	%f164, %f163;
	fma.rn.f32 	%f165, %f158, %f164, %f150;
	mov.f32 	%f166, 0f3FC90FDB;
	sub.f32 	%f167, %f166, %f165;
	setp.gt.f32 	%p46, %f31, %f30;
	selp.f32 	%f168, %f167, %f165, %p46;
	mov.b32 	%r108, %f29;
	setp.lt.s32 	%p47, %r108, 0;
	mov.f32 	%f169, 0f40490FDB;
	sub.f32 	%f170, %f169, %f168;
	selp.f32 	%f171, %f170, %f168, %p47;
	mov.b32 	%r109, %f171;
	mov.b32 	%r110, %f28;
	and.b32  	%r111, %r110, -2147483648;
	or.b32  	%r112, %r111, %r109;
	mov.b32 	%f172, %r112;
	add.f32 	%f173, %f30, %f31;
	setp.le.f32 	%p48, %f173, 0f7F800000;
	selp.f32 	%f290, %f172, %f173, %p48;

$L__BB0_41:
	add.f32 	%f174, %f290, %f290;
	setp.eq.f32 	%p50, %f27, 0f00000000;
	selp.f32 	%f175, 0f3F800000, 0f3F000000, %p50;
	fma.rn.f32 	%f291, %f175, %f174, %f291;

$L__BB0_42:
	@%p34 bra 	$L__BB0_56;

	setp.lt.s32 	%p52, %r2, 1;
	and.pred  	%p54, %p52, %p10;
	@%p54 bra 	$L__BB0_56;

	@%p10 bra 	$L__BB0_46;
	bra.uni 	$L__BB0_45;

$L__BB0_46:
	max.s32 	%r177, %r19, 0;
	bra.uni 	$L__BB0_47;

$L__BB0_45:
	rem.s32 	%r124, %r19, %r50;
	add.s32 	%r125, %r124, %r50;
	rem.s32 	%r177, %r125, %r50;

$L__BB0_47:
	@%p9 bra 	$L__BB0_49;
	bra.uni 	$L__BB0_48;

$L__BB0_49:
	max.s32 	%r178, %r14, 0;
	bra.uni 	$L__BB0_50;

$L__BB0_48:
	rem.s32 	%r126, %r14, %r49;
	add.s32 	%r127, %r126, %r49;
	rem.s32 	%r178, %r127, %r49;

$L__BB0_50:
	add.s32 	%r128, %r177, %r4;
	mad.lo.s32 	%r129, %r128, %r49, %r178;
	mul.wide.s32 	%rd38, %r129, 4;
	add.s64 	%rd39, %rd3, %rd38;
	add.s64 	%rd40, %rd2, %rd38;
	add.s64 	%rd41, %rd1, %rd38;
	ld.global.nc.f32 	%f176, [%rd39];
	ld.global.nc.f32 	%f177, [%rd40];
	mul.f32 	%f178, %f177, %f177;
	fma.rn.f32 	%f179, %f176, %f176, %f178;
	ld.global.nc.f32 	%f180, [%rd41];
	fma.rn.f32 	%f38, %f180, %f180, %f179;
	mul.f32 	%f181, %f12, %f14;
	mul.f32 	%f182, %f11, %f15;
	sub.f32 	%f183, %f182, %f181;
	mul.f32 	%f184, %f10, %f15;
	mul.f32 	%f185, %f12, %f13;
	sub.f32 	%f186, %f185, %f184;
	mul.f32 	%f187, %f11, %f13;
	mul.f32 	%f188, %f10, %f14;
	sub.f32 	%f189, %f188, %f187;
	mul.f32 	%f190, %f2, %f186;
	fma.rn.f32 	%f191, %f1, %f183, %f190;
	fma.rn.f32 	%f39, %f3, %f189, %f191;
	mul.f32 	%f192, %f2, %f11;
	fma.rn.f32 	%f193, %f1, %f10, %f192;
	fma.rn.f32 	%f194, %f3, %f12, %f193;
	add.f32 	%f195, %f194, 0f3F800000;
	mul.f32 	%f196, %f2, %f14;
	fma.rn.f32 	%f197, %f1, %f13, %f196;
	fma.rn.f32 	%f198, %f3, %f15, %f197;
	add.f32 	%f199, %f195, %f198;
	mul.f32 	%f200, %f11, %f14;
	fma.rn.f32 	%f201, %f10, %f13, %f200;
	fma.rn.f32 	%f202, %f12, %f15, %f201;
	add.f32 	%f40, %f202, %f199;
	abs.f32 	%f41, %f40;
	abs.f32 	%f42, %f39;
	setp.eq.f32 	%p57, %f41, 0f00000000;
	setp.eq.f32 	%p58, %f42, 0f00000000;
	and.pred  	%p59, %p57, %p58;
	@%p59 bra 	$L__BB0_54;
	bra.uni 	$L__BB0_51;

$L__BB0_54:
	mov.b32 	%r140, %f40;
	shr.s32 	%r141, %r140, 31;
	and.b32  	%r142, %r141, 1078530011;
	mov.b32 	%r143, %f39;
	and.b32  	%r144, %r143, -2147483648;
	or.b32  	%r145, %r144, %r142;
	mov.b32 	%f292, %r145;
	bra.uni 	$L__BB0_55;

$L__BB0_51:
	setp.eq.f32 	%p60, %f41, 0f7F800000;
	setp.eq.f32 	%p61, %f42, 0f7F800000;
	and.pred  	%p62, %p60, %p61;
	@%p62 bra 	$L__BB0_53;
	bra.uni 	$L__BB0_52;

$L__BB0_53:
	mov.b32 	%r135, %f40;
	setp.lt.s32 	%p66, %r135, 0;
	selp.b32 	%r136, 1075235812, 1061752795, %p66;
	mov.b32 	%r137, %f39;
	and.b32  	%r138, %r137, -2147483648;
	or.b32  	%r139, %r138, %r136;
	mov.b32 	%f292, %r139;
	bra.uni 	$L__BB0_55;

$L__BB0_52:
	max.f32 	%f203, %f42, %f41;
	min.f32 	%f204, %f42, %f41;
	div.rn.f32 	%f205, %f204, %f203;
	mul.rn.f32 	%f206, %f205, %f205;
	mov.f32 	%f207, 0fC0B59883;
	mov.f32 	%f208, 0fBF52C7EA;
	fma.rn.f32 	%f209, %f206, %f208, %f207;
	mov.f32 	%f210, 0fC0D21907;
	fma.rn.f32 	%f211, %f209, %f206, %f210;
	mul.f32 	%f212, %f206, %f211;
	mul.f32 	%f213, %f205, %f212;
	add.f32 	%f214, %f206, 0f41355DC0;
	mov.f32 	%f215, 0f41E6BD60;
	fma.rn.f32 	%f216, %f214, %f206, %f215;
	mov.f32 	%f217, 0f419D92C8;
	fma.rn.f32 	%f218, %f216, %f206, %f217;
	rcp.rn.f32 	%f219, %f218;
	fma.rn.f32 	%f220, %f213, %f219, %f205;
	mov.f32 	%f221, 0f3FC90FDB;
	sub.f32 	%f222, %f221, %f220;
	setp.gt.f32 	%p63, %f42, %f41;
	selp.f32 	%f223, %f222, %f220, %p63;
	mov.b32 	%r130, %f40;
	setp.lt.s32 	%p64, %r130, 0;
	mov.f32 	%f224, 0f40490FDB;
	sub.f32 	%f225, %f224, %f223;
	selp.f32 	%f226, %f225, %f223, %p64;
	mov.b32 	%r131, %f226;
	mov.b32 	%r132, %f39;
	and.b32  	%r133, %r132, -2147483648;
	or.b32  	%r134, %r133, %r131;
	mov.b32 	%f227, %r134;
	add.f32 	%f228, %f41, %f42;
	setp.le.f32 	%p65, %f228, 0f7F800000;
	selp.f32 	%f292, %f227, %f228, %p65;

$L__BB0_55:
	add.f32 	%f229, %f292, %f292;
	setp.eq.f32 	%p67, %f38, 0f00000000;
	selp.f32 	%f230, 0f3F800000, 0f3F000000, %p67;
	fma.rn.f32 	%f291, %f230, %f229, %f291;

$L__BB0_56:
	@%p15 bra 	$L__BB0_70;

	setp.lt.s32 	%p69, %r2, 1;
	and.pred  	%p71, %p69, %p10;
	@%p71 bra 	$L__BB0_70;

	@%p10 bra 	$L__BB0_60;
	bra.uni 	$L__BB0_59;

$L__BB0_60:
	max.s32 	%r179, %r19, 0;
	bra.uni 	$L__BB0_61;

$L__BB0_59:
	rem.s32 	%r146, %r19, %r50;
	add.s32 	%r147, %r146, %r50;
	rem.s32 	%r179, %r147, %r50;

$L__BB0_61:
	add.s32 	%r45, %r179, %r4;
	@%p9 bra 	$L__BB0_63;
	bra.uni 	$L__BB0_62;

$L__BB0_63:
	add.s32 	%r150, %r49, -1;
	min.s32 	%r180, %r6, %r150;
	bra.uni 	$L__BB0_64;

$L__BB0_62:
	rem.s32 	%r148, %r6, %r49;
	add.s32 	%r149, %r148, %r49;
	rem.s32 	%r180, %r149, %r49;

$L__BB0_64:
	mad.lo.s32 	%r151, %r45, %r49, %r180;
	mul.wide.s32 	%rd42, %r151, 4;
	add.s64 	%rd43, %rd3, %rd42;
	add.s64 	%rd44, %rd2, %rd42;
	add.s64 	%rd45, %rd1, %rd42;
	ld.global.nc.f32 	%f231, [%rd43];
	ld.global.nc.f32 	%f232, [%rd44];
	mul.f32 	%f233, %f232, %f232;
	fma.rn.f32 	%f234, %f231, %f231, %f233;
	ld.global.nc.f32 	%f235, [%rd45];
	fma.rn.f32 	%f49, %f235, %f235, %f234;
	mul.f32 	%f236, %f5, %f15;
	mul.f32 	%f237, %f6, %f14;
	sub.f32 	%f238, %f237, %f236;
	mul.f32 	%f239, %f6, %f13;
	mul.f32 	%f240, %f4, %f15;
	sub.f32 	%f241, %f240, %f239;
	mul.f32 	%f242, %f4, %f14;
	mul.f32 	%f243, %f5, %f13;
	sub.f32 	%f244, %f243, %f242;
	mul.f32 	%f245, %f2, %f241;
	fma.rn.f32 	%f246, %f1, %f238, %f245;
	fma.rn.f32 	%f50, %f3, %f244, %f246;
	mul.f32 	%f247, %f2, %f14;
	fma.rn.f32 	%f248, %f1, %f13, %f247;
	fma.rn.f32 	%f249, %f3, %f15, %f248;
	add.f32 	%f250, %f249, 0f3F800000;
	mul.f32 	%f251, %f2, %f5;
	fma.rn.f32 	%f252, %f1, %f4, %f251;
	fma.rn.f32 	%f253, %f3, %f6, %f252;
	add.f32 	%f254, %f253, %f250;
	mul.f32 	%f255, %f5, %f14;
	fma.rn.f32 	%f256, %f4, %f13, %f255;
	fma.rn.f32 	%f257, %f6, %f15, %f256;
	add.f32 	%f51, %f257, %f254;
	abs.f32 	%f52, %f51;
	abs.f32 	%f53, %f50;
	setp.eq.f32 	%p74, %f52, 0f00000000;
	setp.eq.f32 	%p75, %f53, 0f00000000;
	and.pred  	%p76, %p74, %p75;
	@%p76 bra 	$L__BB0_68;
	bra.uni 	$L__BB0_65;

$L__BB0_68:
	mov.b32 	%r162, %f51;
	shr.s32 	%r163, %r162, 31;
	and.b32  	%r164, %r163, 1078530011;
	mov.b32 	%r165, %f50;
	and.b32  	%r166, %r165, -2147483648;
	or.b32  	%r167, %r164, %r166;
	mov.b32 	%f294, %r167;
	bra.uni 	$L__BB0_69;

$L__BB0_65:
	setp.eq.f32 	%p77, %f52, 0f7F800000;
	setp.eq.f32 	%p78, %f53, 0f7F800000;
	and.pred  	%p79, %p77, %p78;
	@%p79 bra 	$L__BB0_67;
	bra.uni 	$L__BB0_66;

$L__BB0_67:
	mov.b32 	%r157, %f51;
	setp.lt.s32 	%p83, %r157, 0;
	selp.b32 	%r158, 1075235812, 1061752795, %p83;
	mov.b32 	%r159, %f50;
	and.b32  	%r160, %r159, -2147483648;
	or.b32  	%r161, %r158, %r160;
	mov.b32 	%f294, %r161;
	bra.uni 	$L__BB0_69;

$L__BB0_66:
	max.f32 	%f258, %f53, %f52;
	min.f32 	%f259, %f53, %f52;
	div.rn.f32 	%f260, %f259, %f258;
	mul.rn.f32 	%f261, %f260, %f260;
	mov.f32 	%f262, 0fC0B59883;
	mov.f32 	%f263, 0fBF52C7EA;
	fma.rn.f32 	%f264, %f261, %f263, %f262;
	mov.f32 	%f265, 0fC0D21907;
	fma.rn.f32 	%f266, %f264, %f261, %f265;
	mul.f32 	%f267, %f261, %f266;
	mul.f32 	%f268, %f260, %f267;
	add.f32 	%f269, %f261, 0f41355DC0;
	mov.f32 	%f270, 0f41E6BD60;
	fma.rn.f32 	%f271, %f269, %f261, %f270;
	mov.f32 	%f272, 0f419D92C8;
	fma.rn.f32 	%f273, %f271, %f261, %f272;
	rcp.rn.f32 	%f274, %f273;
	fma.rn.f32 	%f275, %f268, %f274, %f260;
	mov.f32 	%f276, 0f3FC90FDB;
	sub.f32 	%f277, %f276, %f275;
	setp.gt.f32 	%p80, %f53, %f52;
	selp.f32 	%f278, %f277, %f275, %p80;
	mov.b32 	%r152, %f51;
	setp.lt.s32 	%p81, %r152, 0;
	mov.f32 	%f279, 0f40490FDB;
	sub.f32 	%f280, %f279, %f278;
	selp.f32 	%f281, %f280, %f278, %p81;
	mov.b32 	%r153, %f281;
	mov.b32 	%r154, %f50;
	and.b32  	%r155, %r154, -2147483648;
	or.b32  	%r156, %r155, %r153;
	mov.b32 	%f282, %r156;
	add.f32 	%f283, %f52, %f53;
	setp.le.f32 	%p82, %f283, 0f7F800000;
	selp.f32 	%f294, %f282, %f283, %p82;

$L__BB0_69:
	add.f32 	%f284, %f294, %f294;
	setp.eq.f32 	%p84, %f49, 0f00000000;
	selp.f32 	%f285, 0f3F800000, 0f3F000000, %p84;
	fma.rn.f32 	%f291, %f285, %f284, %f291;

$L__BB0_70:
	mul.f32 	%f286, %f291, %f60;
	st.global.f32 	[%rd4], %f286;

$L__BB0_72:
	ret;

}

`
)
