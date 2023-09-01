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

// CUDA handle for settopologicalcharge kernel
var settopologicalcharge_code cu.Function

// Stores the arguments for settopologicalcharge kernel invocation
type settopologicalcharge_args_t struct {
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

// Stores the arguments for settopologicalcharge kernel invocation
var settopologicalcharge_args settopologicalcharge_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	settopologicalcharge_args.argptr[0] = unsafe.Pointer(&settopologicalcharge_args.arg_s)
	settopologicalcharge_args.argptr[1] = unsafe.Pointer(&settopologicalcharge_args.arg_mx)
	settopologicalcharge_args.argptr[2] = unsafe.Pointer(&settopologicalcharge_args.arg_my)
	settopologicalcharge_args.argptr[3] = unsafe.Pointer(&settopologicalcharge_args.arg_mz)
	settopologicalcharge_args.argptr[4] = unsafe.Pointer(&settopologicalcharge_args.arg_icxcy)
	settopologicalcharge_args.argptr[5] = unsafe.Pointer(&settopologicalcharge_args.arg_Nx)
	settopologicalcharge_args.argptr[6] = unsafe.Pointer(&settopologicalcharge_args.arg_Ny)
	settopologicalcharge_args.argptr[7] = unsafe.Pointer(&settopologicalcharge_args.arg_Nz)
	settopologicalcharge_args.argptr[8] = unsafe.Pointer(&settopologicalcharge_args.arg_PBC)
}

// Wrapper for settopologicalcharge CUDA kernel, asynchronous.
func k_settopologicalcharge_async(s unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, icxcy float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("settopologicalcharge")
	}

	settopologicalcharge_args.Lock()
	defer settopologicalcharge_args.Unlock()

	if settopologicalcharge_code == 0 {
		settopologicalcharge_code = fatbinLoad(settopologicalcharge_map, "settopologicalcharge")
	}

	settopologicalcharge_args.arg_s = s
	settopologicalcharge_args.arg_mx = mx
	settopologicalcharge_args.arg_my = my
	settopologicalcharge_args.arg_mz = mz
	settopologicalcharge_args.arg_icxcy = icxcy
	settopologicalcharge_args.arg_Nx = Nx
	settopologicalcharge_args.arg_Ny = Ny
	settopologicalcharge_args.arg_Nz = Nz
	settopologicalcharge_args.arg_PBC = PBC

	args := settopologicalcharge_args.argptr[:]
	cu.LaunchKernel(settopologicalcharge_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("settopologicalcharge")
	}
}

// maps compute capability on PTX code for settopologicalcharge kernel.
var settopologicalcharge_map = map[int]string{0: "",
	80: settopologicalcharge_ptx_80}

// settopologicalcharge PTX code for various compute capabilities.
const (
	settopologicalcharge_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	settopologicalcharge

.visible .entry settopologicalcharge(
	.param .u64 settopologicalcharge_param_0,
	.param .u64 settopologicalcharge_param_1,
	.param .u64 settopologicalcharge_param_2,
	.param .u64 settopologicalcharge_param_3,
	.param .f32 settopologicalcharge_param_4,
	.param .u32 settopologicalcharge_param_5,
	.param .u32 settopologicalcharge_param_6,
	.param .u32 settopologicalcharge_param_7,
	.param .u8 settopologicalcharge_param_8
)
{
	.reg .pred 	%p<79>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<315>;
	.reg .b32 	%r<93>;
	.reg .b64 	%rd<46>;


	ld.param.u8 	%rs3, [settopologicalcharge_param_8];
	ld.param.u64 	%rd5, [settopologicalcharge_param_0];
	ld.param.u64 	%rd6, [settopologicalcharge_param_1];
	ld.param.u64 	%rd7, [settopologicalcharge_param_2];
	ld.param.u64 	%rd8, [settopologicalcharge_param_3];
	ld.param.f32 	%f138, [settopologicalcharge_param_4];
	ld.param.u32 	%r40, [settopologicalcharge_param_5];
	ld.param.u32 	%r41, [settopologicalcharge_param_6];
	ld.param.u32 	%r42, [settopologicalcharge_param_7];
	cvta.to.global.u64 	%rd1, %rd8;
	cvta.to.global.u64 	%rd2, %rd7;
	cvta.to.global.u64 	%rd3, %rd6;
	mov.u32 	%r43, %ntid.x;
	mov.u32 	%r44, %ctaid.x;
	mov.u32 	%r45, %tid.x;
	mad.lo.s32 	%r1, %r44, %r43, %r45;
	mov.u32 	%r46, %ntid.y;
	mov.u32 	%r47, %ctaid.y;
	mov.u32 	%r48, %tid.y;
	mad.lo.s32 	%r2, %r47, %r46, %r48;
	mov.u32 	%r49, %ntid.z;
	mov.u32 	%r50, %ctaid.z;
	mov.u32 	%r51, %tid.z;
	mad.lo.s32 	%r3, %r50, %r49, %r51;
	setp.ge.s32 	%p1, %r1, %r40;
	setp.ge.s32 	%p2, %r2, %r41;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r42;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_72;

	mul.lo.s32 	%r4, %r3, %r41;
	add.s32 	%r52, %r4, %r2;
	mul.lo.s32 	%r5, %r52, %r40;
	add.s32 	%r53, %r5, %r1;
	mul.wide.s32 	%rd9, %r53, 4;
	add.s64 	%rd10, %rd3, %rd9;
	add.s64 	%rd11, %rd2, %rd9;
	add.s64 	%rd12, %rd1, %rd9;
	ld.global.nc.f32 	%f1, [%rd10];
	ld.global.nc.f32 	%f2, [%rd11];
	mul.f32 	%f139, %f2, %f2;
	fma.rn.f32 	%f140, %f1, %f1, %f139;
	ld.global.nc.f32 	%f3, [%rd12];
	fma.rn.f32 	%f141, %f3, %f3, %f140;
	setp.eq.f32 	%p6, %f141, 0f00000000;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd4, %rd13, %rd9;
	@%p6 bra 	$L__BB0_71;
	bra.uni 	$L__BB0_2;

$L__BB0_71:
	mov.u32 	%r84, 0;
	st.global.u32 	[%rd4], %r84;
	bra.uni 	$L__BB0_72;

$L__BB0_2:
	and.b16  	%rs1, %rs3, 1;
	setp.eq.s16 	%p7, %rs1, 0;
	add.s32 	%r6, %r1, -2;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	max.s32 	%r85, %r6, 0;
	bra.uni 	$L__BB0_5;

$L__BB0_3:
	rem.s32 	%r54, %r6, %r40;
	add.s32 	%r55, %r54, %r40;
	rem.s32 	%r85, %r55, %r40;

$L__BB0_5:
	setp.lt.s32 	%p9, %r1, 2;
	mov.f32 	%f7, 0f00000000;
	and.pred  	%p10, %p9, %p7;
	mov.f32 	%f8, %f7;
	mov.f32 	%f9, %f7;
	@%p10 bra 	$L__BB0_7;

	add.s32 	%r56, %r85, %r5;
	mul.wide.s32 	%rd14, %r56, 4;
	add.s64 	%rd15, %rd3, %rd14;
	add.s64 	%rd16, %rd2, %rd14;
	add.s64 	%rd17, %rd1, %rd14;
	ld.global.nc.f32 	%f9, [%rd17];
	ld.global.nc.f32 	%f8, [%rd16];
	ld.global.nc.f32 	%f7, [%rd15];

$L__BB0_7:
	add.s32 	%r10, %r1, -1;
	@%p7 bra 	$L__BB0_9;
	bra.uni 	$L__BB0_8;

$L__BB0_9:
	max.s32 	%r86, %r10, 0;
	bra.uni 	$L__BB0_10;

$L__BB0_8:
	rem.s32 	%r57, %r10, %r40;
	add.s32 	%r58, %r57, %r40;
	rem.s32 	%r86, %r58, %r40;

$L__BB0_10:
	setp.lt.s32 	%p12, %r1, 1;
	mov.f32 	%f40, 0f00000000;
	and.pred  	%p14, %p12, %p7;
	mov.f32 	%f39, %f40;
	mov.f32 	%f38, %f40;
	@%p14 bra 	$L__BB0_12;

	add.s32 	%r59, %r86, %r5;
	mul.wide.s32 	%rd18, %r59, 4;
	add.s64 	%rd19, %rd3, %rd18;
	add.s64 	%rd20, %rd2, %rd18;
	add.s64 	%rd21, %rd1, %rd18;
	ld.global.nc.f32 	%f38, [%rd21];
	ld.global.nc.f32 	%f39, [%rd20];
	ld.global.nc.f32 	%f40, [%rd19];

$L__BB0_12:
	add.s32 	%r14, %r1, 1;
	@%p7 bra 	$L__BB0_14;
	bra.uni 	$L__BB0_13;

$L__BB0_14:
	add.s32 	%r62, %r40, -1;
	min.s32 	%r87, %r14, %r62;
	bra.uni 	$L__BB0_15;

$L__BB0_13:
	rem.s32 	%r60, %r14, %r40;
	add.s32 	%r61, %r60, %r40;
	rem.s32 	%r87, %r61, %r40;

$L__BB0_15:
	setp.ge.s32 	%p16, %r14, %r40;
	mov.f32 	%f19, 0f00000000;
	and.pred  	%p18, %p16, %p7;
	mov.f32 	%f20, %f19;
	mov.f32 	%f21, %f19;
	@%p18 bra 	$L__BB0_17;

	add.s32 	%r63, %r87, %r5;
	mul.wide.s32 	%rd22, %r63, 4;
	add.s64 	%rd23, %rd3, %rd22;
	add.s64 	%rd24, %rd2, %rd22;
	add.s64 	%rd25, %rd1, %rd22;
	ld.global.nc.f32 	%f21, [%rd25];
	ld.global.nc.f32 	%f20, [%rd24];
	ld.global.nc.f32 	%f19, [%rd23];

$L__BB0_17:
	add.s32 	%r18, %r1, 2;
	@%p7 bra 	$L__BB0_19;
	bra.uni 	$L__BB0_18;

$L__BB0_19:
	add.s32 	%r66, %r40, -1;
	min.s32 	%r88, %r18, %r66;
	bra.uni 	$L__BB0_20;

$L__BB0_18:
	rem.s32 	%r64, %r18, %r40;
	add.s32 	%r65, %r64, %r40;
	rem.s32 	%r88, %r65, %r40;

$L__BB0_20:
	add.s32 	%r22, %r88, %r5;
	setp.ge.s32 	%p20, %r18, %r40;
	mov.f32 	%f25, 0f00000000;
	and.pred  	%p22, %p20, %p7;
	mov.f32 	%f26, %f25;
	mov.f32 	%f27, %f25;
	@%p22 bra 	$L__BB0_22;

	mul.wide.s32 	%rd26, %r22, 4;
	add.s64 	%rd27, %rd3, %rd26;
	add.s64 	%rd28, %rd2, %rd26;
	add.s64 	%rd29, %rd1, %rd26;
	ld.global.nc.f32 	%f27, [%rd29];
	ld.global.nc.f32 	%f26, [%rd28];
	ld.global.nc.f32 	%f25, [%rd27];

$L__BB0_22:
	mul.f32 	%f154, %f20, %f20;
	fma.rn.f32 	%f155, %f19, %f19, %f154;
	fma.rn.f32 	%f31, %f21, %f21, %f155;
	setp.eq.f32 	%p23, %f31, 0f00000000;
	@%p23 bra 	$L__BB0_23;
	bra.uni 	$L__BB0_24;

$L__BB0_23:
	mul.f32 	%f159, %f39, %f39;
	fma.rn.f32 	%f160, %f40, %f40, %f159;
	fma.rn.f32 	%f161, %f38, %f38, %f160;
	setp.eq.f32 	%p24, %f161, 0f00000000;
	mov.f32 	%f294, 0f00000000;
	mov.f32 	%f295, %f294;
	mov.f32 	%f296, %f294;
	@%p24 bra 	$L__BB0_36;

$L__BB0_24:
	mul.f32 	%f162, %f8, %f8;
	fma.rn.f32 	%f163, %f7, %f7, %f162;
	fma.rn.f32 	%f44, %f9, %f9, %f163;
	setp.neu.f32 	%p25, %f44, 0f00000000;
	mul.f32 	%f164, %f26, %f26;
	fma.rn.f32 	%f165, %f25, %f25, %f164;
	fma.rn.f32 	%f48, %f27, %f27, %f165;
	setp.neu.f32 	%p26, %f48, 0f00000000;
	and.pred  	%p27, %p25, %p26;
	or.pred  	%p29, %p23, %p27;
	@%p29 bra 	$L__BB0_26;

	mul.f32 	%f166, %f39, %f39;
	fma.rn.f32 	%f167, %f40, %f40, %f166;
	fma.rn.f32 	%f168, %f38, %f38, %f167;
	setp.neu.f32 	%p30, %f168, 0f00000000;
	@%p30 bra 	$L__BB0_35;
	bra.uni 	$L__BB0_26;

$L__BB0_35:
	sub.f32 	%f201, %f19, %f40;
	sub.f32 	%f202, %f20, %f39;
	sub.f32 	%f203, %f21, %f38;
	mul.f32 	%f296, %f203, 0f3F000000;
	mul.f32 	%f295, %f202, 0f3F000000;
	mul.f32 	%f294, %f201, 0f3F000000;
	bra.uni 	$L__BB0_36;

$L__BB0_26:
	setp.eq.f32 	%p31, %f44, 0f00000000;
	and.pred  	%p33, %p31, %p23;
	@%p33 bra 	$L__BB0_34;
	bra.uni 	$L__BB0_27;

$L__BB0_34:
	sub.f32 	%f296, %f3, %f38;
	sub.f32 	%f295, %f2, %f39;
	sub.f32 	%f294, %f1, %f40;
	bra.uni 	$L__BB0_36;

$L__BB0_27:
	setp.eq.f32 	%p34, %f48, 0f00000000;
	mul.f32 	%f169, %f39, %f39;
	fma.rn.f32 	%f170, %f40, %f40, %f169;
	fma.rn.f32 	%f49, %f38, %f38, %f170;
	setp.eq.f32 	%p35, %f49, 0f00000000;
	and.pred  	%p36, %p35, %p34;
	@%p36 bra 	$L__BB0_33;
	bra.uni 	$L__BB0_28;

$L__BB0_33:
	sub.f32 	%f296, %f21, %f3;
	sub.f32 	%f295, %f20, %f2;
	sub.f32 	%f294, %f19, %f1;
	bra.uni 	$L__BB0_36;

$L__BB0_28:
	setp.neu.f32 	%p38, %f31, 0f00000000;
	or.pred  	%p39, %p31, %p38;
	@%p39 bra 	$L__BB0_30;
	bra.uni 	$L__BB0_29;

$L__BB0_30:
	setp.neu.f32 	%p40, %f49, 0f00000000;
	or.pred  	%p42, %p34, %p40;
	@%p42 bra 	$L__BB0_32;
	bra.uni 	$L__BB0_31;

$L__BB0_32:
	sub.f32 	%f192, %f19, %f40;
	sub.f32 	%f193, %f20, %f39;
	sub.f32 	%f194, %f21, %f38;
	sub.f32 	%f195, %f7, %f25;
	mul.f32 	%f196, %f195, 0f3DAAAAAB;
	sub.f32 	%f197, %f8, %f26;
	mul.f32 	%f198, %f197, 0f3DAAAAAB;
	sub.f32 	%f199, %f9, %f27;
	mul.f32 	%f200, %f199, 0f3DAAAAAB;
	fma.rn.f32 	%f296, %f194, 0f3F2AAAAB, %f200;
	fma.rn.f32 	%f295, %f193, 0f3F2AAAAB, %f198;
	fma.rn.f32 	%f294, %f192, 0f3F2AAAAB, %f196;
	bra.uni 	$L__BB0_36;

$L__BB0_29:
	mul.f32 	%f171, %f7, 0f3F000000;
	add.f32 	%f172, %f40, %f40;
	sub.f32 	%f173, %f171, %f172;
	add.f32 	%f174, %f39, %f39;
	mul.f32 	%f175, %f8, 0f3F000000;
	sub.f32 	%f176, %f175, %f174;
	add.f32 	%f177, %f38, %f38;
	mul.f32 	%f178, %f9, 0f3F000000;
	sub.f32 	%f179, %f178, %f177;
	fma.rn.f32 	%f296, %f3, 0f3FC00000, %f179;
	fma.rn.f32 	%f295, %f2, 0f3FC00000, %f176;
	fma.rn.f32 	%f294, %f1, 0f3FC00000, %f173;
	bra.uni 	$L__BB0_36;

$L__BB0_31:
	mul.f32 	%f180, %f25, 0f3F000000;
	add.f32 	%f181, %f19, %f19;
	sub.f32 	%f182, %f181, %f180;
	add.f32 	%f183, %f20, %f20;
	mul.f32 	%f184, %f26, 0f3F000000;
	sub.f32 	%f185, %f183, %f184;
	add.f32 	%f186, %f21, %f21;
	mul.f32 	%f187, %f27, 0f3F000000;
	sub.f32 	%f188, %f186, %f187;
	mul.f32 	%f189, %f1, 0f3FC00000;
	mul.f32 	%f190, %f2, 0f3FC00000;
	mul.f32 	%f191, %f3, 0f3FC00000;
	sub.f32 	%f296, %f188, %f191;
	sub.f32 	%f295, %f185, %f190;
	sub.f32 	%f294, %f182, %f189;

$L__BB0_36:
	and.b16  	%rs2, %rs3, 2;
	setp.eq.s16 	%p43, %rs2, 0;
	add.s32 	%r23, %r2, -2;
	@%p43 bra 	$L__BB0_38;
	bra.uni 	$L__BB0_37;

$L__BB0_38:
	max.s32 	%r89, %r23, 0;
	bra.uni 	$L__BB0_39;

$L__BB0_37:
	rem.s32 	%r67, %r23, %r41;
	add.s32 	%r68, %r67, %r41;
	rem.s32 	%r89, %r68, %r41;

$L__BB0_39:
	setp.lt.s32 	%p45, %r2, 2;
	mov.f32 	%f74, 0f00000000;
	and.pred  	%p46, %p45, %p43;
	mov.f32 	%f75, %f74;
	mov.f32 	%f76, %f74;
	@%p46 bra 	$L__BB0_41;

	add.s32 	%r69, %r89, %r4;
	mad.lo.s32 	%r70, %r69, %r40, %r1;
	mul.wide.s32 	%rd30, %r70, 4;
	add.s64 	%rd31, %rd3, %rd30;
	add.s64 	%rd32, %rd2, %rd30;
	add.s64 	%rd33, %rd1, %rd30;
	ld.global.nc.f32 	%f76, [%rd33];
	ld.global.nc.f32 	%f75, [%rd32];
	ld.global.nc.f32 	%f74, [%rd31];

$L__BB0_41:
	add.s32 	%r27, %r2, -1;
	@%p43 bra 	$L__BB0_43;
	bra.uni 	$L__BB0_42;

$L__BB0_43:
	max.s32 	%r90, %r27, 0;
	bra.uni 	$L__BB0_44;

$L__BB0_42:
	rem.s32 	%r71, %r27, %r41;
	add.s32 	%r72, %r71, %r41;
	rem.s32 	%r90, %r72, %r41;

$L__BB0_44:
	setp.lt.s32 	%p48, %r2, 1;
	mov.f32 	%f107, 0f00000000;
	and.pred  	%p50, %p48, %p43;
	mov.f32 	%f106, %f107;
	mov.f32 	%f105, %f107;
	@%p50 bra 	$L__BB0_46;

	add.s32 	%r73, %r90, %r4;
	mad.lo.s32 	%r74, %r73, %r40, %r1;
	mul.wide.s32 	%rd34, %r74, 4;
	add.s64 	%rd35, %rd3, %rd34;
	add.s64 	%rd36, %rd2, %rd34;
	add.s64 	%rd37, %rd1, %rd34;
	ld.global.nc.f32 	%f105, [%rd37];
	ld.global.nc.f32 	%f106, [%rd36];
	ld.global.nc.f32 	%f107, [%rd35];

$L__BB0_46:
	add.s32 	%r31, %r2, 1;
	@%p43 bra 	$L__BB0_48;
	bra.uni 	$L__BB0_47;

$L__BB0_48:
	add.s32 	%r77, %r41, -1;
	min.s32 	%r91, %r31, %r77;
	bra.uni 	$L__BB0_49;

$L__BB0_47:
	rem.s32 	%r75, %r31, %r41;
	add.s32 	%r76, %r75, %r41;
	rem.s32 	%r91, %r76, %r41;

$L__BB0_49:
	setp.ge.s32 	%p52, %r31, %r41;
	mov.f32 	%f86, 0f00000000;
	and.pred  	%p54, %p52, %p43;
	mov.f32 	%f87, %f86;
	mov.f32 	%f88, %f86;
	@%p54 bra 	$L__BB0_51;

	add.s32 	%r78, %r91, %r4;
	mad.lo.s32 	%r79, %r78, %r40, %r1;
	mul.wide.s32 	%rd38, %r79, 4;
	add.s64 	%rd39, %rd3, %rd38;
	add.s64 	%rd40, %rd2, %rd38;
	add.s64 	%rd41, %rd1, %rd38;
	ld.global.nc.f32 	%f88, [%rd41];
	ld.global.nc.f32 	%f87, [%rd40];
	ld.global.nc.f32 	%f86, [%rd39];

$L__BB0_51:
	add.s32 	%r35, %r2, 2;
	@%p43 bra 	$L__BB0_53;
	bra.uni 	$L__BB0_52;

$L__BB0_53:
	add.s32 	%r82, %r41, -1;
	min.s32 	%r92, %r35, %r82;
	bra.uni 	$L__BB0_54;

$L__BB0_52:
	rem.s32 	%r80, %r35, %r41;
	add.s32 	%r81, %r80, %r41;
	rem.s32 	%r92, %r81, %r41;

$L__BB0_54:
	add.s32 	%r83, %r92, %r4;
	mad.lo.s32 	%r39, %r83, %r40, %r1;
	setp.ge.s32 	%p56, %r35, %r41;
	mov.f32 	%f92, 0f00000000;
	and.pred  	%p58, %p56, %p43;
	mov.f32 	%f93, %f92;
	mov.f32 	%f94, %f92;
	@%p58 bra 	$L__BB0_56;

	mul.wide.s32 	%rd42, %r39, 4;
	add.s64 	%rd43, %rd3, %rd42;
	add.s64 	%rd44, %rd2, %rd42;
	add.s64 	%rd45, %rd1, %rd42;
	ld.global.nc.f32 	%f94, [%rd45];
	ld.global.nc.f32 	%f93, [%rd44];
	ld.global.nc.f32 	%f92, [%rd43];

$L__BB0_56:
	mul.f32 	%f216, %f87, %f87;
	fma.rn.f32 	%f217, %f86, %f86, %f216;
	fma.rn.f32 	%f98, %f88, %f88, %f217;
	setp.eq.f32 	%p59, %f98, 0f00000000;
	@%p59 bra 	$L__BB0_57;
	bra.uni 	$L__BB0_58;

$L__BB0_57:
	mul.f32 	%f221, %f106, %f106;
	fma.rn.f32 	%f222, %f107, %f107, %f221;
	fma.rn.f32 	%f223, %f105, %f105, %f222;
	setp.eq.f32 	%p60, %f223, 0f00000000;
	mov.f32 	%f312, 0f00000000;
	mov.f32 	%f313, %f312;
	mov.f32 	%f314, %f312;
	@%p60 bra 	$L__BB0_70;

$L__BB0_58:
	mul.f32 	%f224, %f75, %f75;
	fma.rn.f32 	%f225, %f74, %f74, %f224;
	fma.rn.f32 	%f111, %f76, %f76, %f225;
	setp.neu.f32 	%p61, %f111, 0f00000000;
	mul.f32 	%f226, %f93, %f93;
	fma.rn.f32 	%f227, %f92, %f92, %f226;
	fma.rn.f32 	%f115, %f94, %f94, %f227;
	setp.neu.f32 	%p62, %f115, 0f00000000;
	and.pred  	%p63, %p61, %p62;
	or.pred  	%p65, %p59, %p63;
	@%p65 bra 	$L__BB0_60;

	mul.f32 	%f228, %f106, %f106;
	fma.rn.f32 	%f229, %f107, %f107, %f228;
	fma.rn.f32 	%f230, %f105, %f105, %f229;
	setp.neu.f32 	%p66, %f230, 0f00000000;
	@%p66 bra 	$L__BB0_69;
	bra.uni 	$L__BB0_60;

$L__BB0_69:
	sub.f32 	%f263, %f86, %f107;
	sub.f32 	%f264, %f87, %f106;
	sub.f32 	%f265, %f88, %f105;
	mul.f32 	%f314, %f265, 0f3F000000;
	mul.f32 	%f313, %f264, 0f3F000000;
	mul.f32 	%f312, %f263, 0f3F000000;
	bra.uni 	$L__BB0_70;

$L__BB0_60:
	setp.eq.f32 	%p67, %f111, 0f00000000;
	and.pred  	%p69, %p67, %p59;
	@%p69 bra 	$L__BB0_68;
	bra.uni 	$L__BB0_61;

$L__BB0_68:
	sub.f32 	%f314, %f3, %f105;
	sub.f32 	%f313, %f2, %f106;
	sub.f32 	%f312, %f1, %f107;
	bra.uni 	$L__BB0_70;

$L__BB0_61:
	setp.eq.f32 	%p70, %f115, 0f00000000;
	mul.f32 	%f231, %f106, %f106;
	fma.rn.f32 	%f232, %f107, %f107, %f231;
	fma.rn.f32 	%f116, %f105, %f105, %f232;
	setp.eq.f32 	%p71, %f116, 0f00000000;
	and.pred  	%p72, %p71, %p70;
	@%p72 bra 	$L__BB0_67;
	bra.uni 	$L__BB0_62;

$L__BB0_67:
	sub.f32 	%f314, %f88, %f3;
	sub.f32 	%f313, %f87, %f2;
	sub.f32 	%f312, %f86, %f1;
	bra.uni 	$L__BB0_70;

$L__BB0_62:
	setp.neu.f32 	%p74, %f98, 0f00000000;
	or.pred  	%p75, %p67, %p74;
	@%p75 bra 	$L__BB0_64;
	bra.uni 	$L__BB0_63;

$L__BB0_64:
	setp.neu.f32 	%p76, %f116, 0f00000000;
	or.pred  	%p78, %p70, %p76;
	@%p78 bra 	$L__BB0_66;
	bra.uni 	$L__BB0_65;

$L__BB0_66:
	sub.f32 	%f254, %f86, %f107;
	sub.f32 	%f255, %f87, %f106;
	sub.f32 	%f256, %f88, %f105;
	sub.f32 	%f257, %f74, %f92;
	mul.f32 	%f258, %f257, 0f3DAAAAAB;
	sub.f32 	%f259, %f75, %f93;
	mul.f32 	%f260, %f259, 0f3DAAAAAB;
	sub.f32 	%f261, %f76, %f94;
	mul.f32 	%f262, %f261, 0f3DAAAAAB;
	fma.rn.f32 	%f314, %f256, 0f3F2AAAAB, %f262;
	fma.rn.f32 	%f313, %f255, 0f3F2AAAAB, %f260;
	fma.rn.f32 	%f312, %f254, 0f3F2AAAAB, %f258;
	bra.uni 	$L__BB0_70;

$L__BB0_63:
	mul.f32 	%f233, %f74, 0f3F000000;
	add.f32 	%f234, %f107, %f107;
	sub.f32 	%f235, %f233, %f234;
	add.f32 	%f236, %f106, %f106;
	mul.f32 	%f237, %f75, 0f3F000000;
	sub.f32 	%f238, %f237, %f236;
	add.f32 	%f239, %f105, %f105;
	mul.f32 	%f240, %f76, 0f3F000000;
	sub.f32 	%f241, %f240, %f239;
	fma.rn.f32 	%f314, %f3, 0f3FC00000, %f241;
	fma.rn.f32 	%f313, %f2, 0f3FC00000, %f238;
	fma.rn.f32 	%f312, %f1, 0f3FC00000, %f235;
	bra.uni 	$L__BB0_70;

$L__BB0_65:
	mul.f32 	%f242, %f92, 0f3F000000;
	add.f32 	%f243, %f86, %f86;
	sub.f32 	%f244, %f243, %f242;
	add.f32 	%f245, %f87, %f87;
	mul.f32 	%f246, %f93, 0f3F000000;
	sub.f32 	%f247, %f245, %f246;
	add.f32 	%f248, %f88, %f88;
	mul.f32 	%f249, %f94, 0f3F000000;
	sub.f32 	%f250, %f248, %f249;
	mul.f32 	%f251, %f1, 0f3FC00000;
	mul.f32 	%f252, %f2, 0f3FC00000;
	mul.f32 	%f253, %f3, 0f3FC00000;
	sub.f32 	%f314, %f250, %f253;
	sub.f32 	%f313, %f247, %f252;
	sub.f32 	%f312, %f244, %f251;

$L__BB0_70:
	mul.f32 	%f266, %f295, %f314;
	mul.f32 	%f267, %f296, %f313;
	sub.f32 	%f268, %f266, %f267;
	mul.f32 	%f269, %f296, %f312;
	mul.f32 	%f270, %f294, %f314;
	sub.f32 	%f271, %f269, %f270;
	mul.f32 	%f272, %f294, %f313;
	mul.f32 	%f273, %f295, %f312;
	sub.f32 	%f274, %f272, %f273;
	mul.f32 	%f275, %f2, %f271;
	fma.rn.f32 	%f276, %f1, %f268, %f275;
	fma.rn.f32 	%f277, %f3, %f274, %f276;
	mul.f32 	%f278, %f277, %f138;
	st.global.f32 	[%rd4], %f278;

$L__BB0_72:
	ret;

}

`
)
