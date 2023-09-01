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

// CUDA handle for addzhanglitorque2 kernel
var addzhanglitorque2_code cu.Function

// Stores the arguments for addzhanglitorque2 kernel invocation
type addzhanglitorque2_args_t struct {
	arg_tx        unsafe.Pointer
	arg_ty        unsafe.Pointer
	arg_tz        unsafe.Pointer
	arg_mx        unsafe.Pointer
	arg_my        unsafe.Pointer
	arg_mz        unsafe.Pointer
	arg_Ms_       unsafe.Pointer
	arg_Ms_mul    float32
	arg_jx_       unsafe.Pointer
	arg_jx_mul    float32
	arg_jy_       unsafe.Pointer
	arg_jy_mul    float32
	arg_jz_       unsafe.Pointer
	arg_jz_mul    float32
	arg_alpha_    unsafe.Pointer
	arg_alpha_mul float32
	arg_xi_       unsafe.Pointer
	arg_xi_mul    float32
	arg_pol_      unsafe.Pointer
	arg_pol_mul   float32
	arg_cx        float32
	arg_cy        float32
	arg_cz        float32
	arg_Nx        int
	arg_Ny        int
	arg_Nz        int
	arg_PBC       byte
	argptr        [27]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addzhanglitorque2 kernel invocation
var addzhanglitorque2_args addzhanglitorque2_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addzhanglitorque2_args.argptr[0] = unsafe.Pointer(&addzhanglitorque2_args.arg_tx)
	addzhanglitorque2_args.argptr[1] = unsafe.Pointer(&addzhanglitorque2_args.arg_ty)
	addzhanglitorque2_args.argptr[2] = unsafe.Pointer(&addzhanglitorque2_args.arg_tz)
	addzhanglitorque2_args.argptr[3] = unsafe.Pointer(&addzhanglitorque2_args.arg_mx)
	addzhanglitorque2_args.argptr[4] = unsafe.Pointer(&addzhanglitorque2_args.arg_my)
	addzhanglitorque2_args.argptr[5] = unsafe.Pointer(&addzhanglitorque2_args.arg_mz)
	addzhanglitorque2_args.argptr[6] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ms_)
	addzhanglitorque2_args.argptr[7] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ms_mul)
	addzhanglitorque2_args.argptr[8] = unsafe.Pointer(&addzhanglitorque2_args.arg_jx_)
	addzhanglitorque2_args.argptr[9] = unsafe.Pointer(&addzhanglitorque2_args.arg_jx_mul)
	addzhanglitorque2_args.argptr[10] = unsafe.Pointer(&addzhanglitorque2_args.arg_jy_)
	addzhanglitorque2_args.argptr[11] = unsafe.Pointer(&addzhanglitorque2_args.arg_jy_mul)
	addzhanglitorque2_args.argptr[12] = unsafe.Pointer(&addzhanglitorque2_args.arg_jz_)
	addzhanglitorque2_args.argptr[13] = unsafe.Pointer(&addzhanglitorque2_args.arg_jz_mul)
	addzhanglitorque2_args.argptr[14] = unsafe.Pointer(&addzhanglitorque2_args.arg_alpha_)
	addzhanglitorque2_args.argptr[15] = unsafe.Pointer(&addzhanglitorque2_args.arg_alpha_mul)
	addzhanglitorque2_args.argptr[16] = unsafe.Pointer(&addzhanglitorque2_args.arg_xi_)
	addzhanglitorque2_args.argptr[17] = unsafe.Pointer(&addzhanglitorque2_args.arg_xi_mul)
	addzhanglitorque2_args.argptr[18] = unsafe.Pointer(&addzhanglitorque2_args.arg_pol_)
	addzhanglitorque2_args.argptr[19] = unsafe.Pointer(&addzhanglitorque2_args.arg_pol_mul)
	addzhanglitorque2_args.argptr[20] = unsafe.Pointer(&addzhanglitorque2_args.arg_cx)
	addzhanglitorque2_args.argptr[21] = unsafe.Pointer(&addzhanglitorque2_args.arg_cy)
	addzhanglitorque2_args.argptr[22] = unsafe.Pointer(&addzhanglitorque2_args.arg_cz)
	addzhanglitorque2_args.argptr[23] = unsafe.Pointer(&addzhanglitorque2_args.arg_Nx)
	addzhanglitorque2_args.argptr[24] = unsafe.Pointer(&addzhanglitorque2_args.arg_Ny)
	addzhanglitorque2_args.argptr[25] = unsafe.Pointer(&addzhanglitorque2_args.arg_Nz)
	addzhanglitorque2_args.argptr[26] = unsafe.Pointer(&addzhanglitorque2_args.arg_PBC)
}

// Wrapper for addzhanglitorque2 CUDA kernel, asynchronous.
func k_addzhanglitorque2_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, jx_ unsafe.Pointer, jx_mul float32, jy_ unsafe.Pointer, jy_mul float32, jz_ unsafe.Pointer, jz_mul float32, alpha_ unsafe.Pointer, alpha_mul float32, xi_ unsafe.Pointer, xi_mul float32, pol_ unsafe.Pointer, pol_mul float32, cx float32, cy float32, cz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addzhanglitorque2")
	}

	addzhanglitorque2_args.Lock()
	defer addzhanglitorque2_args.Unlock()

	if addzhanglitorque2_code == 0 {
		addzhanglitorque2_code = fatbinLoad(addzhanglitorque2_map, "addzhanglitorque2")
	}

	addzhanglitorque2_args.arg_tx = tx
	addzhanglitorque2_args.arg_ty = ty
	addzhanglitorque2_args.arg_tz = tz
	addzhanglitorque2_args.arg_mx = mx
	addzhanglitorque2_args.arg_my = my
	addzhanglitorque2_args.arg_mz = mz
	addzhanglitorque2_args.arg_Ms_ = Ms_
	addzhanglitorque2_args.arg_Ms_mul = Ms_mul
	addzhanglitorque2_args.arg_jx_ = jx_
	addzhanglitorque2_args.arg_jx_mul = jx_mul
	addzhanglitorque2_args.arg_jy_ = jy_
	addzhanglitorque2_args.arg_jy_mul = jy_mul
	addzhanglitorque2_args.arg_jz_ = jz_
	addzhanglitorque2_args.arg_jz_mul = jz_mul
	addzhanglitorque2_args.arg_alpha_ = alpha_
	addzhanglitorque2_args.arg_alpha_mul = alpha_mul
	addzhanglitorque2_args.arg_xi_ = xi_
	addzhanglitorque2_args.arg_xi_mul = xi_mul
	addzhanglitorque2_args.arg_pol_ = pol_
	addzhanglitorque2_args.arg_pol_mul = pol_mul
	addzhanglitorque2_args.arg_cx = cx
	addzhanglitorque2_args.arg_cy = cy
	addzhanglitorque2_args.arg_cz = cz
	addzhanglitorque2_args.arg_Nx = Nx
	addzhanglitorque2_args.arg_Ny = Ny
	addzhanglitorque2_args.arg_Nz = Nz
	addzhanglitorque2_args.arg_PBC = PBC

	args := addzhanglitorque2_args.argptr[:]
	cu.LaunchKernel(addzhanglitorque2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addzhanglitorque2")
	}
}

// maps compute capability on PTX code for addzhanglitorque2 kernel.
var addzhanglitorque2_map = map[int]string{0: "",
	80: addzhanglitorque2_ptx_80}

// addzhanglitorque2 PTX code for various compute capabilities.
const (
	addzhanglitorque2_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	addzhanglitorque2

.visible .entry addzhanglitorque2(
	.param .u64 addzhanglitorque2_param_0,
	.param .u64 addzhanglitorque2_param_1,
	.param .u64 addzhanglitorque2_param_2,
	.param .u64 addzhanglitorque2_param_3,
	.param .u64 addzhanglitorque2_param_4,
	.param .u64 addzhanglitorque2_param_5,
	.param .u64 addzhanglitorque2_param_6,
	.param .f32 addzhanglitorque2_param_7,
	.param .u64 addzhanglitorque2_param_8,
	.param .f32 addzhanglitorque2_param_9,
	.param .u64 addzhanglitorque2_param_10,
	.param .f32 addzhanglitorque2_param_11,
	.param .u64 addzhanglitorque2_param_12,
	.param .f32 addzhanglitorque2_param_13,
	.param .u64 addzhanglitorque2_param_14,
	.param .f32 addzhanglitorque2_param_15,
	.param .u64 addzhanglitorque2_param_16,
	.param .f32 addzhanglitorque2_param_17,
	.param .u64 addzhanglitorque2_param_18,
	.param .f32 addzhanglitorque2_param_19,
	.param .f32 addzhanglitorque2_param_20,
	.param .f32 addzhanglitorque2_param_21,
	.param .f32 addzhanglitorque2_param_22,
	.param .u32 addzhanglitorque2_param_23,
	.param .u32 addzhanglitorque2_param_24,
	.param .u32 addzhanglitorque2_param_25,
	.param .u8 addzhanglitorque2_param_26
)
{
	.reg .pred 	%p<35>;
	.reg .b16 	%rs<5>;
	.reg .f32 	%f<149>;
	.reg .b32 	%r<173>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<84>;


	ld.param.u8 	%rs4, [addzhanglitorque2_param_26];
	ld.param.u64 	%rd4, [addzhanglitorque2_param_0];
	ld.param.u64 	%rd5, [addzhanglitorque2_param_1];
	ld.param.u64 	%rd6, [addzhanglitorque2_param_2];
	ld.param.u64 	%rd14, [addzhanglitorque2_param_3];
	ld.param.u64 	%rd15, [addzhanglitorque2_param_4];
	ld.param.u64 	%rd16, [addzhanglitorque2_param_5];
	ld.param.u64 	%rd7, [addzhanglitorque2_param_6];
	ld.param.f32 	%f135, [addzhanglitorque2_param_7];
	ld.param.u64 	%rd8, [addzhanglitorque2_param_8];
	ld.param.f32 	%f137, [addzhanglitorque2_param_9];
	ld.param.u64 	%rd9, [addzhanglitorque2_param_10];
	ld.param.f32 	%f138, [addzhanglitorque2_param_11];
	ld.param.u64 	%rd10, [addzhanglitorque2_param_12];
	ld.param.f32 	%f139, [addzhanglitorque2_param_13];
	ld.param.u64 	%rd11, [addzhanglitorque2_param_14];
	ld.param.f32 	%f132, [addzhanglitorque2_param_15];
	ld.param.u64 	%rd12, [addzhanglitorque2_param_16];
	ld.param.f32 	%f133, [addzhanglitorque2_param_17];
	ld.param.u64 	%rd13, [addzhanglitorque2_param_18];
	ld.param.f32 	%f134, [addzhanglitorque2_param_19];
	ld.param.f32 	%f64, [addzhanglitorque2_param_20];
	ld.param.f32 	%f65, [addzhanglitorque2_param_21];
	ld.param.f32 	%f66, [addzhanglitorque2_param_22];
	ld.param.u32 	%r67, [addzhanglitorque2_param_23];
	ld.param.u32 	%r68, [addzhanglitorque2_param_24];
	ld.param.u32 	%r69, [addzhanglitorque2_param_25];
	cvta.to.global.u64 	%rd1, %rd16;
	cvta.to.global.u64 	%rd2, %rd15;
	cvta.to.global.u64 	%rd3, %rd14;
	mov.u32 	%r70, %ntid.x;
	mov.u32 	%r71, %ctaid.x;
	mov.u32 	%r72, %tid.x;
	mad.lo.s32 	%r1, %r71, %r70, %r72;
	mov.u32 	%r73, %ntid.y;
	mov.u32 	%r74, %ctaid.y;
	mov.u32 	%r75, %tid.y;
	mad.lo.s32 	%r2, %r74, %r73, %r75;
	mov.u32 	%r76, %ntid.z;
	mov.u32 	%r77, %ctaid.z;
	mov.u32 	%r78, %tid.z;
	mad.lo.s32 	%r3, %r77, %r76, %r78;
	setp.ge.s32 	%p1, %r1, %r67;
	setp.ge.s32 	%p2, %r2, %r68;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r69;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_78;

	mul.lo.s32 	%r4, %r3, %r68;
	add.s32 	%r79, %r4, %r2;
	mul.lo.s32 	%r5, %r79, %r67;
	add.s32 	%r6, %r5, %r1;
	setp.eq.s64 	%p6, %rd11, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd17, %rd11;
	mul.wide.s32 	%rd18, %r6, 4;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f67, [%rd19];
	mul.f32 	%f132, %f67, %f132;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd12, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd20, %rd12;
	mul.wide.s32 	%rd21, %r6, 4;
	add.s64 	%rd22, %rd20, %rd21;
	ld.global.nc.f32 	%f68, [%rd22];
	mul.f32 	%f133, %f68, %f133;

$L__BB0_5:
	setp.eq.s64 	%p8, %rd13, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd23, %rd13;
	mul.wide.s32 	%rd24, %r6, 4;
	add.s64 	%rd25, %rd23, %rd24;
	ld.global.nc.f32 	%f69, [%rd25];
	mul.f32 	%f134, %f69, %f134;

$L__BB0_7:
	setp.eq.s64 	%p9, %rd7, 0;
	@%p9 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd26, %rd7;
	mul.wide.s32 	%rd27, %r6, 4;
	add.s64 	%rd28, %rd26, %rd27;
	ld.global.nc.f32 	%f70, [%rd28];
	mul.f32 	%f135, %f70, %f135;

$L__BB0_9:
	setp.eq.f32 	%p10, %f135, 0f00000000;
	mov.f32 	%f136, 0f00000000;
	@%p10 bra 	$L__BB0_11;

	rcp.rn.f32 	%f136, %f135;

$L__BB0_11:
	cvt.f64.f32 	%fd1, %f136;
	mul.f64 	%fd2, %fd1, 0d3CA7B4966C8AC112;
	fma.rn.f32 	%f72, %f133, %f133, 0f3F800000;
	cvt.f64.f32 	%fd3, %f72;
	div.rn.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f11, %fd4;
	setp.eq.s64 	%p11, %rd8, 0;
	@%p11 bra 	$L__BB0_13;

	cvta.to.global.u64 	%rd29, %rd8;
	mul.wide.s32 	%rd30, %r6, 4;
	add.s64 	%rd31, %rd29, %rd30;
	ld.global.nc.f32 	%f73, [%rd31];
	mul.f32 	%f137, %f73, %f137;

$L__BB0_13:
	setp.eq.s64 	%p12, %rd9, 0;
	@%p12 bra 	$L__BB0_15;

	cvta.to.global.u64 	%rd32, %rd9;
	mul.wide.s32 	%rd33, %r6, 4;
	add.s64 	%rd34, %rd32, %rd33;
	ld.global.nc.f32 	%f74, [%rd34];
	mul.f32 	%f138, %f74, %f138;

$L__BB0_15:
	setp.eq.s64 	%p13, %rd10, 0;
	@%p13 bra 	$L__BB0_17;

	cvta.to.global.u64 	%rd35, %rd10;
	mul.wide.s32 	%rd36, %r6, 4;
	add.s64 	%rd37, %rd35, %rd36;
	ld.global.nc.f32 	%f75, [%rd37];
	mul.f32 	%f139, %f75, %f139;

$L__BB0_17:
	mul.f32 	%f18, %f134, %f138;
	mul.f32 	%f19, %f134, %f139;
	mul.f32 	%f20, %f134, %f137;
	setp.eq.f32 	%p14, %f20, 0f00000000;
	mov.f32 	%f143, 0f00000000;
	mov.f32 	%f144, %f143;
	mov.f32 	%f145, %f143;
	@%p14 bra 	$L__BB0_37;

	and.b16  	%rs1, %rs4, 1;
	setp.eq.s16 	%p15, %rs1, 0;
	add.s32 	%r7, %r1, 1;
	@%p15 bra 	$L__BB0_20;
	bra.uni 	$L__BB0_19;

$L__BB0_20:
	add.s32 	%r82, %r67, -1;
	min.s32 	%r155, %r7, %r82;
	bra.uni 	$L__BB0_21;

$L__BB0_19:
	rem.s32 	%r80, %r7, %r67;
	add.s32 	%r81, %r80, %r67;
	rem.s32 	%r155, %r81, %r67;

$L__BB0_21:
	add.s32 	%r83, %r155, %r5;
	mul.wide.s32 	%rd38, %r83, 4;
	add.s64 	%rd39, %rd3, %rd38;
	ld.global.nc.f32 	%f21, [%rd39];
	add.s32 	%r11, %r1, -1;
	@%p15 bra 	$L__BB0_23;
	bra.uni 	$L__BB0_22;

$L__BB0_23:
	max.s32 	%r156, %r11, 0;
	bra.uni 	$L__BB0_24;

$L__BB0_22:
	rem.s32 	%r84, %r11, %r67;
	add.s32 	%r85, %r84, %r67;
	rem.s32 	%r156, %r85, %r67;

$L__BB0_24:
	div.rn.f32 	%f79, %f11, %f64;
	mul.f32 	%f22, %f79, %f20;
	add.s32 	%r86, %r156, %r5;
	mul.wide.s32 	%rd40, %r86, 4;
	add.s64 	%rd41, %rd3, %rd40;
	ld.global.nc.f32 	%f80, [%rd41];
	sub.f32 	%f23, %f21, %f80;
	@%p15 bra 	$L__BB0_26;
	bra.uni 	$L__BB0_25;

$L__BB0_26:
	add.s32 	%r89, %r67, -1;
	min.s32 	%r157, %r7, %r89;
	bra.uni 	$L__BB0_27;

$L__BB0_25:
	rem.s32 	%r87, %r7, %r67;
	add.s32 	%r88, %r87, %r67;
	rem.s32 	%r157, %r88, %r67;

$L__BB0_27:
	add.s32 	%r90, %r157, %r5;
	mul.wide.s32 	%rd42, %r90, 4;
	add.s64 	%rd43, %rd2, %rd42;
	ld.global.nc.f32 	%f24, [%rd43];
	@%p15 bra 	$L__BB0_29;
	bra.uni 	$L__BB0_28;

$L__BB0_29:
	max.s32 	%r158, %r11, 0;
	bra.uni 	$L__BB0_30;

$L__BB0_28:
	rem.s32 	%r91, %r11, %r67;
	add.s32 	%r92, %r91, %r67;
	rem.s32 	%r158, %r92, %r67;

$L__BB0_30:
	add.s32 	%r93, %r158, %r5;
	mul.wide.s32 	%rd44, %r93, 4;
	add.s64 	%rd45, %rd2, %rd44;
	ld.global.nc.f32 	%f81, [%rd45];
	sub.f32 	%f25, %f24, %f81;
	@%p15 bra 	$L__BB0_32;
	bra.uni 	$L__BB0_31;

$L__BB0_32:
	add.s32 	%r96, %r67, -1;
	min.s32 	%r159, %r7, %r96;
	bra.uni 	$L__BB0_33;

$L__BB0_31:
	rem.s32 	%r94, %r7, %r67;
	add.s32 	%r95, %r94, %r67;
	rem.s32 	%r159, %r95, %r67;

$L__BB0_33:
	add.s32 	%r97, %r159, %r5;
	mul.wide.s32 	%rd46, %r97, 4;
	add.s64 	%rd47, %rd1, %rd46;
	ld.global.nc.f32 	%f26, [%rd47];
	@%p15 bra 	$L__BB0_35;
	bra.uni 	$L__BB0_34;

$L__BB0_35:
	max.s32 	%r160, %r11, 0;
	bra.uni 	$L__BB0_36;

$L__BB0_34:
	rem.s32 	%r98, %r11, %r67;
	add.s32 	%r99, %r98, %r67;
	rem.s32 	%r160, %r99, %r67;

$L__BB0_36:
	add.s32 	%r100, %r160, %r5;
	mul.wide.s32 	%rd48, %r100, 4;
	add.s64 	%rd49, %rd1, %rd48;
	ld.global.nc.f32 	%f82, [%rd49];
	sub.f32 	%f83, %f26, %f82;
	fma.rn.f32 	%f143, %f22, %f23, 0f00000000;
	fma.rn.f32 	%f144, %f22, %f25, 0f00000000;
	fma.rn.f32 	%f145, %f22, %f83, 0f00000000;

$L__BB0_37:
	setp.eq.f32 	%p21, %f18, 0f00000000;
	@%p21 bra 	$L__BB0_57;

	and.b16  	%rs2, %rs4, 2;
	setp.eq.s16 	%p22, %rs2, 0;
	add.s32 	%r27, %r2, 1;
	@%p22 bra 	$L__BB0_40;
	bra.uni 	$L__BB0_39;

$L__BB0_40:
	add.s32 	%r103, %r68, -1;
	min.s32 	%r161, %r27, %r103;
	bra.uni 	$L__BB0_41;

$L__BB0_39:
	rem.s32 	%r101, %r27, %r68;
	add.s32 	%r102, %r101, %r68;
	rem.s32 	%r161, %r102, %r68;

$L__BB0_41:
	add.s32 	%r104, %r161, %r4;
	mad.lo.s32 	%r105, %r104, %r67, %r1;
	mul.wide.s32 	%rd50, %r105, 4;
	add.s64 	%rd51, %rd3, %rd50;
	ld.global.nc.f32 	%f33, [%rd51];
	add.s32 	%r31, %r2, -1;
	@%p22 bra 	$L__BB0_43;
	bra.uni 	$L__BB0_42;

$L__BB0_43:
	max.s32 	%r162, %r31, 0;
	bra.uni 	$L__BB0_44;

$L__BB0_42:
	rem.s32 	%r106, %r31, %r68;
	add.s32 	%r107, %r106, %r68;
	rem.s32 	%r162, %r107, %r68;

$L__BB0_44:
	div.rn.f32 	%f84, %f11, %f65;
	mul.f32 	%f34, %f84, %f18;
	add.s32 	%r108, %r162, %r4;
	mad.lo.s32 	%r109, %r108, %r67, %r1;
	mul.wide.s32 	%rd52, %r109, 4;
	add.s64 	%rd53, %rd3, %rd52;
	ld.global.nc.f32 	%f85, [%rd53];
	sub.f32 	%f35, %f33, %f85;
	@%p22 bra 	$L__BB0_46;
	bra.uni 	$L__BB0_45;

$L__BB0_46:
	add.s32 	%r112, %r68, -1;
	min.s32 	%r163, %r27, %r112;
	bra.uni 	$L__BB0_47;

$L__BB0_45:
	rem.s32 	%r110, %r27, %r68;
	add.s32 	%r111, %r110, %r68;
	rem.s32 	%r163, %r111, %r68;

$L__BB0_47:
	add.s32 	%r113, %r163, %r4;
	mad.lo.s32 	%r114, %r113, %r67, %r1;
	mul.wide.s32 	%rd54, %r114, 4;
	add.s64 	%rd55, %rd2, %rd54;
	ld.global.nc.f32 	%f36, [%rd55];
	@%p22 bra 	$L__BB0_49;
	bra.uni 	$L__BB0_48;

$L__BB0_49:
	max.s32 	%r164, %r31, 0;
	bra.uni 	$L__BB0_50;

$L__BB0_48:
	rem.s32 	%r115, %r31, %r68;
	add.s32 	%r116, %r115, %r68;
	rem.s32 	%r164, %r116, %r68;

$L__BB0_50:
	add.s32 	%r117, %r164, %r4;
	mad.lo.s32 	%r118, %r117, %r67, %r1;
	mul.wide.s32 	%rd56, %r118, 4;
	add.s64 	%rd57, %rd2, %rd56;
	ld.global.nc.f32 	%f86, [%rd57];
	sub.f32 	%f37, %f36, %f86;
	@%p22 bra 	$L__BB0_52;
	bra.uni 	$L__BB0_51;

$L__BB0_52:
	add.s32 	%r121, %r68, -1;
	min.s32 	%r165, %r27, %r121;
	bra.uni 	$L__BB0_53;

$L__BB0_51:
	rem.s32 	%r119, %r27, %r68;
	add.s32 	%r120, %r119, %r68;
	rem.s32 	%r165, %r120, %r68;

$L__BB0_53:
	add.s32 	%r122, %r165, %r4;
	mad.lo.s32 	%r123, %r122, %r67, %r1;
	mul.wide.s32 	%rd58, %r123, 4;
	add.s64 	%rd59, %rd1, %rd58;
	ld.global.nc.f32 	%f38, [%rd59];
	@%p22 bra 	$L__BB0_55;
	bra.uni 	$L__BB0_54;

$L__BB0_55:
	max.s32 	%r166, %r31, 0;
	bra.uni 	$L__BB0_56;

$L__BB0_54:
	rem.s32 	%r124, %r31, %r68;
	add.s32 	%r125, %r124, %r68;
	rem.s32 	%r166, %r125, %r68;

$L__BB0_56:
	add.s32 	%r126, %r166, %r4;
	mad.lo.s32 	%r127, %r126, %r67, %r1;
	mul.wide.s32 	%rd60, %r127, 4;
	add.s64 	%rd61, %rd1, %rd60;
	ld.global.nc.f32 	%f87, [%rd61];
	sub.f32 	%f88, %f38, %f87;
	fma.rn.f32 	%f143, %f34, %f35, %f143;
	fma.rn.f32 	%f144, %f34, %f37, %f144;
	fma.rn.f32 	%f145, %f34, %f88, %f145;

$L__BB0_57:
	setp.eq.f32 	%p28, %f19, 0f00000000;
	@%p28 bra 	$L__BB0_77;

	div.rn.f32 	%f89, %f11, %f66;
	mul.f32 	%f45, %f89, %f19;
	and.b16  	%rs3, %rs4, 4;
	setp.eq.s16 	%p29, %rs3, 0;
	add.s32 	%r47, %r3, 1;
	@%p29 bra 	$L__BB0_60;
	bra.uni 	$L__BB0_59;

$L__BB0_60:
	add.s32 	%r130, %r69, -1;
	min.s32 	%r167, %r47, %r130;
	bra.uni 	$L__BB0_61;

$L__BB0_59:
	rem.s32 	%r128, %r47, %r69;
	add.s32 	%r129, %r128, %r69;
	rem.s32 	%r167, %r129, %r69;

$L__BB0_61:
	mad.lo.s32 	%r131, %r167, %r68, %r2;
	mad.lo.s32 	%r132, %r131, %r67, %r1;
	mul.wide.s32 	%rd62, %r132, 4;
	add.s64 	%rd63, %rd3, %rd62;
	ld.global.nc.f32 	%f46, [%rd63];
	add.s32 	%r51, %r3, -1;
	@%p29 bra 	$L__BB0_63;
	bra.uni 	$L__BB0_62;

$L__BB0_63:
	max.s32 	%r168, %r51, 0;
	bra.uni 	$L__BB0_64;

$L__BB0_62:
	rem.s32 	%r133, %r51, %r69;
	add.s32 	%r134, %r133, %r69;
	rem.s32 	%r168, %r134, %r69;

$L__BB0_64:
	mad.lo.s32 	%r135, %r168, %r68, %r2;
	mad.lo.s32 	%r136, %r135, %r67, %r1;
	mul.wide.s32 	%rd64, %r136, 4;
	add.s64 	%rd65, %rd3, %rd64;
	ld.global.nc.f32 	%f90, [%rd65];
	sub.f32 	%f47, %f46, %f90;
	@%p29 bra 	$L__BB0_66;
	bra.uni 	$L__BB0_65;

$L__BB0_66:
	add.s32 	%r139, %r69, -1;
	min.s32 	%r169, %r47, %r139;
	bra.uni 	$L__BB0_67;

$L__BB0_65:
	rem.s32 	%r137, %r47, %r69;
	add.s32 	%r138, %r137, %r69;
	rem.s32 	%r169, %r138, %r69;

$L__BB0_67:
	mad.lo.s32 	%r140, %r169, %r68, %r2;
	mad.lo.s32 	%r141, %r140, %r67, %r1;
	mul.wide.s32 	%rd66, %r141, 4;
	add.s64 	%rd67, %rd2, %rd66;
	ld.global.nc.f32 	%f48, [%rd67];
	@%p29 bra 	$L__BB0_69;
	bra.uni 	$L__BB0_68;

$L__BB0_69:
	max.s32 	%r170, %r51, 0;
	bra.uni 	$L__BB0_70;

$L__BB0_68:
	rem.s32 	%r142, %r51, %r69;
	add.s32 	%r143, %r142, %r69;
	rem.s32 	%r170, %r143, %r69;

$L__BB0_70:
	mad.lo.s32 	%r144, %r170, %r68, %r2;
	mad.lo.s32 	%r145, %r144, %r67, %r1;
	mul.wide.s32 	%rd68, %r145, 4;
	add.s64 	%rd69, %rd2, %rd68;
	ld.global.nc.f32 	%f91, [%rd69];
	sub.f32 	%f49, %f48, %f91;
	@%p29 bra 	$L__BB0_72;
	bra.uni 	$L__BB0_71;

$L__BB0_72:
	add.s32 	%r148, %r69, -1;
	min.s32 	%r171, %r47, %r148;
	bra.uni 	$L__BB0_73;

$L__BB0_71:
	rem.s32 	%r146, %r47, %r69;
	add.s32 	%r147, %r146, %r69;
	rem.s32 	%r171, %r147, %r69;

$L__BB0_73:
	mad.lo.s32 	%r149, %r171, %r68, %r2;
	mad.lo.s32 	%r150, %r149, %r67, %r1;
	mul.wide.s32 	%rd70, %r150, 4;
	add.s64 	%rd71, %rd1, %rd70;
	ld.global.nc.f32 	%f50, [%rd71];
	@%p29 bra 	$L__BB0_75;
	bra.uni 	$L__BB0_74;

$L__BB0_75:
	max.s32 	%r172, %r51, 0;
	bra.uni 	$L__BB0_76;

$L__BB0_74:
	rem.s32 	%r151, %r51, %r69;
	add.s32 	%r152, %r151, %r69;
	rem.s32 	%r172, %r152, %r69;

$L__BB0_76:
	mad.lo.s32 	%r153, %r172, %r68, %r2;
	mad.lo.s32 	%r154, %r153, %r67, %r1;
	mul.wide.s32 	%rd72, %r154, 4;
	add.s64 	%rd73, %rd1, %rd72;
	ld.global.nc.f32 	%f92, [%rd73];
	sub.f32 	%f93, %f50, %f92;
	fma.rn.f32 	%f143, %f45, %f47, %f143;
	fma.rn.f32 	%f144, %f45, %f49, %f144;
	fma.rn.f32 	%f145, %f45, %f93, %f145;

$L__BB0_77:
	mul.wide.s32 	%rd74, %r6, 4;
	add.s64 	%rd75, %rd3, %rd74;
	add.s64 	%rd76, %rd2, %rd74;
	add.s64 	%rd77, %rd1, %rd74;
	fma.rn.f32 	%f94, %f132, %f132, 0f3F800000;
	mov.f32 	%f95, 0fBF800000;
	div.rn.f32 	%f96, %f95, %f94;
	fma.rn.f32 	%f97, %f132, %f133, 0f3F800000;
	ld.global.nc.f32 	%f98, [%rd76];
	mul.f32 	%f99, %f145, %f98;
	ld.global.nc.f32 	%f100, [%rd77];
	mul.f32 	%f101, %f144, %f100;
	sub.f32 	%f102, %f99, %f101;
	mul.f32 	%f103, %f143, %f100;
	ld.global.nc.f32 	%f104, [%rd75];
	mul.f32 	%f105, %f145, %f104;
	sub.f32 	%f106, %f103, %f105;
	mul.f32 	%f107, %f144, %f104;
	mul.f32 	%f108, %f143, %f98;
	sub.f32 	%f109, %f107, %f108;
	mul.f32 	%f110, %f98, %f109;
	mul.f32 	%f111, %f100, %f106;
	sub.f32 	%f112, %f110, %f111;
	mul.f32 	%f113, %f100, %f102;
	mul.f32 	%f114, %f104, %f109;
	sub.f32 	%f115, %f113, %f114;
	mul.f32 	%f116, %f104, %f106;
	mul.f32 	%f117, %f98, %f102;
	sub.f32 	%f118, %f116, %f117;
	mul.f32 	%f119, %f97, %f112;
	mul.f32 	%f120, %f97, %f115;
	mul.f32 	%f121, %f97, %f118;
	sub.f32 	%f122, %f133, %f132;
	fma.rn.f32 	%f123, %f122, %f102, %f119;
	fma.rn.f32 	%f124, %f122, %f106, %f120;
	fma.rn.f32 	%f125, %f122, %f109, %f121;
	cvta.to.global.u64 	%rd78, %rd4;
	add.s64 	%rd79, %rd78, %rd74;
	ld.global.f32 	%f126, [%rd79];
	fma.rn.f32 	%f127, %f96, %f123, %f126;
	st.global.f32 	[%rd79], %f127;
	cvta.to.global.u64 	%rd80, %rd5;
	add.s64 	%rd81, %rd80, %rd74;
	ld.global.f32 	%f128, [%rd81];
	fma.rn.f32 	%f129, %f96, %f124, %f128;
	st.global.f32 	[%rd81], %f129;
	cvta.to.global.u64 	%rd82, %rd6;
	add.s64 	%rd83, %rd82, %rd74;
	ld.global.f32 	%f130, [%rd83];
	fma.rn.f32 	%f131, %f96, %f125, %f130;
	st.global.f32 	[%rd83], %f131;

$L__BB0_78:
	ret;

}

`
)
