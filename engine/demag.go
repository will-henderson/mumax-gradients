package engine

// Calculation of magnetostatic field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// Demag variables
var (
	Msat = NewScalarParam("Msat", "A/m", "Saturation magnetization",
		[]derived{&Bsat, &lex2, &din2, &dbulk2, &ku1_red, &ku2_red, &kc1_red, &kc2_red, &kc3_red, &temp_red})
	Bsat        derivedParam
	M_full      = NewVectorField("m_full", "A/m", "Unnormalized magnetization", SetMFull)
	B_demag     = NewVectorField("B_demag", "T", "Magnetostatic field", SetDemagField)
	Edens_demag = NewScalarField("Edens_demag", "J/m3", "Magnetostatic energy density", AddEdens_demag)
	E_demag     = NewScalarValue("E_demag", "J", "Magnetostatic energy", GetDemagEnergy)

	EnableDemag   = true // enable/disable global demag field
	NoDemagSpins  = NewScalarParam("NoDemagSpins", "", "Disable magnetostatic interaction per-spin (set to 1 to disable)", nil)
	conv_         *cuda.DemagConvolution // does the heavy lifting and provides FFTM
	DemagAccuracy = 6.0                  // Demag accuracy (divide cubes in at most N^3 points)
	CacheDir      = ""                   // directory for kernel cache
	TestDemag     = false                // enable convolution self-test
)

var AddEdens_demag = makeEdensAdder(&B_demag, -0.5)

func init() {

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	DeclVar("DemagAccuracy", &DemagAccuracy, "Controls accuracy of demag kernel")
	registerEnergy(GetDemagEnergy, AddEdens_demag)

	//Bsat = Msat * mu0
	Bsat.init(SCALAR, []updater{Msat}, func(p *derivedParam) {
		Ms := Msat.cpuLUT()
		for i, ms := range Ms[0] {
			p.cpu_buf[0][i] = mag.Mu0 * ms
		}
	})
}

// Sets dst to the current demag field
func SetDemagField(dst *data.Slice) {
	if EnableDemag {
		if NoDemagSpins.isZero() {
			// Normal demag, everywhere
			demagConv().Exec(dst, M.Buffer(), geometry.Gpu(), Bsat.gpuLUT1(), regions.Gpu())
		} else {
			setMaskedDemagField(dst)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

// Sets dst to the demag field, but cells where NoDemagSpins != 0 do not generate nor recieve field.
func setMaskedDemagField(dst *data.Slice) {
	// No-demag spins: mask-out geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := cuda.Buffer(SCALAR, geometry.Gpu().Size()) // masked-out geometry
	defer cuda.Recycle(buf)

	// obtain a copy of the geometry mask, which we can overwrite
	geom, r := geometry.Slice()
	if r {
		defer cuda.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	cuda.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())

	// convolution with masked-out cells.
	demagConv().Exec(dst, M.Buffer(), buf, Bsat.gpuLUT1(), regions.Gpu())

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	cuda.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

// Sets dst to the full (unnormalized) magnetization in A/m
func SetMFull(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// returns demag convolution, making sure it's initialized
func demagConv() *cuda.DemagConvolution {
	if conv_ == nil {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.DemagKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, CacheDir)
		conv_ = cuda.NewDemag(Mesh().Size(), Mesh().PBC(), kernel, TestDemag)
	}
	return conv_
}

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}
