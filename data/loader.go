package data

type Loader interface {
	M(n int) *Slice
	B(n int) *Slice
	PushM(M *Slice)
	PushB(B *Slice)
}

// GPULoader acts simply, holding the magnetisation at each timestep in
type GPULoader struct {
	mags   []*Slice
	fields []*Slice
}

func (l GPULoader) M(n int) *Slice {
	return l.mags[n]
}

func (l GPULoader) B(n int) *Slice {
	return l.fields[n]
}

func (l GPULoader) PushM(M *Slice) {
	l.mags = append(l.mags, M)
}

func (l GPULoader) PushB(B *Slice) {
	l.fields = append(l.mags, B)
}
