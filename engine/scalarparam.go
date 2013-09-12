package engine

import (
	//"code.google.com/p/mx3/cuda"
	"reflect"
)

// specialized param with 1 component
type ScalarParam struct {
	inputParam
}

func (p *ScalarParam) init(name, unit, desc string, children []*derivedParam) {
	p.inputParam.init(1, name, unit, children)
	DeclLValue(name, p, desc)
}

func (p *ScalarParam) SetRegion(region int, value float64) {
	p.setRegion(region, []float64{value})
}

func (p *ScalarParam) SetValue(v interface{}) {
	p.setUniform([]float64{v.(float64)})
}

func (p *ScalarParam) Eval() interface{}            { return p }
func (p *ScalarParam) Type() reflect.Type           { return reflect.TypeOf(new(ScalarParam)) }
func (p *ScalarParam) InputType() reflect.Type      { return reflect.TypeOf(float64(0)) }
func (p *ScalarParam) GetRegion(region int) float64 { return float64(p.getRegion(region)[0]) }
func (p *ScalarParam) GetUniform() float64          { return p.getUniform()[0] }
func (p *ScalarParam) Set(v float64)                { p.setUniform([]float64{v}) }
func (p *ScalarParam) GetFloat() float64            { return p.GetUniform() }

func (p *ScalarParam) SetFunc(f func() float64) {
	panic("todo")
}
