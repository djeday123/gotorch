package optim

import (
	"gotorch_v1/autograd"
	"gotorch_v1/tensor"
	"math"
	"testing"
)

func TestSGDStep(t *testing.T) {
	// param = [2.0], grad = [1.0], lr = 0.1
	// after step: param = [1.9]
	p := autograd.NewVar(tensor.New([]float64{2.0}, []int{1}), true)
	p.Grad = tensor.New([]float64{1.0}, []int{1})

	sgd := NewSGD([]*autograd.Variable{p}, 0.1, 0)
	sgd.Step()

	if math.Abs(p.Data.At(0)-1.9) > 1e-9 {
		t.Fatalf("SGD step: expected 1.9, got %f", p.Data.At(0))
	}
}

func TestSGDMomentum(t *testing.T) {
	p := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	p.Grad = tensor.New([]float64{1.0}, []int{1})

	sgd := NewSGD([]*autograd.Variable{p}, 0.1, 0.9)
	sgd.Step() // v = 0.9*0 + 1.0 = 1.0; p = 1.0 - 0.1*1.0 = 0.9

	if math.Abs(p.Data.At(0)-0.9) > 1e-9 {
		t.Fatalf("SGD momentum step 1: expected 0.9, got %f", p.Data.At(0))
	}

	p.Grad = tensor.New([]float64{1.0}, []int{1})
	sgd.Step() // v = 0.9*1.0 + 1.0 = 1.9; p = 0.9 - 0.1*1.9 = 0.71

	if math.Abs(p.Data.At(0)-0.71) > 1e-9 {
		t.Fatalf("SGD momentum step 2: expected 0.71, got %f", p.Data.At(0))
	}
}

func TestSGDZeroGrad(t *testing.T) {
	p := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	p.Grad = tensor.New([]float64{5.0}, []int{1})
	sgd := NewSGD([]*autograd.Variable{p}, 0.1, 0)
	sgd.ZeroGrad()
	if p.Grad != nil {
		t.Fatal("ZeroGrad should set Grad to nil")
	}
}

func TestAdamStep(t *testing.T) {
	// Single param, single step — verify it moves in the right direction
	p := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	p.Grad = tensor.New([]float64{1.0}, []int{1})

	adam := NewAdam([]*autograd.Variable{p}, 0.001, 0.9, 0.999, 1e-8)
	adam.Step()

	// After 1 step: param should decrease
	if p.Data.At(0) >= 1.0 {
		t.Fatalf("Adam: param should have decreased, got %f", p.Data.At(0))
	}
}

func TestAdamMultipleSteps(t *testing.T) {
	// Minimize f(x) = x^2 starting from x=5.0
	// Gradient = 2x
	p := autograd.NewVar(tensor.New([]float64{5.0}, []int{1}), true)
	adam := NewAdam([]*autograd.Variable{p}, 0.1, 0.9, 0.999, 1e-8)

	for i := 0; i < 1000; i++ {
		// grad = 2x
		p.Grad = tensor.MulScalar(p.Data, 2.0)
		adam.Step()
	}

	// Should converge close to 0
	if math.Abs(p.Data.At(0)) > 0.1 {
		t.Fatalf("Adam did not converge: x = %f", p.Data.At(0))
	}
}
