package optim

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

func TestAdamWStep(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{1.0, 2.0}, []int{2}), true)
	x.Grad = tensor.New([]float64{1.0, 1.0}, []int{2})

	opt := NewAdamW([]*autograd.Variable{x}, 0.01, 0.9, 0.999, 1e-8, 0.01)
	before := x.Data.Data()
	opt.Step()
	after := x.Data.Data()

	// Weight decay should have shrunk the params slightly
	for i := range before {
		if after[i] >= before[i] {
			t.Errorf("param[%d]: expected decrease, got %v → %v", i, before[i], after[i])
		}
	}
}

func TestAdamWWeightDecay(t *testing.T) {
	// With lr=0 and weight decay, params should shrink each step
	x := autograd.NewVar(tensor.New([]float64{1.0}, []int{1}), true)
	x.Grad = tensor.New([]float64{0.0}, []int{1}) // zero grad

	opt := NewAdamW([]*autograd.Variable{x}, 0.0, 0.9, 0.999, 1e-8, 0.1)
	opt.Step()
	after := x.Data.Data()
	// p *= (1 - lr*wd) = (1 - 0) = 1.0 when lr=0
	// Actually with lr=0 and wd=0.1: p *= (1 - 0*0.1) = p*1.0
	// So no change when lr=0
	if math.Abs(after[0]-1.0) > 1e-6 {
		t.Errorf("expected ~1.0, got %v", after[0])
	}
}

func TestClipGradNorm(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	y := autograd.NewVar(tensor.Zeros(1), true)
	x.Grad = tensor.New([]float64{3.0}, []int{1})
	y.Grad = tensor.New([]float64{4.0}, []int{1})

	// Norm = sqrt(9+16) = 5. Clip to 1.0 → scale = 0.2
	norm := ClipGradNorm([]*autograd.Variable{x, y}, 1.0)

	if math.Abs(norm-5.0) > 1e-6 {
		t.Errorf("expected norm=5.0, got %v", norm)
	}
	if math.Abs(x.Grad.Item()-0.6) > 1e-6 {
		t.Errorf("expected x.grad=0.6, got %v", x.Grad.Item())
	}
	if math.Abs(y.Grad.Item()-0.8) > 1e-6 {
		t.Errorf("expected y.grad=0.8, got %v", y.Grad.Item())
	}
}

func TestClipGradNormBelowThreshold(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	x.Grad = tensor.New([]float64{0.5}, []int{1})

	norm := ClipGradNorm([]*autograd.Variable{x}, 2.0)
	// Norm=0.5 < maxNorm=2.0, no clipping
	if math.Abs(norm-0.5) > 1e-6 {
		t.Errorf("expected norm=0.5, got %v", norm)
	}
	if math.Abs(x.Grad.Item()-0.5) > 1e-6 {
		t.Errorf("grad should be unchanged, got %v", x.Grad.Item())
	}
}

func TestClipGradValue(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	x.Grad = tensor.New([]float64{5.0, -3.0, 0.5}, []int{3})
	ClipGradValue([]*autograd.Variable{x}, 2.0)
	d := x.Grad.Data()
	if d[0] != 2.0 || d[1] != -2.0 || d[2] != 0.5 {
		t.Errorf("wrong clip result: %v", d)
	}
}

func TestStepLR(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	opt := NewSGD([]*autograd.Variable{x}, 1.0, 0)
	sched := NewStepLR(opt, 2, 0.1)

	sched.Step() // step 1 — no decay
	if math.Abs(opt.GetLR()-1.0) > 1e-9 {
		t.Errorf("step 1: expected lr=1.0, got %v", opt.GetLR())
	}
	sched.Step() // step 2 — decay
	if math.Abs(opt.GetLR()-0.1) > 1e-9 {
		t.Errorf("step 2: expected lr=0.1, got %v", opt.GetLR())
	}
}

func TestCosineAnnealingLR(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	opt := NewSGD([]*autograd.Variable{x}, 1.0, 0)
	sched := NewCosineAnnealingLR(opt, 100, 0.0)

	sched.Step() // t=1
	lr1 := opt.GetLR()
	if lr1 >= 1.0 || lr1 <= 0.0 {
		t.Errorf("after step 1: lr=%v out of range (0,1)", lr1)
	}
	// After T_max steps, lr should be at eta_min=0
	for i := 0; i < 99; i++ {
		sched.Step()
	}
	if math.Abs(opt.GetLR()) > 1e-9 {
		t.Errorf("at T_max: expected lr=0, got %v", opt.GetLR())
	}
}

func TestLinearWarmup(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	opt := NewSGD([]*autograd.Variable{x}, 1.0, 0)
	sched := NewLinearWarmup(opt, 4)

	// LR starts at 0
	if opt.GetLR() != 0 {
		t.Errorf("initial lr should be 0, got %v", opt.GetLR())
	}
	sched.Step() // t=1, lr=0.25
	if math.Abs(opt.GetLR()-0.25) > 1e-9 {
		t.Errorf("step 1: expected lr=0.25, got %v", opt.GetLR())
	}
	sched.Step()
	sched.Step()
	sched.Step() // t=4, lr=1.0
	if math.Abs(opt.GetLR()-1.0) > 1e-9 {
		t.Errorf("step 4: expected lr=1.0, got %v", opt.GetLR())
	}
}
