package autograd

import (
	"testing"

	"github.com/djeday123/gotorch/tensor"
)

func TestNoGrad(t *testing.T) {
	x := NewVar(tensor.New([]float64{2, 3}, []int{2}), true)
	y := NewVar(tensor.New([]float64{1, 1}, []int{2}), false)

	var result *Variable
	NoGrad(func() {
		result = Add(x, y)
	})

	// Under no_grad: result should be a leaf with no grad_fn
	if result.RequiresGrad {
		t.Error("expected RequiresGrad=false under NoGrad")
	}
	// Data should still be correct
	d := result.Data.Data()
	if d[0] != 3 || d[1] != 4 {
		t.Errorf("wrong data: %v", d)
	}
}

func TestNoGradNested(t *testing.T) {
	if !IsGradEnabled() {
		t.Fatal("grad should be enabled at start")
	}
	NoGrad(func() {
		if IsGradEnabled() {
			t.Error("grad should be disabled inside NoGrad")
		}
		// Re-enable inside
		EnableGrad(func() {
			if !IsGradEnabled() {
				t.Error("grad should be re-enabled inside EnableGrad")
			}
		})
		// Should be disabled again
		if IsGradEnabled() {
			t.Error("grad should be disabled after EnableGrad returns")
		}
	})
	if !IsGradEnabled() {
		t.Fatal("grad should be enabled after NoGrad returns")
	}
}

func TestNoGradNoBackward(t *testing.T) {
	x := NewVar(tensor.New([]float64{3}, []int{1}), true)

	var loss *Variable
	NoGrad(func() {
		loss = Mul(x, x) // x^2
	})

	// Should not panic — loss is a leaf, backward is a no-op on leaf
	// (In our impl, backward on a leaf with no gradFn does nothing)
	// Just verify data is correct
	if loss.Data.Item() != 9 {
		t.Errorf("expected 9, got %v", loss.Data.Item())
	}
}
