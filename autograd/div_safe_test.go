package autograd

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/tensor"
)

// TestDivBackwardNoNaNAtZero verifies that Div backward returns finite
// gradients even when the denominator contains zeros. Previously this
// produced NaN/Inf which poisoned downstream computation silently.
func TestDivBackwardNoNaNAtZero(t *testing.T) {
	a := NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), true)
	b := NewVar(tensor.New([]float64{0, 1, 0}, []int{3}), true)

	out := Div(a, b)
	loss := Sum(out)
	loss.Backward()

	checkFinite := func(name string, g *tensor.Tensor) {
		t.Helper()
		if g == nil {
			t.Fatalf("%s grad is nil", name)
		}
		for i, v := range g.Data() {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("%s[%d] = %v (must be finite)", name, i, v)
			}
		}
	}
	checkFinite("dA", a.Grad)
	checkFinite("dB", b.Grad)

	// At b=0, dA[i] should be 0 (smooth reciprocal handles this cleanly).
	if v := a.Grad.At(0); v != 0 {
		t.Errorf("dA[0] at b=0: expected 0, got %v", v)
	}
	if v := a.Grad.At(2); v != 0 {
		t.Errorf("dA[2] at b=0: expected 0, got %v", v)
	}
	// At b=1, dA[i] should be ≈ 1 (= 1/b * grad with grad=1).
	if v := a.Grad.At(1); math.Abs(v-1) > 1e-6 {
		t.Errorf("dA[1] at b=1: expected 1, got %v", v)
	}
}
