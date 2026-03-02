package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

func TestConv2dShape(t *testing.T) {
	// Input [N=1, C=1, H=5, W=5], kernel 3x3, stride 1, padding 0
	// Output: [1, 2, 3, 3]
	conv := NewConv2d(1, 2, 3, 1, 0, true)
	x := autograd.NewVar(tensor.RandN(1, 1, 5, 5), false)
	out := conv.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 3 || shape[3] != 3 {
		t.Fatalf("wrong output shape: %v, expected [1 2 3 3]", shape)
	}
}

func TestConv2dShapeWithPadding(t *testing.T) {
	// Input [1, 3, 8, 8], kernel 3x3, stride 1, padding 1
	// Output: [1, 8, 8, 8]
	conv := NewConv2d(3, 8, 3, 1, 1, true)
	x := autograd.NewVar(tensor.RandN(1, 3, 8, 8), false)
	out := conv.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 1 || shape[1] != 8 || shape[2] != 8 || shape[3] != 8 {
		t.Fatalf("wrong output shape: %v, expected [1 8 8 8]", shape)
	}
}

func TestConv2dNoBias(t *testing.T) {
	conv := NewConv2d(1, 1, 3, 1, 0, false)
	if len(conv.Parameters()) != 1 {
		t.Fatalf("expected 1 parameter (no bias), got %d", len(conv.Parameters()))
	}
}

func TestConv2dGradNumerical(t *testing.T) {
	// Numerical gradient check on a small conv
	conv := NewConv2d(1, 1, 2, 1, 0, false)
	// Fix weights to known values
	w := tensor.New([]float64{1, 0, 0, 1}, []int{1, 1, 2, 2})
	conv.Weight = autograd.NewVar(w, true)

	xData := tensor.New([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 1, 3, 3})
	x := autograd.NewVar(xData, false)

	out := conv.Forward(x)
	// Output should be [1,1,2,2]:
	// top-left: 1*1+2*0+4*0+5*1 = 6
	// top-right: 2*1+3*0+5*0+6*1 = 8
	// bot-left: 4*1+5*0+7*0+8*1 = 12
	// bot-right: 5*1+6*0+8*0+9*1 = 14
	expected := []float64{6, 8, 12, 14}
	got := out.Data.Data()
	for i, v := range expected {
		if math.Abs(got[i]-v) > 1e-9 {
			t.Errorf("output[%d] = %v, want %v", i, got[i], v)
		}
	}

	// Backward
	loss := autograd.Sum(out)
	loss.Backward()

	// dW should be sum of input patches:
	// patch(0,0)=[1,2,4,5] patch(0,1)=[2,3,5,6] patch(1,0)=[4,5,7,8] patch(1,1)=[5,6,8,9]
	// All grad_out = 1, so dW = sum of patches = [12,16,24,28]
	dwExpected := []float64{12, 16, 24, 28}
	dw := conv.Weight.Grad.Data()
	for i, v := range dwExpected {
		if math.Abs(dw[i]-v) > 1e-9 {
			t.Errorf("dW[%d] = %v, want %v", i, dw[i], v)
		}
	}
}

func TestMaxPool2dShape(t *testing.T) {
	// Input [1, 1, 4, 4], kernel 2, stride 2 → output [1, 1, 2, 2]
	pool := NewMaxPool2d(2, 2)
	x := autograd.NewVar(tensor.RandN(1, 1, 4, 4), false)
	out := pool.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 2 || shape[3] != 2 {
		t.Fatalf("wrong shape: %v, expected [1 1 2 2]", shape)
	}
}

func TestMaxPool2dValues(t *testing.T) {
	pool := NewMaxPool2d(2, 2)
	x := autograd.NewVar(tensor.New([]float64{
		1, 3, 2, 4,
		5, 6, 1, 2,
		7, 8, 9, 0,
		1, 2, 3, 4,
	}, []int{1, 1, 4, 4}), true)

	out := pool.Forward(x)
	got := out.Data.Data()
	want := []float64{6, 4, 8, 9}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("output[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestMaxPool2dGrad(t *testing.T) {
	pool := NewMaxPool2d(2, 2)
	x := autograd.NewVar(tensor.New([]float64{
		1, 3,
		5, 2,
	}, []int{1, 1, 2, 2}), true)
	out := pool.Forward(x)
	loss := autograd.Sum(out)
	loss.Backward()

	// Max is at (1,0) = 5, so grad there is 1, rest 0
	dX := x.Grad.Data()
	if dX[0] != 0 || dX[1] != 0 || dX[2] != 1 || dX[3] != 0 {
		t.Errorf("wrong gradient: %v, expected [0 0 1 0]", dX)
	}
}

func TestFlatten2d(t *testing.T) {
	x := autograd.NewVar(tensor.RandN(2, 3, 4, 4), false)
	flat := Flatten2d(x)
	shape := flat.Data.Shape()
	if shape[0] != 2 || shape[1] != 48 {
		t.Fatalf("wrong shape: %v, expected [2 48]", shape)
	}
}
