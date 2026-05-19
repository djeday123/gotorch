package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// numericalGrad returns dL/dx where L = sum(fn(x)), via central differences.
func numericalGrad(fn func(*tensor.Tensor) *tensor.Tensor, x *tensor.Tensor, eps float64) *tensor.Tensor {
	grad := tensor.Zeros(x.Shape()...)
	xFlat := x.Data()
	gFlat := grad.Data()
	sumOf := func(t *tensor.Tensor) float64 {
		s := 0.0
		for _, v := range t.Data() {
			s += v
		}
		return s
	}
	for i := range xFlat {
		orig := xFlat[i]
		xFlat[i] = orig + eps
		fp := sumOf(fn(tensor.New(xFlat, x.Shape())))
		xFlat[i] = orig - eps
		fm := sumOf(fn(tensor.New(xFlat, x.Shape())))
		xFlat[i] = orig
		gFlat[i] = (fp - fm) / (2 * eps)
	}
	return tensor.New(gFlat, x.Shape())
}

func maxAbsDiff(a, b *tensor.Tensor) float64 {
	ad := a.Data()
	bd := b.Data()
	max := 0.0
	for i := range ad {
		d := math.Abs(ad[i] - bd[i])
		if d > max {
			max = d
		}
	}
	return max
}

// --- 1. Sum backward ---

func TestSumBackwardCorrect(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}), true)
	loss := autograd.MulScalar(autograd.Sum(x), 3.0) // L = 3*sum(x)
	loss.Backward()
	// expected gradient: 3 everywhere
	expected := []float64{3, 3, 3, 3}
	got := x.Grad.Data()
	for i, v := range got {
		if math.Abs(v-expected[i]) > 1e-9 {
			t.Errorf("Sum grad[%d]: got %v, want %v", i, v, expected[i])
		}
	}
}

// --- 2. Softmax backward ---

func TestSoftmaxBackwardNumerical(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{1.0, 2.0, 0.5, -1.0, 0.3, 1.5}, []int{2, 3}), true)
	loss := autograd.Sum(autograd.Softmax(x, 1))
	loss.Backward()
	analytical := x.Grad

	numerical := numericalGrad(func(xt *tensor.Tensor) *tensor.Tensor {
		return tensor.Softmax(xt, 1)
	}, x.Data, 1e-5)

	diff := maxAbsDiff(analytical, numerical)
	if diff > 1e-5 {
		t.Errorf("Softmax grad: max diff %v exceeds tolerance\n  analytical=%v\n  numerical=%v",
			diff, analytical.Data(), numerical.Data())
	}
}

// --- 3. LayerNorm backward ---

func TestLayerNormBackwardNumerical(t *testing.T) {
	ln := NewLayerNorm([]int{4})
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4, 0.5, -1, 2.5, 0.1}, []int{2, 4}), true)

	loss := autograd.Sum(ln.Forward(x))
	loss.Backward()
	analytical := x.Grad

	numerical := numericalGrad(func(xt *tensor.Tensor) *tensor.Tensor {
		v := autograd.NewVar(xt, false)
		return ln.Forward(v).Data
	}, x.Data, 1e-5)

	diff := maxAbsDiff(analytical, numerical)
	if diff > 1e-4 {
		t.Errorf("LayerNorm grad: max diff %v\n  analytical=%v\n  numerical=%v",
			diff, analytical.Data(), numerical.Data())
	}
}

// --- 4. BatchNorm1d backward ---

func TestBatchNorm1dBackwardNumerical(t *testing.T) {
	bn := NewBatchNorm1d(3)
	bn.Eval() // use running stats (constant) for clean numerical test
	x := autograd.NewVar(tensor.New([]float64{
		1, 2, 3,
		0.5, -1, 2.5,
		2, 0.1, 1.0,
		-0.5, 1.5, 0.3,
	}, []int{4, 3}), true)

	loss := autograd.Sum(bn.Forward(x))
	loss.Backward()
	analytical := x.Grad

	numerical := numericalGrad(func(xt *tensor.Tensor) *tensor.Tensor {
		v := autograd.NewVar(xt, false)
		return bn.Forward(v).Data
	}, x.Data, 1e-5)

	diff := maxAbsDiff(analytical, numerical)
	if diff > 1e-4 {
		t.Errorf("BatchNorm1d (eval) grad: max diff %v", diff)
	}
}

// --- 5. MultiheadAttention backward ---

func TestMHABackwardShape(t *testing.T) {
	// We don't verify exact numerical equality (cross-step coupling makes it complex);
	// just verify gradients flow with correct shapes and are nonzero.
	mha := NewMultiheadAttention(4, 2, false)
	x := autograd.NewVar(tensor.RandN(3, 4), true)

	out := mha.Forward(x)
	loss := autograd.Sum(out)
	loss.Backward()

	if x.Grad == nil {
		t.Fatal("input gradient is nil")
	}
	if mha.WQ.Grad == nil || mha.WK.Grad == nil || mha.WV.Grad == nil || mha.WO.Grad == nil {
		t.Fatal("weight gradients are nil")
	}

	// At least one non-trivial value
	nonZero := false
	for _, v := range x.Grad.Data() {
		if math.Abs(v) > 1e-10 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Error("input gradient is all zeros — attention backward not flowing")
	}
}

// --- 6. LSTM BPTT ---

func TestLSTMBPTTNumerical(t *testing.T) {
	l := NewLSTM(2, 3)
	x := autograd.NewVar(tensor.New([]float64{
		0.1, 0.2,
		0.3, -0.1,
		-0.2, 0.4,
	}, []int{3, 2}), true)

	forward := func() *autograd.Variable {
		outs, _ := l.Forward(x, nil)
		// loss = sum over all timesteps and hidden units
		var loss *autograd.Variable
		for _, o := range outs {
			s := autograd.Sum(o)
			if loss == nil {
				loss = s
			} else {
				loss = autograd.Add(loss, s)
			}
		}
		return loss
	}

	loss := forward()
	loss.Backward()
	gradX := x.Grad

	// Numerical grad w.r.t. x
	numerical := tensor.Zeros(x.Data.Shape()...)
	nFlat := numerical.Data()
	xFlat := x.Data.Data()
	eps := 1e-5
	for i := range xFlat {
		orig := xFlat[i]
		xFlat[i] = orig + eps
		x2 := autograd.NewVar(tensor.New(xFlat, x.Data.Shape()), false)
		outsP, _ := l.Forward(x2, nil)
		fp := 0.0
		for _, o := range outsP {
			for _, v := range o.Data.Data() {
				fp += v
			}
		}
		xFlat[i] = orig - eps
		x3 := autograd.NewVar(tensor.New(xFlat, x.Data.Shape()), false)
		outsM, _ := l.Forward(x3, nil)
		fm := 0.0
		for _, o := range outsM {
			for _, v := range o.Data.Data() {
				fm += v
			}
		}
		xFlat[i] = orig
		nFlat[i] = (fp - fm) / (2 * eps)
	}
	numerical = tensor.New(nFlat, x.Data.Shape())

	diff := maxAbsDiff(gradX, numerical)
	if diff > 1e-3 {
		t.Errorf("LSTM BPTT grad w.r.t. x: max diff %v\n  analytical=%v\n  numerical=%v",
			diff, gradX.Data(), numerical.Data())
	}
}
