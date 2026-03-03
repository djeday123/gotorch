package nn

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ─────────────────────────────────────────────────────────────────────────────
// Single-device DataParallel (transparent pass-through)
// ─────────────────────────────────────────────────────────────────────────────

func TestDataParallelSingleDevice(t *testing.T) {
	model := NewSequential(NewLinear(4, 4, true))
	dp := NewDataParallel(model, []int{0})

	x := autograd.NewVar(tensor.New([]float64{1, 0, 0, 0}, []int{1, 4}), false)

	out1 := model.Forward(x)
	out2 := dp.Forward(x)

	d1 := out1.Data.Data()
	d2 := out2.Data.Data()
	if len(d1) != len(d2) {
		t.Fatalf("output length mismatch: %d vs %d", len(d1), len(d2))
	}
	for i := range d1 {
		if math.Abs(d1[i]-d2[i]) > 1e-12 {
			t.Errorf("output[%d]: single-device DP differs from direct forward: %g vs %g", i, d2[i], d1[i])
		}
	}
}

func TestDataParallelParameters(t *testing.T) {
	model := NewSequential(NewLinear(4, 4, true))
	dp := NewDataParallel(model, []int{0})
	if len(dp.Parameters()) != len(model.Parameters()) {
		t.Errorf("parameter count mismatch: %d vs %d", len(dp.Parameters()), len(model.Parameters()))
	}
}

func TestDataParallelZeroGrad(t *testing.T) {
	model := NewSequential(NewLinear(4, 4, true))
	dp := NewDataParallel(model, []int{0})

	// Set some fake gradient.
	for _, p := range dp.Parameters() {
		if p.Grad == nil {
			p.Grad = tensor.Ones(p.Data.Shape()...)
		}
	}
	dp.ZeroGrad()
	for _, p := range dp.Parameters() {
		if p.Grad != nil {
			for _, v := range p.Grad.Data() {
				if v != 0 {
					t.Errorf("gradient not zeroed after ZeroGrad")
				}
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-device DataParallel (simulated — sequential CPU execution)
// ─────────────────────────────────────────────────────────────────────────────

func TestDataParallelMultiDevice_Shape(t *testing.T) {
	// With 2 "devices" and batch=4, output batch must still be 4.
	model := NewSequential(NewLinear(8, 4, true))
	dp := NewDataParallel(model, []int{0, 1})

	batch := tensor.New(make([]float64, 4*8), []int{4, 8})
	x := autograd.NewVar(batch, false)
	out := dp.Forward(x)

	if out.Data.Shape()[0] != 4 {
		t.Errorf("expected batch=4, got %d", out.Data.Shape()[0])
	}
	if out.Data.Shape()[1] != 4 {
		t.Errorf("expected out=4, got %d", out.Data.Shape()[1])
	}
}

func TestDataParallelMultiDevice_MatchesSingle(t *testing.T) {
	// Multi-device (sequential simulation) must produce the same result as
	// running the model directly — since all shards use the same weights.
	model := NewSequential(NewLinear(4, 2, false))
	dp := NewDataParallel(model, []int{0, 1})

	batchData := make([]float64, 4*4)
	for i := range batchData {
		batchData[i] = float64(i) * 0.1
	}
	x := autograd.NewVar(tensor.New(batchData, []int{4, 4}), false)

	outDirect := model.Forward(x)
	outDP := dp.Forward(x)

	d1 := outDirect.Data.Data()
	d2 := outDP.Data.Data()
	if len(d1) != len(d2) {
		t.Fatalf("length mismatch: %d vs %d", len(d1), len(d2))
	}
	for i := range d1 {
		if math.Abs(d1[i]-d2[i]) > 1e-12 {
			t.Errorf("output[%d]: %g vs %g", i, d2[i], d1[i])
		}
	}
}

func TestDataParallelSmallBatch(t *testing.T) {
	// Batch smaller than n devices falls back to primary device.
	model := NewSequential(NewLinear(4, 2, true))
	dp := NewDataParallel(model, []int{0, 1, 2, 3})

	x := autograd.NewVar(tensor.New(make([]float64, 2*4), []int{2, 4}), false)
	out := dp.Forward(x)
	if out.Data.Shape()[0] != 2 {
		t.Errorf("expected batch=2 (fallback), got %d", out.Data.Shape()[0])
	}
}

func TestDataParallelString(t *testing.T) {
	model := NewSequential(NewLinear(2, 2, true))
	dp := NewDataParallel(model, []int{0, 1})
	s := dp.String()
	if s == "" {
		t.Error("String() returned empty")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// splitBatch / gatherBatch helpers
// ─────────────────────────────────────────────────────────────────────────────

func TestSplitBatch(t *testing.T) {
	data := make([]float64, 6*3)
	for i := range data {
		data[i] = float64(i)
	}
	x := autograd.NewVar(tensor.New(data, []int{6, 3}), false)
	shards := splitBatch(x, 3)

	if len(shards) != 3 {
		t.Fatalf("expected 3 shards, got %d", len(shards))
	}
	for _, s := range shards {
		if s.Data.Shape()[0] != 2 {
			t.Errorf("shard rows: got %d, want 2", s.Data.Shape()[0])
		}
	}
}

func TestGatherBatch(t *testing.T) {
	a := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}), false)
	b := autograd.NewVar(tensor.New([]float64{5, 6, 7, 8}, []int{2, 2}), false)
	out := gatherBatch([]*autograd.Variable{a, b})

	if out.Data.Shape()[0] != 4 {
		t.Errorf("gathered rows: got %d, want 4", out.Data.Shape()[0])
	}
	expected := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	for i, v := range out.Data.Data() {
		if v != expected[i] {
			t.Errorf("gathered[%d]: %g, want %g", i, v, expected[i])
		}
	}
}

func TestNewDataParallelPanicsOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty deviceIDs")
		}
	}()
	NewDataParallel(NewSequential(), []int{})
}
