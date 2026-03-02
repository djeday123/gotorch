package nn

import (
	"math"
	"os"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ── BatchNorm2d ─────────────────────────────────────────────────────────────

func TestBatchNorm2dShape(t *testing.T) {
	bn := NewBatchNorm2d(4)
	x := autograd.NewVar(tensor.RandN(2, 4, 8, 8), false)
	out := bn.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 2 || shape[1] != 4 || shape[2] != 8 || shape[3] != 8 {
		t.Fatalf("wrong shape: %v", shape)
	}
}

func TestBatchNorm2dNormalized(t *testing.T) {
	// With gamma=1, beta=0, output should be approx N(0,1) per channel
	bn := NewBatchNorm2d(2)
	// Make x with known mean/var per channel
	data := make([]float64, 2*2*4*4) // [2,2,4,4]
	for i := range data {
		data[i] = float64(i % 10)
	}
	x := autograd.NewVar(tensor.New(data, []int{2, 2, 4, 4}), false)
	out := bn.Forward(x)
	// Each channel should have mean≈0 and std≈1
	outData := out.Data.Data()
	N, C, H, W := 2, 2, 4, 4
	for c := 0; c < C; c++ {
		sum := 0.0
		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					sum += outData[n*C*H*W+c*H*W+h*W+w]
				}
			}
		}
		mean := sum / float64(N*H*W)
		if math.Abs(mean) > 1e-9 {
			t.Errorf("channel %d: mean=%v, expected ~0", c, mean)
		}
	}
}

func TestBatchNorm2dRunningStats(t *testing.T) {
	bn := NewBatchNorm2d(2)
	x := autograd.NewVar(tensor.RandN(4, 2, 3, 3), false)
	bn.Forward(x)
	// Running stats should be updated
	updated := false
	for _, v := range bn.RunningMean {
		if v != 0 {
			updated = true
		}
	}
	if !updated {
		t.Error("RunningMean should be non-zero after a forward pass")
	}
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

func TestLayerNormShape(t *testing.T) {
	ln := NewLayerNorm([]int{8})
	x := autograd.NewVar(tensor.RandN(4, 8), false)
	out := ln.Forward(x)
	if out.Data.Shape()[0] != 4 || out.Data.Shape()[1] != 8 {
		t.Fatalf("wrong shape: %v", out.Data.Shape())
	}
}

func TestLayerNormNormalized(t *testing.T) {
	ln := NewLayerNorm([]int{16})
	x := autograd.NewVar(tensor.RandN(10, 16), false)
	out := ln.Forward(x)
	outData := out.Data.Data()
	// Each row should have mean≈0
	for r := 0; r < 10; r++ {
		sum := 0.0
		for c := 0; c < 16; c++ {
			sum += outData[r*16+c]
		}
		mean := sum / 16.0
		if math.Abs(mean) > 1e-9 {
			t.Errorf("row %d: mean=%v, expected ~0", r, mean)
		}
	}
}

// ── Dropout ──────────────────────────────────────────────────────────────────

func TestDropoutShape(t *testing.T) {
	d := NewDropout(0.5)
	x := autograd.NewVar(tensor.RandN(4, 8), false)
	out := d.Forward(x)
	if out.Data.Shape()[0] != 4 || out.Data.Shape()[1] != 8 {
		t.Fatalf("wrong shape: %v", out.Data.Shape())
	}
}

func TestDropoutEvalPassthrough(t *testing.T) {
	d := NewDropout(0.9)
	d.Eval()
	x := autograd.NewVar(tensor.Ones(10), false)
	out := d.Forward(x)
	// In eval mode, output == input exactly
	for i, v := range out.Data.Data() {
		if v != 1.0 {
			t.Errorf("eval dropout: index %d = %v, want 1.0", i, v)
		}
	}
}

func TestDropoutScaling(t *testing.T) {
	// With p=0.5, surviving elements should be scaled by 2.0
	d := NewDropout(0.5)
	x := autograd.NewVar(tensor.Ones(1000), false)
	out := d.Forward(x)
	data := out.Data.Data()
	for _, v := range data {
		if v != 0.0 && math.Abs(v-2.0) > 1e-9 {
			t.Fatalf("expected 0 or 2, got %v", v)
		}
	}
}

// ── Save / Load ──────────────────────────────────────────────────────────────

func TestSaveLoad(t *testing.T) {
	model := NewLinear(4, 2, true)
	params := model.Parameters()

	// Save original weights
	origW := make([]float64, len(params[0].Data.Data()))
	copy(origW, params[0].Data.Data())

	path := "/tmp/gotorch_test_weights.json"
	defer os.Remove(path)

	if err := Save(params, path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Modify weights in place
	for _, p := range params {
		d := p.Data.Data()
		for i := range d {
			d[i] = 0
		}
		p.Data = tensor.New(d, p.Data.Shape())
	}

	// Load back
	if err := Load(params, path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Check restored
	restored := params[0].Data.Data()
	for i, v := range origW {
		if math.Abs(restored[i]-v) > 1e-12 {
			t.Errorf("param[0][%d]: got %v want %v", i, restored[i], v)
		}
	}
}

func TestLoadShapeMismatch(t *testing.T) {
	model1 := NewLinear(4, 2, true)
	model2 := NewLinear(8, 4, true)
	path := "/tmp/gotorch_test_shape.json"
	defer os.Remove(path)

	Save(model1.Parameters(), path)
	err := Load(model2.Parameters(), path)
	if err == nil {
		t.Error("expected shape mismatch error, got nil")
	}
}
