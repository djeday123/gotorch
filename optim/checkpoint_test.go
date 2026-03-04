package optim

import (
	"math"
	"os"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

func TestAdamWCheckpointRoundtrip(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{1.0, 2.0, 3.0}, []int{3}), true)
	opt := NewAdamW([]*autograd.Variable{x}, 0.01, 0.9, 0.999, 1e-8, 0.01)

	// Run a few steps
	for i := 0; i < 5; i++ {
		x.Grad = tensor.New([]float64{0.1, 0.2, 0.3}, []int{3})
		opt.Step()
	}

	lr := opt.GetLR()

	// Save
	path := t.TempDir() + "/opt.json"
	if err := SaveOptimizer(opt, path); err != nil {
		t.Fatalf("SaveOptimizer: %v", err)
	}

	// Create a fresh optimizer and load
	y := autograd.NewVar(tensor.New([]float64{1.0, 2.0, 3.0}, []int{3}), true)
	opt2 := NewAdamW([]*autograd.Variable{y}, 0.001, 0.9, 0.999, 1e-8, 0.01)
	if err := LoadOptimizer(opt2, path); err != nil {
		t.Fatalf("LoadOptimizer: %v", err)
	}

	// LR should be restored
	if math.Abs(opt2.GetLR()-lr) > 1e-10 {
		t.Errorf("LR mismatch: saved %v, loaded %v", lr, opt2.GetLR())
	}

	// Step count should match
	state := opt2.GetState()
	if state.Step != 5 {
		t.Errorf("step mismatch: expected 5, got %d", state.Step)
	}

	// First moment should be restored
	origM := opt.GetState().M
	loadedM := state.M
	if len(origM) != len(loadedM) {
		t.Fatalf("M length mismatch")
	}
	for i := range origM {
		for j := range origM[i] {
			if math.Abs(origM[i][j]-loadedM[i][j]) > 1e-12 {
				t.Errorf("M[%d][%d] mismatch: %v vs %v", i, j, origM[i][j], loadedM[i][j])
			}
		}
	}
}

func TestAdamWCheckpointFileNotFound(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(1), true)
	opt := NewAdamW([]*autograd.Variable{x}, 0.01, 0.9, 0.999, 1e-8, 0.01)
	err := LoadOptimizer(opt, "/nonexistent/path/opt.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestAdamWCheckpointPreservesTraining(t *testing.T) {
	// After load, optimizer should continue training correctly
	x := autograd.NewVar(tensor.Scalar(5.0), true)
	opt := NewAdamW([]*autograd.Variable{x}, 0.1, 0.9, 0.999, 1e-8, 0.0)

	// Train partially
	for i := 0; i < 10; i++ {
		xv := x.Data.Item()
		x.Grad = tensor.Scalar(2 * xv)
		opt.Step()
	}
	xAfter10 := x.Data.Item()

	// Save and reload
	path := t.TempDir() + "/opt2.json"
	_ = SaveOptimizer(opt, path)

	x2 := autograd.NewVar(tensor.Scalar(xAfter10), true)
	opt2 := NewAdamW([]*autograd.Variable{x2}, 0.1, 0.9, 0.999, 1e-8, 0.0)
	_ = LoadOptimizer(opt2, path)

	// Both should make a similar step
	xv := x.Data.Item()
	x.Grad = tensor.Scalar(2 * xv)
	opt.Step()
	v1 := x.Data.Item()

	x2v := x2.Data.Item()
	x2.Grad = tensor.Scalar(2 * x2v)
	opt2.Step()
	v2 := x2.Data.Item()

	if math.Abs(v1-v2) > 1e-8 {
		t.Errorf("step after load mismatch: %v vs %v", v1, v2)
	}
}

func TestSaveOptimizerCreatesFile(t *testing.T) {
	x := autograd.NewVar(tensor.Zeros(2), true)
	opt := NewAdamW([]*autograd.Variable{x}, 0.01, 0.9, 0.999, 1e-8, 0.0)

	path := t.TempDir() + "/test_ckpt.json"
	if err := SaveOptimizer(opt, path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file not created: %v", err)
	}
	if info.Size() == 0 {
		t.Error("saved file is empty")
	}
}
