package export

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/nn"
)

// ─────────────────────────────────────────────────────────────────────────────
// ExportONNX
// ─────────────────────────────────────────────────────────────────────────────

func TestExportONNX_WritesFile(t *testing.T) {
	model := nn.NewSequential(
		nn.NewLinear(4, 8, true),
		nn.NewReLU(),
		nn.NewLinear(8, 2, true),
	)
	path := filepath.Join(t.TempDir(), "model.onnx")
	if err := ExportONNX(model, []int{1, 4}, path); err != nil {
		t.Fatalf("ExportONNX error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file not created: %v", err)
	}
	if info.Size() == 0 {
		t.Error("ONNX file is empty")
	}
}

func TestExportONNX_MagicBytes(t *testing.T) {
	// A valid protobuf file starts with field 1 (ir_version) varint tag = 0x08.
	model := nn.NewSequential(nn.NewLinear(2, 2, false))
	path := filepath.Join(t.TempDir(), "model.onnx")
	if err := ExportONNX(model, []int{1, 2}, path); err != nil {
		t.Fatalf("ExportONNX error: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read file: %v", err)
	}
	if len(data) < 2 {
		t.Fatal("file too short")
	}
	// First byte should be 0x08 (field 1, wire type 0 = varint = ir_version).
	if data[0] != 0x08 {
		t.Errorf("first byte: got 0x%02x, want 0x08 (ir_version field tag)", data[0])
	}
	// Second byte: ir_version = 8.
	if data[1] != 0x08 {
		t.Errorf("ir_version value: got %d, want 8", data[1])
	}
}

func TestExportONNX_AllLayers(t *testing.T) {
	// Test that all supported layer types export without error.
	model := nn.NewSequential(
		nn.NewLinear(4, 8, true),
		nn.NewReLU(),
		nn.NewLinear(8, 8, false),
		nn.NewSigmoid(),
		nn.NewLinear(8, 8, false),
		nn.NewTanh(),
		nn.NewDropout(0.5),
		nn.NewLinear(8, 2, true),
	)
	path := filepath.Join(t.TempDir(), "all_layers.onnx")
	if err := ExportONNX(model, []int{1, 4}, path); err != nil {
		t.Fatalf("ExportONNX error: %v", err)
	}
}

func TestExportONNX_NoBias(t *testing.T) {
	// Linear without bias should export without panicking.
	model := nn.NewSequential(nn.NewLinear(4, 4, false))
	path := filepath.Join(t.TempDir(), "nobias.onnx")
	if err := ExportONNX(model, []int{1, 4}, path); err != nil {
		t.Fatalf("ExportONNX error: %v", err)
	}
}

func TestExportONNX_UnsupportedLayer(t *testing.T) {
	// An unsupported layer type must return an error, not panic.
	unsupported := &unsupportedLayer{}
	model2 := nn.NewSequential(unsupported)

	path := filepath.Join(t.TempDir(), "bad.onnx")
	err := ExportONNX(model2, []int{1, 4}, path)
	if err == nil {
		t.Error("expected error for unsupported layer type, got nil")
	}
}

func TestExportONNX_LargeModel(t *testing.T) {
	// A deeper model should export without error.
	layers := []nn.Module{}
	inFeat := 64
	for i := 0; i < 6; i++ {
		layers = append(layers, nn.NewLinear(inFeat, inFeat, true))
		layers = append(layers, nn.NewReLU())
	}
	model := nn.NewSequential(layers...)
	path := filepath.Join(t.TempDir(), "large.onnx")
	if err := ExportONNX(model, []int{1, 64}, path); err != nil {
		t.Fatalf("ExportONNX error: %v", err)
	}
	info, _ := os.Stat(path)
	if info.Size() < 1000 {
		t.Errorf("large model ONNX file too small: %d bytes", info.Size())
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// unsupportedLayer satisfies nn.Module but is not handled by ExportONNX.
type unsupportedLayer struct{}

func (u *unsupportedLayer) Forward(x *autograd.Variable) *autograd.Variable { return x }
func (u *unsupportedLayer) Parameters() []*autograd.Variable                 { return nil }
func (u *unsupportedLayer) ZeroGrad()                                        {}
