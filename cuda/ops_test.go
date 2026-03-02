//go:build gpu

package cuda

import (
	"github.com/djeday123/gotorch/tensor"
	"math"
	"testing"
)

// tolerance for float64 comparisons
const gpuTol = 1e-9

func almostEqualGPU(a, b float64) bool {
	return math.Abs(a-b) < gpuTol
}

// allCloseSlices compares two float64 slices element-wise.
func allCloseSlices(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Phase 1: Device + memory tests
// ---------------------------------------------------------------------------

func TestGPUInit(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	dev, err := Init(0)
	if err != nil {
		t.Fatalf("Init(0) failed: %v", err)
	}
	if dev < 0 {
		t.Fatalf("expected valid device index, got %d", dev)
	}
	name := DeviceName(dev)
	if name == "" {
		t.Fatal("DeviceName returned empty string")
	}
	t.Logf("Device %d: %s", dev, name)
}

func TestGPUMemoryInfo(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	free, total := MemoryInfo()
	if total == 0 {
		t.Fatal("total GPU memory is 0")
	}
	if free > total {
		t.Fatalf("free (%d) > total (%d) — impossible", free, total)
	}
	t.Logf("GPU memory: %.1f / %.1f GB", float64(free)/1e9, float64(total)/1e9)
}

func TestGPUMalloc(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	ptr := Malloc(1024)
	if ptr == nil {
		t.Fatal("Malloc(1024) returned nil")
	}
	Free(ptr) // should not panic
}

func TestDetectGPU(t *testing.T) {
	if !DetectGPU() {
		t.Skip("DetectGPU returned false, skipping")
	}
	info := DeviceInfo()
	if info == "" || info == "no GPU" {
		t.Fatalf("DeviceInfo returned unexpected: %q", info)
	}
	t.Log("DeviceInfo:", info)
}

// ---------------------------------------------------------------------------
// Phase 1: H2D / D2H round-trip
// ---------------------------------------------------------------------------

func TestH2D_D2H_RoundTrip(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	cpu := tensor.New(data, []int{2, 4})

	gt, err := NewGPUTensor(cpu)
	if err != nil {
		t.Fatalf("NewGPUTensor: %v", err)
	}
	defer gt.Free()

	result := gt.ToCPU()
	got := result.Data()
	if !allCloseSlices(got, data, gpuTol) {
		t.Fatalf("H2D/D2H round-trip failed:\n got:  %v\n want: %v", got, data)
	}
	if result.Shape()[0] != 2 || result.Shape()[1] != 4 {
		t.Fatalf("shape mismatch after D2H: %v", result.Shape())
	}
}

// ---------------------------------------------------------------------------
// Phase 2: Elementwise ops — compare against CPU reference
// ---------------------------------------------------------------------------

func TestGPUAdd(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	aData := []float64{1, 2, 3, 4}
	xData := []float64{10, 20, 30, 40}
	want := []float64{11, 22, 33, 44}

	ga, _ := NewGPUTensor(tensor.New(aData, []int{4}))
	defer ga.Free()
	gx, _ := NewGPUTensor(tensor.New(xData, []int{4}))
	defer gx.Free()

	out, err := b.Add(ga, gx)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	defer out.Free()

	got := out.ToCPU().Data()
	if !allCloseSlices(got, want, gpuTol) {
		t.Fatalf("Add: got %v, want %v", got, want)
	}
}

func TestGPUReLU(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	data := []float64{-3, -1, 0, 1, 3}
	want := []float64{0, 0, 0, 1, 3}

	g, _ := NewGPUTensor(tensor.New(data, []int{5}))
	defer g.Free()

	out, err := b.ReLU(g)
	if err != nil {
		t.Fatalf("ReLU: %v", err)
	}
	defer out.Free()

	got := out.ToCPU().Data()
	if !allCloseSlices(got, want, gpuTol) {
		t.Fatalf("ReLU: got %v, want %v", got, want)
	}
}

func TestGPUSigmoid(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	// sigmoid(0) = 0.5
	data := []float64{0.0}
	g, _ := NewGPUTensor(tensor.New(data, []int{1}))
	defer g.Free()

	out, err := b.Sigmoid(g)
	if err != nil {
		t.Fatalf("Sigmoid: %v", err)
	}
	defer out.Free()

	got := out.ToCPU().Data()[0]
	if !almostEqualGPU(got, 0.5) {
		t.Fatalf("Sigmoid(0) = %f, want 0.5", got)
	}
}

func TestGPUTanh(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	// tanh(0) = 0
	data := []float64{0.0}
	g, _ := NewGPUTensor(tensor.New(data, []int{1}))
	defer g.Free()

	out, err := b.Tanh(g)
	if err != nil {
		t.Fatalf("Tanh: %v", err)
	}
	defer out.Free()

	got := out.ToCPU().Data()[0]
	if !almostEqualGPU(got, 0.0) {
		t.Fatalf("Tanh(0) = %f, want 0.0", got)
	}
}

func TestGPUSum(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	data := []float64{1, 2, 3, 4, 5, 6}
	g, _ := NewGPUTensor(tensor.New(data, []int{6}))
	defer g.Free()

	got := b.Sum(g)
	if !almostEqualGPU(got, 21.0) {
		t.Fatalf("Sum = %f, want 21.0", got)
	}
}

// ---------------------------------------------------------------------------
// Phase 2: MatMul — compare GPU result with CPU tensor.MatMul
// ---------------------------------------------------------------------------

func TestGPUMatMul(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	b, _ := NewGPUBackend(0)

	aData := []float64{1, 2, 3, 4, 5, 6}           // [2x3]
	bData := []float64{7, 8, 9, 10, 11, 12}          // [3x2]
	// Reference (CPU):
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58,  64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	want := []float64{58, 64, 139, 154}

	cpuA := tensor.New(aData, []int{2, 3})
	cpuB := tensor.New(bData, []int{3, 2})

	ga, _ := NewGPUTensor(cpuA)
	defer ga.Free()
	gb, _ := NewGPUTensor(cpuB)
	defer gb.Free()

	out, err := b.MatMul(ga, gb)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	defer out.Free()

	got := out.ToCPU().Data()
	if !allCloseSlices(got, want, 1e-6) {
		t.Fatalf("MatMul:\n got:  %v\n want: %v", got, want)
	}
	if out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Fatalf("MatMul output shape = %v, want [2,2]", out.Shape())
	}
}

// ---------------------------------------------------------------------------
// Phase 3: Multi-arch smoke test — just verify we can init and run
// ---------------------------------------------------------------------------

func TestMultiArchSmoke(t *testing.T) {
	if DeviceCount() == 0 {
		t.Skip("no CUDA device available")
	}
	Init(0)
	info := DeviceInfo()
	t.Logf("Running on: %s", info)

	// Quick sanity: upload, compute, download
	data := make([]float64, 1024)
	for i := range data {
		data[i] = float64(i)
	}
	cpu := tensor.New(data, []int{1024})
	g, err := NewGPUTensor(cpu)
	if err != nil {
		t.Fatalf("upload failed: %v", err)
	}
	defer g.Free()

	b, _ := NewGPUBackend(0)
	out, err := b.MulScalar(g, 2.0)
	if err != nil {
		t.Fatalf("MulScalar failed: %v", err)
	}
	defer out.Free()

	result := out.ToCPU().Data()
	if !almostEqualGPU(result[512], data[512]*2) {
		t.Fatalf("MulScalar[512] = %f, want %f", result[512], data[512]*2)
	}
	t.Log("Multi-arch smoke test passed.")
}
