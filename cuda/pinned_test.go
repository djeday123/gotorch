//go:build gpu

package cuda

import (
	"math"
	"testing"
)

// TestPinnedAlloc: allocate, write via Slice(), read back.
func TestPinnedAlloc(t *testing.T) {
	if _, err := Init(0); err != nil {
		t.Skip("no GPU:", err)
	}

	p, err := NewPinnedTensor(4)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Free()

	sl := p.Slice()
	if len(sl) != 4 {
		t.Fatalf("expected slice len 4, got %d", len(sl))
	}

	sl[0] = 1.0
	sl[1] = 2.0
	sl[2] = 3.0
	sl[3] = 4.0

	sl2 := p.Slice()
	for i, want := range []float64{1, 2, 3, 4} {
		if sl2[i] != want {
			t.Errorf("index %d: got %v, want %v", i, sl2[i], want)
		}
	}
}

// TestPinnedZeroCopy: write via Slice() → ToGPU → read back via D2H directly.
// Verifies the GPU saw the values written through the pinned pointer (zero-copy).
func TestPinnedZeroCopy(t *testing.T) {
	if _, err := Init(0); err != nil {
		t.Skip("no GPU:", err)
	}

	const n = 8
	p, err := NewPinnedTensor(n)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Free()

	// Write values via zero-copy slice
	sl := p.Slice()
	for i := range sl {
		sl[i] = float64(i + 1)
	}

	// Upload to GPU (async DMA from pinned memory)
	g, err := p.ToGPU()
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()

	// Read back from GPU via regular D2H
	got := make([]float64, n)
	D2H(got, g.ptr, n)

	for i, want := range sl {
		if math.Abs(got[i]-want) > 1e-12 {
			t.Errorf("index %d: got %v, want %v", i, got[i], want)
		}
	}
}

// TestPinnedRoundTrip: write → ToGPU → FromGPU → read via Slice().
func TestPinnedRoundTrip(t *testing.T) {
	if _, err := Init(0); err != nil {
		t.Skip("no GPU:", err)
	}

	const n = 1024
	src, err := NewPinnedTensor(n)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Free()

	dst, err := NewPinnedTensor(n)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()

	// Fill source
	for i, s := 0, src.Slice(); i < n; i++ {
		s[i] = float64(i) * 0.5
	}

	// Upload
	g, err := src.ToGPU()
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()

	// Download into dst
	if err := dst.FromGPU(g); err != nil {
		t.Fatal(err)
	}

	// Verify
	s, d := src.Slice(), dst.Slice()
	for i := 0; i < n; i++ {
		if math.Abs(d[i]-s[i]) > 1e-12 {
			t.Errorf("index %d: got %v, want %v", i, d[i], s[i])
		}
	}
}

// TestPinnedToCPUTensor: verify ToCPUTensor copies correctly.
func TestPinnedToCPUTensor(t *testing.T) {
	if _, err := Init(0); err != nil {
		t.Skip("no GPU:", err)
	}

	p, err := NewPinnedTensor(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Free()

	sl := p.Slice()
	for i := range sl {
		sl[i] = float64(i + 10)
	}

	ct := p.ToCPUTensor()
	if ct.Shape()[0] != 3 || ct.Shape()[1] != 2 {
		t.Fatalf("wrong shape: %v", ct.Shape())
	}
	for i := 0; i < 6; i++ {
		if ct.Data()[i] != float64(i+10) {
			t.Errorf("index %d: got %v, want %v", i, ct.Data()[i], float64(i+10))
		}
	}
}

// NOTE: Transfer benchmarks (BenchmarkPinnedH2D / BenchmarkPageableH2D) are
// excluded — cudaDeviceSynchronize triggers the CUDA TDR watchdog on display
// GPUs (RTX 4090 under X11). Run on a compute-only / headless GPU to benchmark.
