package tensor

import "testing"

// TestMatMulFloat32Preserved verifies that the dtype of the result matches
// the inputs when both are Float32. Previously MatMul always produced
// Float64 regardless of input dtype.
func TestMatMulFloat32Preserved(t *testing.T) {
	a64 := New([]float64{1, 2, 3, 4}, []int{2, 2})
	b64 := New([]float64{5, 6, 7, 8}, []int{2, 2})
	out64 := MatMul(a64, b64)
	if out64.dtype != Float64 {
		t.Errorf("F64xF64: expected Float64, got %v", out64.dtype)
	}

	a32 := a64.Float32()
	b32 := b64.Float32()
	out32 := MatMul(a32, b32)
	if out32.dtype != Float32 {
		t.Errorf("F32xF32: expected Float32, got %v", out32.dtype)
	}

	// Values must match between the two paths (within F32 precision).
	want := out64.Data()
	got := out32.Data()
	for i := range want {
		if diff := want[i] - got[i]; diff > 1e-4 || diff < -1e-4 {
			t.Errorf("element %d: F64=%v F32=%v diff=%v", i, want[i], got[i], diff)
		}
	}

	// Mixed precision should promote to Float64.
	outMixed := MatMul(a32, b64)
	if outMixed.dtype != Float64 {
		t.Errorf("F32xF64 should promote to Float64, got %v", outMixed.dtype)
	}
}

// TestBatchMatMulFloat32Preserved is the same for BatchMatMul.
func TestBatchMatMulFloat32Preserved(t *testing.T) {
	a := New([]float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, []int{2, 2, 3})
	b := New([]float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, []int{2, 3, 2})

	out64 := BatchMatMul(a, b)
	if out64.dtype != Float64 {
		t.Errorf("F64xF64 batch: expected Float64, got %v", out64.dtype)
	}

	out32 := BatchMatMul(a.Float32(), b.Float32())
	if out32.dtype != Float32 {
		t.Errorf("F32xF32 batch: expected Float32, got %v", out32.dtype)
	}
}
