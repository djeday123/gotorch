package tensor

import (
	"math"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────────────
// DType string
// ─────────────────────────────────────────────────────────────────────────────

func TestDTypeString(t *testing.T) {
	cases := []struct{ d DType; want string }{
		{Float64, "float64"},
		{Float32, "float32"},
		{Float16, "float16"},
		{BFloat16, "bfloat16"},
		{Int8, "int8"},
		{Int32, "int32"},
	}
	for _, c := range cases {
		if got := DTypeString(c.d); got != c.want {
			t.Errorf("DTypeString(%d) = %q, want %q", c.d, got, c.want)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Float16
// ─────────────────────────────────────────────────────────────────────────────

func TestHalfRoundtrip(t *testing.T) {
	// Values that fit exactly in fp16 (multiples of power-of-2).
	vals := []float64{0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 4.0, 0.25}
	src := New(vals, []int{len(vals)})
	h := src.Half()

	if h.DType() != Float16 {
		t.Fatalf("expected Float16, got %s", DTypeString(h.DType()))
	}
	for i, v := range h.Data() {
		if math.Abs(v-vals[i]) > 1e-3 {
			t.Errorf("Half[%d]: got %.6f, want %.6f", i, v, vals[i])
		}
	}
}

func TestHalfClamp(t *testing.T) {
	// Values beyond fp16 max (65504) must be clamped.
	src := New([]float64{1e10, -1e10}, []int{2})
	h := src.Half()
	d := h.Data()
	const maxF16 = 65504.0
	if d[0] != maxF16 {
		t.Errorf("expected +65504, got %g", d[0])
	}
	if d[1] != -maxF16 {
		t.Errorf("expected -65504, got %g", d[1])
	}
}

func TestHalfPrecisionLoss(t *testing.T) {
	// π cannot be represented exactly in fp16; error must be small but non-zero.
	src := New([]float64{math.Pi}, []int{1})
	h := src.Half()
	diff := math.Abs(h.Data()[0] - math.Pi)
	if diff == 0 {
		t.Error("expected precision loss in fp16, got exact representation")
	}
	if diff > 0.01 { // error must be < 1% of π
		t.Errorf("fp16 error too large: %g", diff)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// BFloat16
// ─────────────────────────────────────────────────────────────────────────────

func TestBF16Roundtrip(t *testing.T) {
	// 1.0 and -1.0 should be exact in bf16.
	src := New([]float64{0.0, 1.0, -1.0, 2.0, 0.5}, []int{5})
	bf := src.BF16()

	if bf.DType() != BFloat16 {
		t.Fatalf("expected BFloat16, got %s", DTypeString(bf.DType()))
	}
	for i, v := range bf.Data() {
		if math.Abs(v-src.Data()[i]) > 0.05 {
			t.Errorf("BF16[%d]: got %.6f, want %.6f", i, v, src.Data()[i])
		}
	}
}

func TestBF16PrecisionLoss(t *testing.T) {
	// 0.1 cannot be exact in bf16 (7 mantissa bits).
	src := New([]float64{0.1}, []int{1})
	bf := src.BF16()
	diff := math.Abs(bf.Data()[0] - 0.1)
	if diff == 0 {
		t.Error("expected precision loss in bf16, got exact representation")
	}
	if diff > 0.01 {
		t.Errorf("bf16 error too large: %g", diff)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Int8
// ─────────────────────────────────────────────────────────────────────────────

func TestInt8Clamp(t *testing.T) {
	src := New([]float64{0, 1, -1, 127, 128, -128, -129, 255}, []int{8})
	i8 := src.Int8T()

	if i8.DType() != Int8 {
		t.Fatalf("expected Int8, got %s", DTypeString(i8.DType()))
	}
	expected := []float64{0, 1, -1, 127, 127, -128, -128, 127}
	for i, got := range i8.Data() {
		if got != expected[i] {
			t.Errorf("Int8[%d]: got %.0f, want %.0f", i, got, expected[i])
		}
	}
}

func TestInt8Roundtrip(t *testing.T) {
	src := New([]float64{-100, -50, 0, 50, 100}, []int{5})
	i8 := src.Int8T()
	d := i8.Data()
	for i, v := range src.Data() {
		if math.Round(v) != d[i] {
			t.Errorf("Int8[%d]: got %.0f, want %.0f", i, d[i], math.Round(v))
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// To() method
// ─────────────────────────────────────────────────────────────────────────────

func TestTo(t *testing.T) {
	src := New([]float64{1.0, 2.0, 3.0}, []int{3})

	if src.To(Float64).DType() != Float64 {
		t.Error("To(Float64) dtype wrong")
	}
	if src.To(Float32).DType() != Float32 {
		t.Error("To(Float32) dtype wrong")
	}
	if src.To(Float16).DType() != Float16 {
		t.Error("To(Float16) dtype wrong")
	}
	if src.To(BFloat16).DType() != BFloat16 {
		t.Error("To(BFloat16) dtype wrong")
	}
	if src.To(Int8).DType() != Int8 {
		t.Error("To(Int8) dtype wrong")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantize8 / Dequantize8
// ─────────────────────────────────────────────────────────────────────────────

func TestQuantize8(t *testing.T) {
	data := []float64{-1.0, -0.5, 0.0, 0.5, 1.0}
	src := New(data, []int{5})

	q, qp := Quantize8(src)
	if q.DType() != Int8 {
		t.Fatalf("Quantize8 dtype: got %s, want int8", DTypeString(q.DType()))
	}
	// scale should be 1.0/127 ≈ 0.00787
	expectedScale := 1.0 / 127.0
	if math.Abs(qp.Scale-expectedScale) > 1e-6 {
		t.Errorf("scale: got %g, want %g", qp.Scale, expectedScale)
	}
	// -1.0 → -127, 0 → 0, 1.0 → 127
	qd := q.Data()
	if qd[0] != -127 || qd[2] != 0 || qd[4] != 127 {
		t.Errorf("quantized values: %v", qd)
	}
}

func TestDequantize8Roundtrip(t *testing.T) {
	data := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	src := New(data, []int{5})

	q, qp := Quantize8(src)
	deq := Dequantize8(q, qp)

	for i, orig := range data {
		got := deq.Data()[i]
		if math.Abs(got-orig) > 0.05 { // <2.5% error for 8-bit
			t.Errorf("dequant[%d]: got %g, want %g (err=%g)", i, got, orig, math.Abs(got-orig))
		}
	}
}

func TestQuantize8Zero(t *testing.T) {
	// All-zero tensor should not panic.
	src := Zeros(5)
	q, qp := Quantize8(src)
	if qp.Scale != 1.0 {
		t.Errorf("zero tensor scale: %g", qp.Scale)
	}
	for _, v := range q.Data() {
		if v != 0 {
			t.Errorf("zero tensor quantized has non-zero: %g", v)
		}
	}
}
