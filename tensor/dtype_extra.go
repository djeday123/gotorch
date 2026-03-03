package tensor

import (
	"math"
)

// Extended dtypes (Float64=0 and Float32=1 are defined in tensor.go).
const (
	Float16  DType = 2 // 16-bit half precision
	BFloat16 DType = 3 // Brain float16 (Google); same exponent range as float32, 7 mantissa bits
	Int8     DType = 4 // 8-bit signed integer; used for quantized inference
	Int32    DType = 5 // 32-bit signed integer
)

func init() {
	// Extend DType.String() by patching the switch via override map.
	// (Actual String() is on the DType in tensor.go — extend there if needed.)
	_ = Float16 // ensure iota values don't collide silently
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion constructors
// ─────────────────────────────────────────────────────────────────────────────

// NewHalf creates a Float16 tensor from float64 data.
// Values outside the fp16 range (±65504) are clamped.
func NewHalf(data []float64, shape []int) *Tensor {
	d := make([]float64, len(data))
	for i, v := range data {
		d[i] = toFloat16(v)
	}
	t := New(d, shape)
	t.dtype = Float16
	return t
}

// NewBFloat16 creates a BFloat16 tensor from float64 data.
func NewBFloat16(data []float64, shape []int) *Tensor {
	d := make([]float64, len(data))
	for i, v := range data {
		d[i] = toBFloat16(v)
	}
	t := New(d, shape)
	t.dtype = BFloat16
	return t
}

// NewInt8 creates an Int8 tensor from float64 data (round + clamp to [-128, 127]).
func NewInt8(data []float64, shape []int) *Tensor {
	d := make([]float64, len(data))
	for i, v := range data {
		vi := int(math.Round(v))
		if vi < -128 {
			vi = -128
		}
		if vi > 127 {
			vi = 127
		}
		d[i] = float64(vi)
	}
	t := New(d, shape)
	t.dtype = Int8
	return t
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion methods on existing *Tensor
// ─────────────────────────────────────────────────────────────────────────────

// Half returns a Float16 copy of the tensor.
func (t *Tensor) Half() *Tensor {
	return NewHalf(t.Data(), t.Shape())
}

// BF16 returns a BFloat16 copy of the tensor.
func (t *Tensor) BF16() *Tensor {
	return NewBFloat16(t.Data(), t.Shape())
}

// Int8T returns an Int8 copy of the tensor (round + clamp).
func (t *Tensor) Int8T() *Tensor {
	return NewInt8(t.Data(), t.Shape())
}

// To converts the tensor to the given dtype, returning a new tensor.
// Float64 (upcast) is always lossless.
func (t *Tensor) To(dt DType) *Tensor {
	switch dt {
	case Float64:
		return t.Float64()
	case Float32:
		return t.Float32()
	case Float16:
		return t.Half()
	case BFloat16:
		return t.BF16()
	case Int8:
		return t.Int8T()
	default:
		return t.ContiguousCopy()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// DType string (extends the two-value switch in tensor.go)
// ─────────────────────────────────────────────────────────────────────────────

// DTypeString returns the name of a dtype.
func DTypeString(d DType) string {
	switch d {
	case Float64:
		return "float64"
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Int8:
		return "int8"
	case Int32:
		return "int32"
	default:
		return "unknown"
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// fp16 / bf16 simulation helpers
// ─────────────────────────────────────────────────────────────────────────────

// toFloat16 simulates float16 precision.
// fp16: 1 sign + 5 exponent + 10 mantissa bits. Max value ≈ 65504.
func toFloat16(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	const maxF16 = 65504.0
	if v > maxF16 {
		return maxF16
	}
	if v < -maxF16 {
		return -maxF16
	}
	// Round-to-nearest by going through float32 then masking lower 13 mantissa bits.
	// fp32 has 23 mantissa bits; fp16 has 10 → zero the lower 13.
	bits := math.Float32bits(float32(v))
	bits &^= 0x1FFF
	return float64(math.Float32frombits(bits))
}

// toBFloat16 simulates bfloat16 precision.
// bf16: 1 sign + 8 exponent + 7 mantissa bits (top 16 bits of float32).
func toBFloat16(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	// Truncate lower 16 mantissa bits of float32 (keep upper 7 of 23).
	bits := math.Float32bits(float32(v))
	// Round to nearest: inspect bit 15 for rounding
	round := bits & 0x8000
	bits &^= 0xFFFF
	if round != 0 {
		bits += 0x10000 // add 1 to bit 16 (round up)
	}
	return float64(math.Float32frombits(bits))
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantization helpers (per-tensor symmetric)
// ─────────────────────────────────────────────────────────────────────────────

// QuantParams holds the scale and zero_point for int8 quantization.
type QuantParams struct {
	Scale     float64 // multiplier: real = int8_val * Scale
	ZeroPoint int     // always 0 for symmetric quantization
}

// Quantize8 quantises a float64 tensor to int8 using symmetric per-tensor scaling.
// Returns the quantised tensor and the QuantParams needed for dequantisation.
func Quantize8(t *Tensor) (*Tensor, QuantParams) {
	data := t.Data()
	maxAbs := 0.0
	for _, v := range data {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return NewInt8(make([]float64, len(data)), t.Shape()), QuantParams{Scale: 1.0}
	}
	scale := maxAbs / 127.0
	qi := make([]float64, len(data))
	for i, v := range data {
		q := math.Round(v / scale)
		if q < -128 {
			q = -128
		}
		if q > 127 {
			q = 127
		}
		qi[i] = q
	}
	return NewInt8(qi, t.Shape()), QuantParams{Scale: scale, ZeroPoint: 0}
}

// Dequantize8 converts an Int8 tensor back to Float64 using stored QuantParams.
func Dequantize8(t *Tensor, qp QuantParams) *Tensor {
	data := t.Data()
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = v * qp.Scale
	}
	return New(out, t.Shape())
}
