package nn

import (
	"fmt"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// QLinear is a quantized Linear layer.
//
// Weights are stored as Int8 (8-bit signed integers) with a per-tensor symmetric
// scale factor. Inputs are passed through as float64 for simplicity (weight-only
// quantization). This reduces model size ~4× compared to float32 and is suitable
// for inference; training is done in float64 before quantization.
//
//	qlin := nn.QuantizeLinear(lin)
//	out  := qlin.Forward(x)  // same interface as nn.Linear
type QLinear struct {
	qWeight   *tensor.Tensor   // Int8 quantized weights [outFeatures, inFeatures]
	qParams   tensor.QuantParams
	bias      *autograd.Variable // float64 bias (nil if NoBias)
	inFeat    int
	outFeat   int
	hasBias   bool
}

// QuantizeLinear creates a QLinear from a trained *Linear layer.
// The weights are quantised once and the original layer is no longer needed.
func QuantizeLinear(l *Linear) *QLinear {
	qw, qp := tensor.Quantize8(l.Weight.Data)
	ql := &QLinear{
		qWeight: qw,
		qParams: qp,
		inFeat:  l.Weight.Data.Shape()[1],
		outFeat: l.Weight.Data.Shape()[0],
		hasBias: l.Bias != nil,
	}
	if l.Bias != nil {
		// Copy bias as float64 — bias is kept in full precision.
		bd := make([]float64, l.Bias.Data.Size())
		copy(bd, l.Bias.Data.Data())
		ql.bias = autograd.NewVar(tensor.New(bd, l.Bias.Data.Shape()), false)
	}
	return ql
}

// Forward performs y = x @ W_dequant^T + b.
// Weight is dequantised on-the-fly during inference.
func (q *QLinear) Forward(x *autograd.Variable) *autograd.Variable {
	// Dequantise weights back to float64.
	wFloat := tensor.Dequantize8(q.qWeight, q.qParams)

	batchSize := x.Data.Shape()[0]

	// Manual matmul: [batch, in] @ [out, in]^T → [batch, out]
	xd := x.Data.Data()
	wd := wFloat.Data()
	outData := make([]float64, batchSize*q.outFeat)

	for b := 0; b < batchSize; b++ {
		for o := 0; o < q.outFeat; o++ {
			sum := 0.0
			for i := 0; i < q.inFeat; i++ {
				sum += xd[b*q.inFeat+i] * wd[o*q.inFeat+i]
			}
			outData[b*q.outFeat+o] = sum
		}
	}

	// Add bias.
	if q.hasBias {
		bd := q.bias.Data.Data()
		for b := 0; b < batchSize; b++ {
			for o := 0; o < q.outFeat; o++ {
				outData[b*q.outFeat+o] += bd[o]
			}
		}
	}

	outT := tensor.New(outData, []int{batchSize, q.outFeat})
	// QLinear is inference-only; no autograd graph is built.
	return autograd.NewVar(outT, false)
}

// Parameters returns an empty slice — QLinear is inference-only; weights are int8.
func (q *QLinear) Parameters() []*autograd.Variable { return nil }

// ZeroGrad is a no-op for QLinear.
func (q *QLinear) ZeroGrad() {}

// String describes the quantized layer.
func (q *QLinear) String() string {
	return fmt.Sprintf("QLinear(%d → %d, int8, scale=%.6f)", q.inFeat, q.outFeat, q.qParams.Scale)
}

// CompressionRatio returns the approximate size ratio vs float64.
// Int8 uses 1 byte per element vs 8 bytes for float64 → ratio ≈ 8×.
func (q *QLinear) CompressionRatio() float64 { return 8.0 }

// ─────────────────────────────────────────────────────────────────────────────
// QuantizeModel: replace all Linear layers in a Sequential with QLinear
// ─────────────────────────────────────────────────────────────────────────────

// QuantizedSequential holds a mix of regular and quantized layers.
type QuantizedSequential struct {
	layers []Module
}

// QuantizeModel converts all *Linear layers in a *Sequential to *QLinear.
// Non-Linear layers are kept as-is. Returns a *QuantizedSequential.
func QuantizeModel(s *Sequential) *QuantizedSequential {
	qs := &QuantizedSequential{}
	for _, layer := range s.layers {
		if lin, ok := layer.(*Linear); ok {
			qs.layers = append(qs.layers, QuantizeLinear(lin))
		} else {
			qs.layers = append(qs.layers, layer)
		}
	}
	return qs
}

// Forward runs the quantized sequential model.
func (qs *QuantizedSequential) Forward(x *autograd.Variable) *autograd.Variable {
	h := x
	for _, layer := range qs.layers {
		h = layer.Forward(h)
	}
	return h
}

// Parameters returns parameters of non-quantized layers only.
func (qs *QuantizedSequential) Parameters() []*autograd.Variable {
	var p []*autograd.Variable
	for _, l := range qs.layers {
		p = append(p, l.Parameters()...)
	}
	return p
}

// ZeroGrad zeroes gradients for all layers.
func (qs *QuantizedSequential) ZeroGrad() {
	for _, l := range qs.layers {
		l.ZeroGrad()
	}
}

// Stats prints quantization statistics for each layer.
func (qs *QuantizedSequential) Stats() string {
	s := "QuantizedSequential:\n"
	for i, l := range qs.layers {
		if ql, ok := l.(*QLinear); ok {
			s += fmt.Sprintf("  [%d] %s (%.1f× compression)\n", i, ql.String(), ql.CompressionRatio())
		} else {
			s += fmt.Sprintf("  [%d] %T\n", i, l)
		}
	}
	return s
}
