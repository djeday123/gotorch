package nn

import (
	"math"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// ConvTranspose2d implements a 2D transposed convolution (fractionally strided convolution).
//
// Input:  [N, inC,  H,  W]
// Output: [N, outC, oH, oW]  where oH = (H-1)*stride - 2*padding + kernelSize
//
// Weight: [inC, outC, kH, kW]
// Bias:   [outC]
//
// Implementation: the equivalence
//
//   y[n, oc, oh, ow] = Σ_{ic,kh,kw}  x[n, ic, h, w] · W[ic, oc, kh, kw]
//                       where oh = h·s + kh − p, ow = w·s + kw − p
//
// is rearranged as a per-batch matmul plus a col2im scatter:
//
//   cols[oc·kH·kW + kh·kW + kw, h·W + w]  =  Σ_ic  W[ic, oc, kh, kw] · x[n, ic, h, w]
//                                          = (Wᵀ ·  Xₙ)
//
// where W is flattened to [inC, outC·kH·kW] and Xₙ to [inC, H·W]. The
// resulting [outC·kH·kW, H·W] matrix is then scattered into the output by
// col2im. This replaces the quintuple-nested-loop O(N·inC·H·W·outC·kH·kW)
// scalar accumulation with one tiled matmul per batch (same op count, but
// 10-50× faster in practice from cache locality).
type ConvTranspose2d struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Weight      *autograd.Variable
	Bias        *autograd.Variable
	useBias     bool
}

// NewConvTranspose2d creates a ConvTranspose2d with He initialisation.
func NewConvTranspose2d(inCh, outCh, kernelSize, stride, padding int) *ConvTranspose2d {
	fan := float64(inCh * kernelSize * kernelSize)
	std := math.Sqrt(2.0 / fan)
	wSize := inCh * outCh * kernelSize * kernelSize
	wData := make([]float64, wSize)
	for i := range wData {
		wData[i] = rand.NormFloat64() * std
	}
	w := autograd.NewVar(tensor.New(wData, []int{inCh, outCh, kernelSize, kernelSize}), true)
	b := autograd.NewVar(tensor.Zeros(outCh), true)

	return &ConvTranspose2d{
		InChannels:  inCh,
		OutChannels: outCh,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Weight:      w,
		Bias:        b,
		useBias:     true,
	}
}

// Parameters returns [weight, bias].
func (ct *ConvTranspose2d) Parameters() []*autograd.Variable {
	if ct.useBias {
		return []*autograd.Variable{ct.Weight, ct.Bias}
	}
	return []*autograd.Variable{ct.Weight}
}

// ZeroGrad zeros gradients of all parameters.
func (ct *ConvTranspose2d) ZeroGrad() {
	ct.Weight.ZeroGrad()
	if ct.useBias {
		ct.Bias.ZeroGrad()
	}
}

// Forward computes the transposed convolution.
func (ct *ConvTranspose2d) Forward(x *autograd.Variable) *autograd.Variable {
	return convTranspose2dForward(x, ct.Weight, ct.Bias, ct.useBias, ct.Stride, ct.Padding)
}

// ── ConvTranspose2d forward + backward ──────────────────────────────────────

type convTranspose2dBackward struct {
	xData  *tensor.Tensor // [N, inC, H, W]
	wData  *tensor.Tensor // [inC, outC, kH, kW]
	stride int
	pad    int
}

func (f *convTranspose2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	xShape := f.xData.Shape() // [N, inC, H, W]
	wShape := f.wData.Shape() // [inC, outC, kH, kW]
	N, inC, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	_, outC, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	oH := (H-1)*f.stride - 2*f.pad + kH
	oW := (W-1)*f.stride - 2*f.pad + kW

	// Pre-flatten weight as [inC, outC·kH·kW] for matmul.
	wFlat := f.wData.ContiguousCopy().Reshape(inC, outC*kH*kW)

	gradData := grad.Data() // [N, outC, oH, oW] flat
	xData := f.xData.ContiguousCopy().Data()

	dXData := make([]float64, N*inC*H*W)
	dWFlatData := make([]float64, inC*outC*kH*kW)
	hw := H * W

	for n := 0; n < N; n++ {
		// Build cols_grad_n = im2col of grad_n, shape [outC·kH·kW, H·W]
		// where cols[(oc·kH+kh)·kW+kw, h·W+w] = grad_n[oc, h·s+kh-p, w·s+kw-p].
		colsGrad := im2colForTransposeBackward(gradData, n, outC, kH, kW, H, W, oH, oW, f.stride, f.pad)
		colsGradT := tensor.New(colsGrad, []int{outC * kH * kW, hw})

		// dX_n = W_flat · cols_grad_n  → [inC, H·W]
		dXn := tensor.MatMul(wFlat, colsGradT)
		copy(dXData[n*inC*hw:(n+1)*inC*hw], dXn.Data())

		// dW += x_n · cols_grad_nᵀ → [inC, outC·kH·kW]
		xnData := xData[n*inC*hw : (n+1)*inC*hw]
		xn := tensor.New(append([]float64(nil), xnData...), []int{inC, hw})
		dWPartial := tensor.MatMul(xn, colsGradT.T())
		dPartialData := dWPartial.Data()
		for i := range dWFlatData {
			dWFlatData[i] += dPartialData[i]
		}
	}

	dX := tensor.New(dXData, []int{N, inC, H, W})
	dW := tensor.New(dWFlatData, []int{inC, outC, kH, kW})
	return []*tensor.Tensor{dX, dW}
}

func convTranspose2dForward(x, w, b *autograd.Variable, useBias bool, stride, pad int) *autograd.Variable {
	xShape := x.Data.Shape() // [N, inC, H, W]
	wShape := w.Data.Shape() // [inC, outC, kH, kW]
	N, inC, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	_, outC, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]

	oH := (H-1)*stride - 2*pad + kH
	oW := (W-1)*stride - 2*pad + kW

	outData := make([]float64, N*outC*oH*oW)
	outShape := []int{N, outC, oH, oW}

	// Flatten weight: [inC, outC·kH·kW]. Its transpose [outC·kH·kW, inC] is
	// used as the left operand against x_n shaped [inC, H·W].
	wFlat := w.Data.ContiguousCopy().Reshape(inC, outC*kH*kW)
	wFlatT := wFlat.T() // [outC·kH·kW, inC]

	xData := x.Data.ContiguousCopy().Data()
	hw := H * W

	for n := 0; n < N; n++ {
		// x_n as a [inC, H·W] matrix (own copy so MatMul can ContiguousCopy).
		xnData := append([]float64(nil), xData[n*inC*hw:(n+1)*inC*hw]...)
		xn := tensor.New(xnData, []int{inC, hw})

		// cols = Wᵀ · x_n  → [outC·kH·kW, H·W]
		cols := tensor.MatMul(wFlatT, xn)
		colsData := cols.Data()

		// col2im: scatter cols into output[n, :, :, :].
		for oc := 0; oc < outC; oc++ {
			for kh := 0; kh < kH; kh++ {
				for kw := 0; kw < kW; kw++ {
					rowIdx := (oc*kH+kh)*kW + kw
					rowBase := rowIdx * hw
					for h := 0; h < H; h++ {
						oh := h*stride + kh - pad
						if oh < 0 || oh >= oH {
							continue
						}
						for w_ := 0; w_ < W; w_++ {
							ow := w_*stride + kw - pad
							if ow < 0 || ow >= oW {
								continue
							}
							outData[((n*outC+oc)*oH+oh)*oW+ow] += colsData[rowBase+h*W+w_]
						}
					}
				}
			}
		}
	}

	// Add bias.
	if useBias && b != nil {
		bData := b.Data.Data()
		for n := 0; n < N; n++ {
			for oc := 0; oc < outC; oc++ {
				bias := bData[oc]
				for oh := 0; oh < oH; oh++ {
					for ow := 0; ow < oW; ow++ {
						outData[((n*outC+oc)*oH+oh)*oW+ow] += bias
					}
				}
			}
		}
	}

	outT := tensor.New(outData, outShape)
	return autograd.NewResult(outT, &convTranspose2dBackward{x.Data, w.Data, stride, pad}, x, w)
}

// im2colForTransposeBackward extracts grad[n, oc, h·s+kh-p, w·s+kw-p] into a
// 2D matrix of shape [outC·kH·kW, H·W] for use in ConvTranspose2d backward.
func im2colForTransposeBackward(grad []float64, n, outC, kH, kW, H, W, oH, oW, stride, pad int) []float64 {
	hw := H * W
	cols := make([]float64, outC*kH*kW*hw)
	nOff := n * outC * oH * oW
	for oc := 0; oc < outC; oc++ {
		ocOff := nOff + oc*oH*oW
		for kh := 0; kh < kH; kh++ {
			for kw := 0; kw < kW; kw++ {
				rowIdx := (oc*kH+kh)*kW + kw
				rowBase := rowIdx * hw
				for h := 0; h < H; h++ {
					oh := h*stride + kh - pad
					if oh < 0 || oh >= oH {
						continue
					}
					ohBase := ocOff + oh*oW
					for w := 0; w < W; w++ {
						ow := w*stride + kw - pad
						if ow < 0 || ow >= oW {
							continue
						}
						cols[rowBase+h*W+w] = grad[ohBase+ow]
					}
				}
			}
		}
	}
	return cols
}

// flatIdx4 computes flat index for a 4D tensor [d0, d1, d2, d3].
func flatIdx4(i0, i1, i2, i3 int, shape []int) int {
	return i0*shape[1]*shape[2]*shape[3] + i1*shape[2]*shape[3] + i2*shape[3] + i3
}
