package nn

import (
	"math"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/tensor"
)

// Conv1d implements a 1D convolution layer.
//
// Input:  [N, inC, L]
// Output: [N, outC, oL]  where oL = (L + 2*P - K) / S + 1
//
// Weight: [outC, inC, K]
// Bias:   [outC]
type Conv1d struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Weight      *autograd.Variable
	Bias        *autograd.Variable
	useBias     bool
}

// NewConv1d creates a Conv1d with He initialisation.
func NewConv1d(inC, outC, kernelSize, stride, padding int, bias bool) *Conv1d {
	fan := float64(inC * kernelSize)
	std := math.Sqrt(2.0 / fan)
	wData := make([]float64, outC*inC*kernelSize)
	for i := range wData {
		wData[i] = rand.NormFloat64() * std
	}
	w := autograd.NewVar(tensor.New(wData, []int{outC, inC, kernelSize}), true)

	var b *autograd.Variable
	if bias {
		b = autograd.NewVar(tensor.Zeros(outC), true)
	}

	return &Conv1d{
		InChannels:  inC,
		OutChannels: outC,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Weight:      w,
		Bias:        b,
		useBias:     bias,
	}
}

func (c *Conv1d) Forward(x *autograd.Variable) *autograd.Variable {
	return conv1dForward(x, c.Weight, c.Bias, c.Stride, c.Padding)
}

func (c *Conv1d) Parameters() []*autograd.Variable {
	if c.useBias {
		return []*autograd.Variable{c.Weight, c.Bias}
	}
	return []*autograd.Variable{c.Weight}
}

func (c *Conv1d) ZeroGrad() {
	c.Weight.ZeroGrad()
	if c.useBias {
		c.Bias.ZeroGrad()
	}
}

// ---------------------------------------------------------------------------
// im2col-style 1D convolution with autograd
// ---------------------------------------------------------------------------

// im2col1d: [N, inC, L] → [N*oL, inC*K]
func im2col1d(x *tensor.Tensor, K, stride, padding int) (*tensor.Tensor, int) {
	shape := x.Shape()
	N, inC, L := shape[0], shape[1], shape[2]
	oL := (L+2*padding-K)/stride + 1
	cols := tensor.Zeros(N*oL, inC*K)
	xData := x.ContiguousCopy().Data()
	colData := cols.Data()

	for n := 0; n < N; n++ {
		for ol := 0; ol < oL; ol++ {
			for c := 0; c < inC; c++ {
				for k := 0; k < K; k++ {
					lIn := ol*stride + k - padding
					var val float64
					if lIn >= 0 && lIn < L {
						val = xData[n*inC*L+c*L+lIn]
					}
					colIdx := (n*oL+ol)*(inC*K) + c*K + k
					colData[colIdx] = val
				}
			}
		}
	}
	return cols, oL
}

// col2im1d: [N*oL, inC*K] → [N, inC, L]
func col2im1d(cols *tensor.Tensor, N, inC, L, K, stride, padding int) *tensor.Tensor {
	oL := (cols.Shape()[0]) / N
	out := tensor.Zeros(N, inC, L)
	colData := cols.Data()
	outData := out.Data()

	for n := 0; n < N; n++ {
		for ol := 0; ol < oL; ol++ {
			for c := 0; c < inC; c++ {
				for k := 0; k < K; k++ {
					lIn := ol*stride + k - padding
					if lIn >= 0 && lIn < L {
						colIdx := (n*oL+ol)*(inC*K) + c*K + k
						outData[n*inC*L+c*L+lIn] += colData[colIdx]
					}
				}
			}
		}
	}
	return out
}

type conv1dBackward struct {
	xCol   *tensor.Tensor // [N*oL, inC*K]
	wFlat  *tensor.Tensor // [outC, inC*K]
	xShape []int          // [N, inC, L]
	K, stride, padding int
	oL     int
}

func (f *conv1dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	N, inC, L := f.xShape[0], f.xShape[1], f.xShape[2]
	outC := f.wFlat.Shape()[0]

	// grad: [N, outC, oL] → reshape to [N*oL, outC]
	gradFlat := tensor.New(grad.ContiguousCopy().Data(), []int{N * f.oL, outC})

	// dW: [outC, inC*K] = gradFlat^T @ xCol → reshape to [outC, inC, K]
	dWFlat := tensor.MatMul(gradFlat.T(), f.xCol)
	outC2 := f.wFlat.Shape()[0]
	inC2 := f.wFlat.Shape()[1] / f.K
	dW := tensor.New(dWFlat.ContiguousCopy().Data(), []int{outC2, inC2, f.K})

	// dXcol: [N*oL, inC*K] = gradFlat @ W
	dXcol := tensor.MatMul(gradFlat, f.wFlat)

	// dX: [N, inC, L]
	dX := col2im1d(dXcol, N, inC, L, f.K, f.stride, f.padding)

	return []*tensor.Tensor{dX, dW}
}

func conv1dForward(x, w *autograd.Variable, b *autograd.Variable, stride, padding int) *autograd.Variable {
	wShape := w.Data.Shape()
	outC, inC, K := wShape[0], wShape[1], wShape[2]
	xShape := x.Data.Shape()
	N := xShape[0]
	_ = inC

	// im2col
	xCol, oL := im2col1d(x.Data, K, stride, padding)

	// wFlat: [outC, inC*K]
	wFlat := tensor.New(w.Data.ContiguousCopy().Data(), []int{outC, inC * K})

	// out: [N*oL, outC] = xCol @ wFlat^T
	outFlat := tensor.MatMul(xCol, wFlat.T())

	// reshape to [N, outC, oL]
	outData := outFlat.ContiguousCopy().Data()
	outReshaped := make([]float64, N*outC*oL)
	for n := 0; n < N; n++ {
		for ol := 0; ol < oL; ol++ {
			for oc := 0; oc < outC; oc++ {
				// outFlat[n*oL+ol, oc] → out[n, oc, ol]
				outReshaped[n*outC*oL+oc*oL+ol] = outData[(n*oL+ol)*outC+oc]
			}
		}
	}

	result := tensor.New(outReshaped, []int{N, outC, oL})

	children := []*autograd.Variable{x, w}
	out := autograd.NewResult(result, &conv1dBackward{xCol, wFlat, xShape, K, stride, padding, oL}, children...)

	// Add bias
	if b != nil {
		out = addConv1dBias(out, b, N, outC, oL)
	}
	return out
}

type addConv1dBiasBackward struct {
	outC int
	N, oL int
}

func (f *addConv1dBiasBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	// grad: [N, outC, oL]
	// dBias[c] = sum over N and oL
	dBias := tensor.Zeros(f.outC)
	gd := grad.ContiguousCopy().Data()
	bd := dBias.Data()
	for n := 0; n < f.N; n++ {
		for c := 0; c < f.outC; c++ {
			for ol := 0; ol < f.oL; ol++ {
				bd[c] += gd[n*f.outC*f.oL+c*f.oL+ol]
			}
		}
	}
	return []*tensor.Tensor{grad, dBias}
}

func addConv1dBias(out *autograd.Variable, bias *autograd.Variable, N, outC, oL int) *autograd.Variable {
	// broadcast bias [outC] to [N, outC, oL]
	outData := out.Data.ContiguousCopy().Data()
	bData := bias.Data.Data()
	for n := 0; n < N; n++ {
		for c := 0; c < outC; c++ {
			for ol := 0; ol < oL; ol++ {
				outData[n*outC*oL+c*oL+ol] += bData[c]
			}
		}
	}
	result := tensor.New(outData, []int{N, outC, oL})
	return autograd.NewResult(result, &addConv1dBiasBackward{outC, N, oL}, out, bias)
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────────
// Reduces spatial dims to (outH, outW) via average pooling.
// Input:  [N, C, H, W]
// Output: [N, C, outH, outW]

type AdaptiveAvgPool2d struct {
	OutH, OutW int
}

func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{OutH: outH, OutW: outW}
}
func (a *AdaptiveAvgPool2d) Parameters() []*autograd.Variable { return nil }
func (a *AdaptiveAvgPool2d) ZeroGrad()                         {}
func (a *AdaptiveAvgPool2d) Forward(x *autograd.Variable) *autograd.Variable {
	return adaptiveAvgPool2dForward(x, a.OutH, a.OutW)
}

type adaptiveAvgPool2dBackward struct {
	xShape         []int
	outH, outW     int
	startH, endH   []int
	startW, endW   []int
}

func (f *adaptiveAvgPool2dBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
	N, C, H, W := f.xShape[0], f.xShape[1], f.xShape[2], f.xShape[3]
	dX := tensor.Zeros(N, C, H, W)
	dxData := dX.Data()
	gData := grad.ContiguousCopy().Data()

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < f.outH; oh++ {
				for ow := 0; ow < f.outW; ow++ {
					sh, eh := f.startH[oh], f.endH[oh]
					sw, ew := f.startW[ow], f.endW[ow]
					area := float64((eh - sh) * (ew - sw))
					gv := gData[n*C*f.outH*f.outW+c*f.outH*f.outW+oh*f.outW+ow] / area
					for ih := sh; ih < eh; ih++ {
						for iw := sw; iw < ew; iw++ {
							dxData[n*C*H*W+c*H*W+ih*W+iw] += gv
						}
					}
				}
			}
		}
	}
	return []*tensor.Tensor{dX}
}

func adaptiveAvgPool2dForward(x *autograd.Variable, outH, outW int) *autograd.Variable {
	shape := x.Data.Shape()
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]

	// Compute pooling windows
	startH := make([]int, outH)
	endH := make([]int, outH)
	for i := 0; i < outH; i++ {
		startH[i] = i * H / outH
		endH[i] = (i+1)*H/outH
	}
	startW := make([]int, outW)
	endW := make([]int, outW)
	for i := 0; i < outW; i++ {
		startW[i] = i * W / outW
		endW[i] = (i+1)*W/outW
	}

	xData := x.Data.ContiguousCopy().Data()
	outData := make([]float64, N*C*outH*outW)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sh, eh := startH[oh], endH[oh]
					sw, ew := startW[ow], endW[ow]
					area := float64((eh - sh) * (ew - sw))
					sum := 0.0
					for ih := sh; ih < eh; ih++ {
						for iw := sw; iw < ew; iw++ {
							sum += xData[n*C*H*W+c*H*W+ih*W+iw]
						}
					}
					outData[n*C*outH*outW+c*outH*outW+oh*outW+ow] = sum / area
				}
			}
		}
	}

	result := tensor.New(outData, []int{N, C, outH, outW})
	return autograd.NewResult(result, &adaptiveAvgPool2dBackward{
		x.Data.Shape(), outH, outW, startH, endH, startW, endW,
	}, x)
}
