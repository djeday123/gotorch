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
	// grad shape: [N, outC, oH, oW]
	xShape := f.xData.Shape() // [N, inC, H, W]
	wShape := f.wData.Shape() // [inC, outC, kH, kW]
	N, inC, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	_, outC, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	oH := (H-1)*f.stride - 2*f.pad + kH
	oW := (W-1)*f.stride - 2*f.pad + kW

	// dL/dInput: for each (n, ic, h, w): sum over oc, kh, kw of grad[n,oc,h*s+kh-p,w*s+kw-p]*weight[ic,oc,kh,kw]
	dX := tensor.Zeros(N, inC, H, W)
	dXd := dX.Data()
	xdShape := []int{N, inC, H, W}
	grad4d := grad // [N, outC, oH, oW]

	for n := 0; n < N; n++ {
		for ic := 0; ic < inC; ic++ {
			for h := 0; h < H; h++ {
				for w := 0; w < W; w++ {
					sum := 0.0
					for oc := 0; oc < outC; oc++ {
						for kh := 0; kh < kH; kh++ {
							for kw := 0; kw < kW; kw++ {
								oh := h*f.stride + kh - f.pad
								ow := w*f.stride + kw - f.pad
								if oh >= 0 && oh < oH && ow >= 0 && ow < oW {
									gv := grad4d.At(n, oc, oh, ow)
									wv := f.wData.At(ic, oc, kh, kw)
									sum += gv * wv
								}
							}
						}
					}
					dXd[flatIdx4(n, ic, h, w, xdShape)] = sum
				}
			}
		}
	}

	// dL/dWeight: [inC, outC, kH, kW]
	dW := tensor.Zeros(inC, outC, kH, kW)
	dWd := dW.Data()
	wdShape := []int{inC, outC, kH, kW}

	for ic := 0; ic < inC; ic++ {
		for oc := 0; oc < outC; oc++ {
			for kh := 0; kh < kH; kh++ {
				for kw := 0; kw < kW; kw++ {
					sum := 0.0
					for n := 0; n < N; n++ {
						for h := 0; h < H; h++ {
							for w := 0; w < W; w++ {
								oh := h*f.stride + kh - f.pad
								ow := w*f.stride + kw - f.pad
								if oh >= 0 && oh < oH && ow >= 0 && ow < oW {
									xv := f.xData.At(n, ic, h, w)
									gv := grad4d.At(n, oc, oh, ow)
									sum += xv * gv
								}
							}
						}
					}
					dWd[flatIdx4(ic, oc, kh, kw, wdShape)] = sum
				}
			}
		}
	}

	return []*tensor.Tensor{dX, dW}
}

func convTranspose2dForward(x, w, b *autograd.Variable, useBias bool, stride, pad int) *autograd.Variable {
	xShape := x.Data.Shape() // [N, inC, H, W]
	wShape := w.Data.Shape() // [inC, outC, kH, kW]
	N, inC, H, W := xShape[0], xShape[1], xShape[2], xShape[3]
	wInC, outC, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	_ = wInC

	oH := (H-1)*stride - 2*pad + kH
	oW := (W-1)*stride - 2*pad + kW

	outData := make([]float64, N*outC*oH*oW)
	outShape := []int{N, outC, oH, oW}

	for n := 0; n < N; n++ {
		for ic := 0; ic < inC; ic++ {
			for h := 0; h < H; h++ {
				for w_ := 0; w_ < W; w_++ {
					xv := x.Data.At(n, ic, h, w_)
					for oc := 0; oc < outC; oc++ {
						for kh := 0; kh < kH; kh++ {
							for kw := 0; kw < kW; kw++ {
								oh := h*stride + kh - pad
								ow := w_*stride + kw - pad
								if oh >= 0 && oh < oH && ow >= 0 && ow < oW {
									wv := w.Data.At(ic, oc, kh, kw)
									outData[flatIdx4(n, oc, oh, ow, outShape)] += xv * wv
								}
							}
						}
					}
				}
			}
		}
	}

	// Add bias
	if useBias && b != nil {
		bData := b.Data.Data()
		for n := 0; n < N; n++ {
			for oc := 0; oc < outC; oc++ {
				for oh := 0; oh < oH; oh++ {
					for ow := 0; ow < oW; ow++ {
						outData[flatIdx4(n, oc, oh, ow, outShape)] += bData[oc]
					}
				}
			}
		}
	}

	outT := tensor.New(outData, outShape)

	if useBias && b != nil {
		return autograd.NewResult(outT, &convTranspose2dBackward{x.Data, w.Data, stride, pad}, x, w)
	}
	return autograd.NewResult(outT, &convTranspose2dBackward{x.Data, w.Data, stride, pad}, x, w)
}

// flatIdx4 computes flat index for a 4D tensor [d0, d1, d2, d3].
func flatIdx4(i0, i1, i2, i3 int, shape []int) int {
	return i0*shape[1]*shape[2]*shape[3] + i1*shape[2]*shape[3] + i2*shape[3] + i3
}
