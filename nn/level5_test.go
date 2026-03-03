package nn_test

import (
	"math"
	"testing"

	"github.com/djeday123/gotorch/autograd"
	F "github.com/djeday123/gotorch/nn/functional"
	"github.com/djeday123/gotorch/tensor"

	"github.com/djeday123/gotorch/nn"
)

// ── Functional: Activations ───────────────────────────────────────────────────

func TestFunctionalReLU(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}), true)
	out := F.ReLU(x)
	got := out.Data.Data()
	want := []float64{0, 0, 0, 1, 2}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("F.ReLU[%d]: want %v, got %v", i, v, got[i])
		}
	}
	loss := autograd.Sum(out)
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("F.ReLU: no gradient")
	}
	grad := x.Grad.Data()
	wantGrad := []float64{0, 0, 0, 1, 1}
	for i, v := range wantGrad {
		if grad[i] != v {
			t.Errorf("F.ReLU grad[%d]: want %v, got %v", i, v, grad[i])
		}
	}
}

func TestFunctionalGELU(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{0, 1, -1}, []int{3}), true)
	out := F.GELU(x)
	// GELU(0) = 0, GELU(1) ≈ 0.841
	if math.Abs(out.Data.At(0)) > 1e-6 {
		t.Errorf("F.GELU(0): want ~0, got %v", out.Data.At(0))
	}
	if out.Data.At(1) < 0.8 || out.Data.At(1) > 0.9 {
		t.Errorf("F.GELU(1): want ~0.841, got %v", out.Data.At(1))
	}
}

func TestFunctionalLeakyReLU(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{-2, 1}, []int{2}), true)
	out := F.LeakyReLU(x, 0.1)
	if math.Abs(out.Data.At(0)-(-0.2)) > 1e-9 {
		t.Errorf("F.LeakyReLU(-2): want -0.2, got %v", out.Data.At(0))
	}
	if out.Data.At(1) != 1 {
		t.Errorf("F.LeakyReLU(1): want 1, got %v", out.Data.At(1))
	}
}

func TestFunctionalSiLU(t *testing.T) {
	// SiLU(x) = x * sigmoid(x); SiLU(0) = 0
	x := autograd.NewVar(tensor.New([]float64{0, 2}, []int{2}), true)
	out := F.SiLU(x)
	if math.Abs(out.Data.At(0)) > 1e-9 {
		t.Errorf("F.SiLU(0): want 0, got %v", out.Data.At(0))
	}
	// SiLU(2) ≈ 2 * 0.8808 ≈ 1.762
	if out.Data.At(1) < 1.7 || out.Data.At(1) > 1.85 {
		t.Errorf("F.SiLU(2): want ~1.762, got %v", out.Data.At(1))
	}
}

// ── Functional: Softmax ───────────────────────────────────────────────────────

func TestFunctionalSoftmax(t *testing.T) {
	// Softmax probabilities must sum to 1 along dim=1
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}), false)
	out := F.Softmax(x, 1)
	data := out.Data.Data()
	// Row 0: sum of first 3
	row0sum := data[0] + data[1] + data[2]
	if math.Abs(row0sum-1.0) > 1e-6 {
		t.Errorf("F.Softmax row0 sum: want 1, got %v", row0sum)
	}
	row1sum := data[3] + data[4] + data[5]
	if math.Abs(row1sum-1.0) > 1e-6 {
		t.Errorf("F.Softmax row1 sum: want 1, got %v", row1sum)
	}
}

func TestFunctionalLogSoftmax(t *testing.T) {
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3}, []int{1, 3}), false)
	out := F.LogSoftmax(x, 1)
	data := out.Data.Data()
	// exp of log-softmax outputs should sum to ~1
	sum := math.Exp(data[0]) + math.Exp(data[1]) + math.Exp(data[2])
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("F.LogSoftmax: exp sum should be 1, got %v", sum)
	}
}

// ── Functional: Dropout ───────────────────────────────────────────────────────

func TestFunctionalDropout(t *testing.T) {
	// In eval mode (training=false), dropout is a no-op
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}), false)
	out := F.Dropout(x, 0.5, false)
	got := out.Data.Data()
	want := x.Data.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("F.Dropout eval[%d]: want %v, got %v", i, want[i], got[i])
		}
	}
}

// ── Functional: MSELoss ───────────────────────────────────────────────────────

func TestFunctionalMSELoss(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), true)
	target := autograd.NewVar(tensor.New([]float64{1, 2, 3}, []int{3}), false)
	loss := F.MSELoss(pred, target)
	if loss.Data.Item() != 0 {
		t.Errorf("F.MSELoss perfect pred: want 0, got %v", loss.Data.Item())
	}

	// Compare with nn.MSELoss
	pred2 := autograd.NewVar(tensor.New([]float64{2, 4, 6}, []int{3}), true)
	nnLoss := nn.MSELoss(pred2, target)
	fLoss := F.MSELoss(pred2, target)
	if math.Abs(nnLoss.Data.Item()-fLoss.Data.Item()) > 1e-9 {
		t.Errorf("F.MSELoss != nn.MSELoss: %v vs %v", fLoss.Data.Item(), nnLoss.Data.Item())
	}
}

// ── Functional: L1Loss ────────────────────────────────────────────────────────

func TestFunctionalL1Loss(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{1, 3, 5}, []int{3}), true)
	target := autograd.NewVar(tensor.New([]float64{2, 2, 4}, []int{3}), false)
	loss := F.L1Loss(pred, target)
	// |1-2| + |3-2| + |5-4| = 3, mean = 1
	if math.Abs(loss.Data.Item()-1.0) > 1e-9 {
		t.Errorf("F.L1Loss: want 1.0, got %v", loss.Data.Item())
	}
	loss.Backward()
	if pred.Grad == nil {
		t.Fatal("F.L1Loss: no gradient")
	}
}

// ── Functional: HuberLoss ─────────────────────────────────────────────────────

func TestFunctionalHuberLoss(t *testing.T) {
	delta := 1.0
	// Near 0: behaves like 0.5 * x^2 / delta
	pred := autograd.NewVar(tensor.New([]float64{0.5}, []int{1}), true)
	target := autograd.NewVar(tensor.New([]float64{0}, []int{1}), false)
	loss := F.HuberLoss(pred, target, delta)
	// 0.5 * 0.25 / 1.0 = 0.125
	if math.Abs(loss.Data.Item()-0.125) > 1e-9 {
		t.Errorf("F.HuberLoss near0: want 0.125, got %v", loss.Data.Item())
	}

	// Far from 0: behaves like |x| - 0.5*delta
	pred2 := autograd.NewVar(tensor.New([]float64{3.0}, []int{1}), true)
	loss2 := F.HuberLoss(pred2, target, delta)
	// 3.0 - 0.5 = 2.5
	if math.Abs(loss2.Data.Item()-2.5) > 1e-9 {
		t.Errorf("F.HuberLoss far: want 2.5, got %v", loss2.Data.Item())
	}
}

// ── Functional: CrossEntropy ──────────────────────────────────────────────────

func TestFunctionalCrossEntropy(t *testing.T) {
	logits := autograd.NewVar(tensor.New([]float64{2, 1, 0.5, 0.5, 1, 2}, []int{2, 3}), true)
	targets := []int{0, 2}
	fLoss := F.CrossEntropyLoss(logits, targets)
	nnLoss := nn.CrossEntropyLoss(logits, targets)
	if math.Abs(fLoss.Data.Item()-nnLoss.Data.Item()) > 1e-9 {
		t.Errorf("F.CrossEntropy != nn.CrossEntropy: %v vs %v", fLoss.Data.Item(), nnLoss.Data.Item())
	}
}

func TestFunctionalNLLLoss(t *testing.T) {
	// log-probs: 2 classes, 2 samples
	logProbs := autograd.NewVar(tensor.New([]float64{
		math.Log(0.8), math.Log(0.2),
		math.Log(0.3), math.Log(0.7),
	}, []int{2, 2}), true)
	targets := []int{0, 1}
	loss := F.NLLLoss(logProbs, targets)
	// Expected: -(log(0.8) + log(0.7)) / 2
	expected := -(math.Log(0.8) + math.Log(0.7)) / 2
	if math.Abs(loss.Data.Item()-expected) > 1e-9 {
		t.Errorf("F.NLLLoss: want %v, got %v", expected, loss.Data.Item())
	}
}

// ── nn: L1Loss ────────────────────────────────────────────────────────────────

func TestL1LossLayer(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{0, 3}, []int{2}), true)
	target := autograd.NewVar(tensor.New([]float64{2, 2}, []int{2}), false)
	loss := nn.L1Loss(pred, target)
	// |0-2| + |3-2| = 2+1 = 3, mean = 1.5
	if math.Abs(loss.Data.Item()-1.5) > 1e-9 {
		t.Errorf("nn.L1Loss: want 1.5, got %v", loss.Data.Item())
	}
	loss.Backward()
	if pred.Grad == nil {
		t.Fatal("nn.L1Loss: no grad on pred")
	}
	// grad[0] = -1/2 (pred < target), grad[1] = +1/2 (pred > target)
	g := pred.Grad.Data()
	if math.Abs(g[0]-(-0.5)) > 1e-9 {
		t.Errorf("nn.L1Loss grad[0]: want -0.5, got %v", g[0])
	}
	if math.Abs(g[1]-0.5) > 1e-9 {
		t.Errorf("nn.L1Loss grad[1]: want 0.5, got %v", g[1])
	}
}

// ── nn: HuberLoss ─────────────────────────────────────────────────────────────

func TestHuberLossLayer(t *testing.T) {
	pred := autograd.NewVar(tensor.New([]float64{0.5, 3.0}, []int{2}), true)
	target := autograd.NewVar(tensor.New([]float64{0, 0}, []int{2}), false)
	delta := 1.0
	loss := nn.HuberLoss(pred, target, delta)
	// elem0: 0.5 * 0.25 / 1 = 0.125; elem1: 3.0 - 0.5 = 2.5 → mean = (0.125+2.5)/2 = 1.3125
	if math.Abs(loss.Data.Item()-1.3125) > 1e-9 {
		t.Errorf("nn.HuberLoss: want 1.3125, got %v", loss.Data.Item())
	}
	loss.Backward()
	if pred.Grad == nil {
		t.Fatal("nn.HuberLoss: no gradient")
	}
}

// ── nn: NLLLoss ───────────────────────────────────────────────────────────────

func TestNLLLossLayer(t *testing.T) {
	logProbs := autograd.NewVar(tensor.New([]float64{
		math.Log(0.9), math.Log(0.1),
		math.Log(0.4), math.Log(0.6),
	}, []int{2, 2}), true)
	targets := []int{0, 1}
	loss := nn.NLLLoss(logProbs, targets)
	expected := -(math.Log(0.9) + math.Log(0.6)) / 2
	if math.Abs(loss.Data.Item()-expected) > 1e-9 {
		t.Errorf("nn.NLLLoss: want %v, got %v", expected, loss.Data.Item())
	}
	loss.Backward()
	if logProbs.Grad == nil {
		t.Fatal("nn.NLLLoss: no gradient")
	}
}

// ── nn: KLDivLoss ─────────────────────────────────────────────────────────────

func TestKLDivLossLayer(t *testing.T) {
	// When input == log(target), KL divergence should be 0
	target := []float64{0.5, 0.5}
	logInput := []float64{math.Log(0.5), math.Log(0.5)}
	inp := autograd.NewVar(tensor.New(logInput, []int{1, 2}), true)
	tgt := autograd.NewVar(tensor.New(target, []int{1, 2}), false)
	loss := nn.KLDivLoss(inp, tgt)
	if math.Abs(loss.Data.Item()) > 1e-9 {
		t.Errorf("KLDivLoss(p||p): want 0, got %v", loss.Data.Item())
	}

	// KL > 0 for different distributions
	inp2 := autograd.NewVar(tensor.New([]float64{math.Log(0.9), math.Log(0.1)}, []int{1, 2}), true)
	loss2 := nn.KLDivLoss(inp2, tgt)
	if loss2.Data.Item() <= 0 {
		t.Errorf("KLDivLoss different dists: want > 0, got %v", loss2.Data.Item())
	}
}

// ── nn: ConvTranspose2d ───────────────────────────────────────────────────────

func TestConvTranspose2dShape(t *testing.T) {
	// Input [1, 2, 4, 4], ConvTranspose2d(2→4, k=3, s=1, p=0)
	// output H = (4-1)*1 - 2*0 + 3 = 6
	ct := nn.NewConvTranspose2d(2, 4, 3, 1, 0)
	x := autograd.NewVar(tensor.Zeros(1, 2, 4, 4), true)
	out := ct.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 6 || shape[3] != 6 {
		t.Errorf("ConvTranspose2d shape: want [1,4,6,6], got %v", shape)
	}
}

func TestConvTranspose2dBackward(t *testing.T) {
	ct := nn.NewConvTranspose2d(1, 1, 2, 1, 0)
	x := autograd.NewVar(tensor.Rand(1, 1, 3, 3), true)
	out := ct.Forward(x)
	loss := autograd.Mean(out)
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("ConvTranspose2d: no gradient on input")
	}
	if ct.Weight.Grad == nil {
		t.Fatal("ConvTranspose2d: no gradient on weight")
	}
}

func TestConvTranspose2dStrided(t *testing.T) {
	// stride=2: output H = (3-1)*2 - 2*0 + 2 = 6
	ct := nn.NewConvTranspose2d(1, 1, 2, 2, 0)
	x := autograd.NewVar(tensor.Zeros(1, 1, 3, 3), true)
	out := ct.Forward(x)
	shape := out.Data.Shape()
	if shape[2] != 6 || shape[3] != 6 {
		t.Errorf("ConvTranspose2d stride=2 shape: want [1,1,6,6], got %v", shape)
	}
}

// ── nn: Upsample ──────────────────────────────────────────────────────────────

func TestUpsampleNearest(t *testing.T) {
	up := nn.NewUpsample(2, "nearest")
	x := autograd.NewVar(tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{1, 2, 2, 2}), true)
	out := up.Forward(x)
	shape := out.Data.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 4 || shape[3] != 4 {
		t.Errorf("Upsample shape: want [1,2,4,4], got %v", shape)
	}
	// Value at (0,0,0,0) and (0,0,0,1) should both equal input (0,0,0,0) = 1
	if out.Data.At(0, 0, 0, 0) != 1 {
		t.Errorf("Upsample value: want 1, got %v", out.Data.At(0, 0, 0, 0))
	}
	if out.Data.At(0, 0, 0, 1) != 1 {
		t.Errorf("Upsample value: want 1, got %v", out.Data.At(0, 0, 0, 1))
	}
	if out.Data.At(0, 0, 0, 2) != 2 {
		t.Errorf("Upsample value: want 2, got %v", out.Data.At(0, 0, 0, 2))
	}
}

func TestUpsampleBackward(t *testing.T) {
	up := nn.NewUpsample(2, "nearest")
	x := autograd.NewVar(tensor.Rand(1, 1, 2, 2), true)
	out := up.Forward(x)
	loss := autograd.Mean(out)
	loss.Backward()
	if x.Grad == nil {
		t.Fatal("Upsample: no gradient on input")
	}
}

// ── nn: ModuleList ────────────────────────────────────────────────────────────

func TestModuleList(t *testing.T) {
	l1 := nn.NewLinear(4, 8, true)   // 4*8 + 8 = 40 params
	l2 := nn.NewLinear(8, 2, true)   // 8*2 + 2 = 18 params
	ml := nn.NewModuleList(l1, l2)

	if ml.Len() != 2 {
		t.Errorf("ModuleList.Len: want 2, got %d", ml.Len())
	}
	params := ml.Parameters()
	if len(params) != 4 { // weight+bias for each layer
		t.Errorf("ModuleList.Parameters: want 4, got %d", len(params))
	}
	// Test forward
	x := autograd.NewVar(tensor.Zeros(1, 4), false)
	out := ml.Forward(x)
	if out.Data.Shape()[1] != 2 {
		t.Errorf("ModuleList.Forward output shape: want [1,2], got %v", out.Data.Shape())
	}
}

func TestModuleListAppend(t *testing.T) {
	ml := nn.NewModuleList()
	ml.Append(nn.NewReLU())
	ml.Append(nn.NewReLU())
	if ml.Len() != 2 {
		t.Errorf("ModuleList after append: want 2, got %d", ml.Len())
	}
}

// ── nn: ModelSummary ──────────────────────────────────────────────────────────

func TestModelSummary(t *testing.T) {
	model := nn.NewLinear(10, 5, true) // 10*5 + 5 = 55 params
	s := nn.Summary(model)
	if s.TotalParams != 55 {
		t.Errorf("Summary.TotalParams: want 55, got %d", s.TotalParams)
	}
	if s.TrainableParams != 55 {
		t.Errorf("Summary.TrainableParams: want 55, got %d", s.TrainableParams)
	}
	if len(s.Layers) == 0 {
		t.Error("Summary.Layers: want at least 1 layer")
	}
}

func TestModelSummaryPrint(t *testing.T) {
	model := nn.NewLinear(4, 2, true)
	// Just make sure it doesn't panic
	nn.PrintSummary(model)
}
