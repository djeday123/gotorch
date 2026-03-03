// Пример 6: nn.functional API
//
// Демонстрирует использование stateless functional API:
//   F.ReLU, F.GELU, F.SiLU, F.Softmax, F.LogSoftmax, F.Dropout
//   F.MSELoss, F.L1Loss, F.HuberLoss, F.CrossEntropyLoss
//   F.Linear (ручное управление весами без слоёв-объектов)
//
// Пример: простая сеть с полностью ручным forward pass через F.xxx.
package main

import (
	"fmt"
	"math"

	"github.com/djeday123/gotorch/autograd"
	F "github.com/djeday123/gotorch/nn/functional"
	"github.com/djeday123/gotorch/tensor"
)

func main() {
	fmt.Println("═══════════════════════════════════════════════════")
	fmt.Println("  GoTorch nn.functional — демонстрация")
	fmt.Println("════════════════════════════════════════════════════")

	// ── 1. Активации ─────────────────────────────────────────────────────────
	fmt.Println("── Активации ───────────────────────────────────────")
	x := autograd.NewVar(tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}), true)

	relu := F.ReLU(x)
	gelu := F.GELU(x)
	silu := F.SiLU(x)
	sigmoid := F.Sigmoid(x)

	fmt.Printf("%-12s %v\n", "Вход:", fmtSlice(x.Data.Data()))
	fmt.Printf("%-12s %v\n", "ReLU:", fmtSliceRound(relu.Data.Data()))
	fmt.Printf("%-12s %v\n", "GELU:", fmtSliceRound(gelu.Data.Data()))
	fmt.Printf("%-12s %v\n", "SiLU:", fmtSliceRound(silu.Data.Data()))
	fmt.Printf("%-12s %v\n", "Sigmoid:", fmtSliceRound(sigmoid.Data.Data()))

	// Backward через ReLU
	loss := autograd.Sum(relu)
	loss.Backward()
	fmt.Printf("%-12s %v\n\n", "∂ReLU/∂x:", fmtSlice(x.Grad.Data()))

	// ── 2. Softmax / LogSoftmax ───────────────────────────────────────────────
	fmt.Println("── Softmax / LogSoftmax ────────────────────────────")
	logits2D := autograd.NewVar(tensor.New([]float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	}, []int{2, 3}), false)

	sm := F.Softmax(logits2D, 1)
	lsm := F.LogSoftmax(logits2D, 1)

	row0 := sm.Data.Data()[:3]
	row1 := sm.Data.Data()[3:]
	sumRow0 := row0[0] + row0[1] + row0[2]
	sumRow1 := row1[0] + row1[1] + row1[2]

	fmt.Printf("Softmax row 0: %v → sum=%.4f\n", fmtSliceRound(row0), sumRow0)
	fmt.Printf("Softmax row 1: %v → sum=%.4f\n", fmtSliceRound(row1), sumRow1)
	_ = lsm
	lsmRow0 := lsm.Data.Data()[:3]
	fmt.Printf("LogSoftmax r0: %v\n", fmtSliceRound(lsmRow0))
	fmt.Printf("  exp(log-sm): %v\n\n", fmtSliceRound([]float64{
		math.Exp(lsmRow0[0]), math.Exp(lsmRow0[1]), math.Exp(lsmRow0[2]),
	}))

	// ── 3. Dropout ────────────────────────────────────────────────────────────
	fmt.Println("── Dropout ─────────────────────────────────────────")
	xD := autograd.NewVar(tensor.New([]float64{1, 1, 1, 1, 1, 1, 1, 1}, []int{8}), false)

	evalOut := F.Dropout(xD, 0.5, false) // eval: identity
	fmt.Printf("Eval mode (p=0.5): %v  [identity]\n", fmtSlice(evalOut.Data.Data()))

	// Train mode: примерно половина занулена, остальные * 2
	trainOut := F.Dropout(xD, 0.5, true)
	zeros := 0
	for _, v := range trainOut.Data.Data() {
		if v == 0 {
			zeros++
		}
	}
	fmt.Printf("Train mode (p=0.5): %v  [%d нулей]\n\n", fmtSliceRound(trainOut.Data.Data()), zeros)

	// ── 4. Loss функции ───────────────────────────────────────────────────────
	fmt.Println("── Loss функции ────────────────────────────────────")
	pred := autograd.NewVar(tensor.New([]float64{1.5, 2.5, 3.5}, []int{3}), true)
	tgt := autograd.NewVar(tensor.New([]float64{1.0, 2.0, 3.0}, []int{3}), false)

	mseLoss := F.MSELoss(pred, tgt)
	l1Loss := F.L1Loss(pred, tgt)
	huberLoss := F.HuberLoss(pred, tgt, 1.0)

	fmt.Printf("pred:   %v\n", fmtSlice(pred.Data.Data()))
	fmt.Printf("target: %v\n", fmtSlice(tgt.Data.Data()))
	fmt.Printf("MSELoss:   %.6f  (mean((pred-tgt)^2))\n", mseLoss.Data.Item())
	fmt.Printf("L1Loss:    %.6f  (mean(|pred-tgt|))\n", l1Loss.Data.Item())
	fmt.Printf("HuberLoss: %.6f  (delta=1.0)\n\n", huberLoss.Data.Item())

	// CrossEntropyLoss
	logits := autograd.NewVar(tensor.New([]float64{
		2.0, 1.0, 0.1,  // sample 0: class 0 наиболее вероятен
		0.1, 0.2, 3.0,  // sample 1: class 2 наиболее вероятен
	}, []int{2, 3}), true)
	targets := []int{0, 2}
	ceLoss := F.CrossEntropyLoss(logits, targets)
	fmt.Printf("CrossEntropyLoss (правильные классы): %.6f\n", ceLoss.Data.Item())
	ceLoss.Backward()
	fmt.Printf("∂CE/∂logits[0]: %v\n", fmtSliceRound4(logits.Grad.Data()[:3]))
	fmt.Printf("∂CE/∂logits[1]: %v\n\n", fmtSliceRound4(logits.Grad.Data()[3:]))

	// ── 5. Ручная нейросеть через F.Linear ───────────────────────────────────
	fmt.Println("── Ручная сеть через F.Linear ──────────────────────")
	fmt.Println("(без создания nn.Linear объектов)")

	// Создаём параметры вручную
	w1 := autograd.NewVar(tensor.RandN(4, 2), true)  // [out=4, in=2]
	b1 := autograd.NewVar(tensor.Zeros(4), true)     // [4]
	w2 := autograd.NewVar(tensor.RandN(1, 4), true)  // [out=1, in=4]
	b2 := autograd.NewVar(tensor.Zeros(1), true)     // [1]

	// XOR данные
	xXOR := autograd.NewVar(tensor.New([]float64{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2}), false)
	yXOR := autograd.NewVar(tensor.New([]float64{0, 1, 1, 0}, []int{4, 1}), false)

	allParams := []*autograd.Variable{w1, b1, w2, b2}
	optF := newSimpleAdam(allParams, 0.01)

	fmt.Println("Обучение XOR через F.Linear (50 шагов):")
	for step := 1; step <= 50; step++ {
		// Zero grad
		for _, p := range allParams {
			p.ZeroGrad()
		}

		// Forward
		h := F.Linear(xXOR, w1, b1) // [4, 4]
		h = F.ReLU(h)
		out := F.Linear(h, w2, b2)   // [4, 1]
		out = F.Sigmoid(out)

		loss := F.MSELoss(out, yXOR)
		loss.Backward()
		optF.step()

		if step%10 == 0 {
			fmt.Printf("  step %2d  loss=%.4f\n", step, loss.Data.Item())
		}
	}
	fmt.Println()

	// ── 6. L1 vs Huber vs MSE сравнение ──────────────────────────────────────
	fmt.Println("── L1 vs Huber vs MSE (разные ошибки) ─────────────")
	fmt.Printf("%-8s  %-10s  %-10s  %-10s\n", "err", "MSE", "L1", "Huber(d=1)")
	fmt.Println("────────────────────────────────────────────────")
	for _, err := range []float64{0.1, 0.5, 1.0, 2.0, 5.0} {
		p2 := autograd.NewVar(tensor.New([]float64{err}, []int{1}), false)
		t2 := autograd.NewVar(tensor.New([]float64{0}, []int{1}), false)
		mse := err * err                    // (0.5*err^2/1 for huber would be different)
		l1 := err
		var huber float64
		if err <= 1.0 {
			huber = 0.5 * err * err
		} else {
			huber = err - 0.5
		}
		_ = p2
		_ = t2
		fmt.Printf("%-8.1f  %-10.4f  %-10.4f  %-10.4f\n", err, mse, l1, huber)
	}

	fmt.Println("\n✅ Functional API демонстрация завершена!")
}

// ── Простой Adam для примера ─────────────────────────────────────────────────

type simpleAdam struct {
	params []*autograd.Variable
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64
	m      [][]float64
	v      [][]float64
	t      int
}

func newSimpleAdam(params []*autograd.Variable, lr float64) *simpleAdam {
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, p.Data.Size())
		v[i] = make([]float64, p.Data.Size())
	}
	return &simpleAdam{params: params, lr: lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: m, v: v}
}

func (a *simpleAdam) step() {
	a.t++
	for i, p := range a.params {
		if p.Grad == nil {
			continue
		}
		g := p.Grad.Data()
		d := p.Data.Data()
		for j := range g {
			a.m[i][j] = a.beta1*a.m[i][j] + (1-a.beta1)*g[j]
			a.v[i][j] = a.beta2*a.v[i][j] + (1-a.beta2)*g[j]*g[j]
			mHat := a.m[i][j] / (1 - math.Pow(a.beta1, float64(a.t)))
			vHat := a.v[i][j] / (1 - math.Pow(a.beta2, float64(a.t)))
			d[j] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
		// Обновить данные тензора
		p.Data = tensor.New(d, p.Data.Shape())
	}
}

// ── Утилиты форматирования ───────────────────────────────────────────────────

func fmtSlice(s []float64) string {
	r := "["
	for i, v := range s {
		if i > 0 {
			r += ", "
		}
		r += fmt.Sprintf("%.2f", v)
	}
	return r + "]"
}

func fmtSliceRound(s []float64) string {
	r := "["
	for i, v := range s {
		if i > 0 {
			r += ", "
		}
		r += fmt.Sprintf("%.3f", v)
	}
	return r + "]"
}

func fmtSliceRound4(s []float64) string {
	r := "["
	for i, v := range s {
		if i > 0 {
			r += ", "
		}
		r += fmt.Sprintf("%.4f", v)
	}
	return r + "]"
}
