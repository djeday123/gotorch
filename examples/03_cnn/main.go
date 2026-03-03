// Пример 3: CNN-классификатор
//
// Синтетические "изображения" 16×16, 4 класса (разные текстуры).
// Показывает: Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, Flatten2d.
//
// Архитектура:
//   Conv2d(1→16,3,pad=1) → BN → ReLU → MaxPool2d(2)
//   Conv2d(16→32,3,pad=1) → BN → ReLU → MaxPool2d(2)
//   AdaptiveAvgPool2d(2,2)
//   Flatten → Linear(128→4)
package main

import (
	"fmt"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/nn"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
)

const (
	H      = 16
	W      = 16
	nCls   = 4
	nTrain = 400
	nTest  = 100
)

// Генерация: класс = паттерн изображения
func makeImages(n int) ([][]float64, []int) {
	imgs := make([][]float64, n)
	labs := make([]int, n)
	for i := 0; i < n; i++ {
		cls := rand.Intn(nCls)
		labs[i] = cls
		img := make([]float64, H*W)
		for h := 0; h < H; h++ {
			for w := 0; w < W; w++ {
				var v float64
				switch cls {
				case 0: // горизонтальные полосы
					if h%4 < 2 { v = 1 }
				case 1: // вертикальные полосы
					if w%4 < 2 { v = 1 }
				case 2: // шахмат
					if (h+w)%2 == 0 { v = 1 }
				case 3: // круг
					dh := float64(h-H/2); dw := float64(w-W/2)
					if dh*dh+dw*dw < float64(H*H/6) { v = 1 }
				}
				img[h*W+w] = v + rand.NormFloat64()*0.15
			}
		}
		imgs[i] = img
	}
	return imgs, labs
}

func toBatch(imgs [][]float64, idxs []int) *tensor.Tensor {
	n := len(idxs)
	d := make([]float64, n*H*W)
	for b, i := range idxs {
		copy(d[b*H*W:], imgs[i])
	}
	return tensor.New(d, []int{n, 1, H, W})
}

func main() {
	rand.Seed(42)
	trainImgs, trainLabs := makeImages(nTrain)
	testImgs, testLabs := makeImages(nTest)

	fmt.Printf("Train: %d  Test: %d  Классов: %d  Размер: %d×%d\n\n",
		nTrain, nTest, nCls, H, W)

	// ── Модель ───────────────────────────────────────────────────────────────
	conv1 := nn.NewConv2d(1, 16, 3, 1, 1, true)
	bn1   := nn.NewBatchNorm2d(16)
	conv2 := nn.NewConv2d(16, 32, 3, 1, 1, true)
	bn2   := nn.NewBatchNorm2d(32)
	pool  := nn.NewMaxPool2d(2, 2)
	avgP  := nn.NewAdaptiveAvgPool2d(2, 2)
	fc    := nn.NewLinear(32*2*2, nCls, true)
	relu  := nn.NewReLU()

	forward := func(x *autograd.Variable, training bool) *autograd.Variable {
		if training { bn1.Train(); bn2.Train() } else { bn1.Eval(); bn2.Eval() }
		out := relu.Forward(bn1.Forward(conv1.Forward(x))) // [N,16,16,16]
		out = pool.Forward(out)                            // [N,16,8,8]
		out = relu.Forward(bn2.Forward(conv2.Forward(out)))// [N,32,8,8]
		out = pool.Forward(out)                            // [N,32,4,4]
		out = avgP.Forward(out)                            // [N,32,2,2]
		out = nn.Flatten2d(out)                            // [N,128]
		return fc.Forward(out)                             // [N,4]
	}

	var params []*autograd.Variable
	for _, m := range []nn.Module{conv1, bn1, conv2, bn2, fc} {
		params = append(params, m.Parameters()...)
	}
	opt := optim.NewAdamW(params, 0.001, 0.9, 0.999, 1e-8, 0.01)

	totalP := 0
	for _, p := range params { totalP += p.Data.Size() }
	fmt.Printf("Параметров: %d\n\n", totalP)
	fmt.Printf("%-6s  %-10s  %-12s  %-10s\n", "Epoch", "Loss", "Train Acc", "Test Acc")
	fmt.Println("──────────────────────────────────────────────")

	batchSz := 32
	for epoch := 1; epoch <= 25; epoch++ {
		perm := rand.Perm(nTrain)
		totalLoss, steps := 0.0, 0

		for start := 0; start < nTrain; start += batchSz {
			end := start + batchSz
			if end > nTrain { end = nTrain }
			idxs := perm[start:end]
			labs := make([]int, len(idxs))
			for b, i := range idxs { labs[b] = trainLabs[i] }

			xBatch := toBatch(trainImgs, idxs)
			xV := autograd.NewVar(xBatch, false)

			for _, p := range params { p.ZeroGrad() }
			logits := forward(xV, true)
			loss := nn.CrossEntropyLoss(logits, labs)
			loss.Backward()
			optim.ClipGradNorm(params, 1.0)
			opt.Step()

			totalLoss += loss.Data.Item()
			steps++
		}

		trainAcc := evalCNN(forward, trainImgs[:100], trainLabs[:100])
		testAcc  := evalCNN(forward, testImgs, testLabs)
		fmt.Printf("%-6d  %-10.4f  %-12.1f  %-10.1f\n",
			epoch, totalLoss/float64(steps), trainAcc, testAcc)
	}
	fmt.Println("\n✅ CNN обучена!")
}

func evalCNN(forward func(*autograd.Variable, bool) *autograd.Variable, imgs [][]float64, labs []int) float64 {
	n := len(imgs)
	idxs := make([]int, n)
	for i := range idxs { idxs[i] = i }
	xBatch := toBatch(imgs, idxs)
	logits := forward(autograd.NewVar(xBatch, false), false)
	correct := 0
	nC := logits.Data.Shape()[1]
	for i := 0; i < n; i++ {
		best, bestV := 0, logits.Data.At(i, 0)
		for c := 1; c < nC; c++ {
			if v := logits.Data.At(i, c); v > bestV { bestV = v; best = c }
		}
		if best == labs[i] { correct++ }
	}
	return float64(correct) / float64(n) * 100
}
