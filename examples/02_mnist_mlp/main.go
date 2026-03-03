// Пример 2: MLP-классификатор с DataLoader
//
// 10 классов, синтетические данные 64-мерные.
// DataLoader (shuffle + prefetch), CrossEntropyLoss, Adam, StepLR.
package main

import (
	"fmt"
	"math/rand"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/data"
	"github.com/djeday123/gotorch/nn"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
)

const (
	nSamples = 1000
	nClasses = 10
	inDim    = 64
	nEpochs  = 20
	bs       = 32
)

func makeDataset(n int) (*tensor.Tensor, *tensor.Tensor) {
	xd := make([]float64, n*inDim)
	yd := make([]float64, n)
	// Классы перекрываются — задача нетривиальная
	centers := make([][]float64, nClasses)
	for c := range centers {
		centers[c] = make([]float64, inDim)
		// Только первые 8 признаков информативны
		for d := 0; d < 8; d++ {
			centers[c][d] = float64(c%4)*0.8 - 1.6 + float64(c/4)*0.8
		}
	}
	for i := 0; i < n; i++ {
		cls := rand.Intn(nClasses)
		yd[i] = float64(cls)
		for d := 0; d < inDim; d++ {
			noise := 1.2
			xd[i*inDim+d] = centers[cls][d] + rand.NormFloat64()*noise
		}
	}
	return tensor.New(xd, []int{n, inDim}),
		tensor.New(yd, []int{n, 1})
}

func accuracy(model nn.Module, x, y *tensor.Tensor) float64 {
	n := x.Shape()[0]
	xV := autograd.NewVar(x, false)
	logits := model.Forward(xV)
	correct := 0
	for i := 0; i < n; i++ {
		best, bestV := 0, logits.Data.At(i, 0)
		for c := 1; c < nClasses; c++ {
			if v := logits.Data.At(i, c); v > bestV {
				bestV = v; best = c
			}
		}
		if best == int(y.At(i, 0)) {
			correct++
		}
	}
	return float64(correct) / float64(n) * 100
}

func main() {
	rand.Seed(42)
	xTrain, yTrain := makeDataset(nSamples)
	xTest, yTest := makeDataset(200)

	fmt.Printf("Train: %d  Test: %d  Classes: %d  Features: %d\n\n",
		nSamples, 200, nClasses, inDim)

	// DataLoader — x и y хранятся вместе в TensorDataset
	ds := data.NewTensorDataset(xTrain, yTrain)
	loader := data.NewDataLoader(ds, bs,
		data.WithShuffle(true),
		data.WithPrefetch(2),
	)

	model := nn.NewSequential(
		nn.NewLinear(inDim, 128, true), nn.NewReLU(),
		nn.NewLinear(128, 64, true), nn.NewReLU(),
		nn.NewLinear(64, nClasses, true),
	)
	opt := optim.NewAdamW(model.Parameters(), 3e-4, 0.9, 0.999, 1e-8, 0.01)
	sched := optim.NewStepLR(opt, 7, 0.5)

	totalP := 0
	for _, p := range model.Parameters() {
		totalP += p.Data.Size()
	}
	fmt.Printf("Параметров: %d\n\n", totalP)
	fmt.Printf("%-6s  %-10s  %-12s  %-10s\n", "Epoch", "Loss", "Train Acc", "Test Acc")
	fmt.Println("──────────────────────────────────────────────")

	for epoch := 1; epoch <= nEpochs; epoch++ {
		loader.Reset()
		totalLoss, steps := 0.0, 0

		for loader.HasNext() {
			batch := loader.Next()
			bX, bY := batch.X, batch.Y
			bN := bX.Shape()[0]

			// Извлекаем метки из bY
			labels := make([]int, bN)
			for i := 0; i < bN; i++ {
				labels[i] = int(bY.At(i, 0))
			}

			model.ZeroGrad()
			xV := autograd.NewVar(bX, false)
			logits := model.Forward(xV)
			loss := nn.CrossEntropyLoss(logits, labels)
			loss.Backward()
			optim.ClipGradNorm(model.Parameters(), 1.0)
			opt.Step()

			totalLoss += loss.Data.Item()
			steps++
		}

		sched.Step()
		fmt.Printf("%-6d  %-10.4f  %-12.2f  %-10.2f\n",
			epoch, totalLoss/float64(steps),
			accuracy(model, xTrain, yTrain),
			accuracy(model, xTest, yTest))
	}
	fmt.Println("\n✅ MLP обучен!")
}
