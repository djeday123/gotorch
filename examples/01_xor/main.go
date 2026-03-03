// Пример 1: XOR — базовый MLP
//
// Задача: научить сеть решать XOR:
//   f(0,0)=0  f(0,1)=1  f(1,0)=1  f(1,1)=0
//
// Архитектура: Linear(2→8) → Tanh → Linear(8→1) → Sigmoid
// Оптимизатор: Adam, lr=0.05
// Loss: MSELoss
package main

import (
	"fmt"
	"math"

	"github.com/djeday123/gotorch/autograd"
	"github.com/djeday123/gotorch/nn"
	"github.com/djeday123/gotorch/optim"
	"github.com/djeday123/gotorch/tensor"
)

func main() {
	// ── Данные ───────────────────────────────────────────────────────────────
	X := autograd.NewVar(tensor.New([]float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, []int{4, 2}), false)

	Y := autograd.NewVar(tensor.New([]float64{0, 1, 1, 0}, []int{4, 1}), false)

	// ── Модель ───────────────────────────────────────────────────────────────
	model := nn.NewSequential(
		nn.NewLinear(2, 8, true),
		nn.NewTanh(),
		nn.NewLinear(8, 1, true),
		nn.NewSigmoid(),
	)

	opt := optim.NewAdam(model.Parameters(), 0.05, 0.9, 0.999, 1e-8)

	fmt.Println("Обучение XOR...")
	fmt.Printf("%-8s  %-10s\n", "Epoch", "Loss")
	fmt.Println("──────────────────")

	// ── Обучение ─────────────────────────────────────────────────────────────
	for epoch := 1; epoch <= 3000; epoch++ {
		opt.ZeroGrad()

		pred := model.Forward(X)
		loss := nn.MSELoss(pred, Y)
		loss.Backward()
		opt.Step()

		if epoch%300 == 0 {
			fmt.Printf("%-8d  %.6f\n", epoch, loss.Data.Item())
		}
	}

	// ── Результаты ───────────────────────────────────────────────────────────
	fmt.Println("\nПредсказания:")
	fmt.Printf("%-12s  %-12s  %-12s\n", "Вход", "Ожидание", "Сеть")
	fmt.Println("────────────────────────────────────────")

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	expected := []float64{0, 1, 1, 0}

	allCorrect := true
	for i, inp := range inputs {
		xI := autograd.NewVar(tensor.New(inp, []int{1, 2}), false)
		out := model.Forward(xI)
		pred := out.Data.Item()
		rounded := math.Round(pred)

		correct := "✓"
		if rounded != expected[i] {
			correct = "✗"
			allCorrect = false
		}
		fmt.Printf("[%.0f, %.0f]      %.0f          %.4f  %s\n",
			inp[0], inp[1], expected[i], pred, correct)
	}

	if allCorrect {
		fmt.Println("\n✅ XOR решён!")
	} else {
		fmt.Println("\n⚠️  Не все верно — попробуй ещё несколько эпох")
	}

	fmt.Println()
	nn.PrintSummary(model)
}
