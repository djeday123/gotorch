package autograd

import "github.com/djeday123/gotorch/tensor"

// backward performs a full backward pass starting from v with the given gradient.
// It uses a topological sort so that each node's gradient is fully accumulated
// before its gradFn is called.
func backward(v *Variable, grad *tensor.Tensor) {
	// Build topological order
	order := make([]*Variable, 0)
	visited := make(map[*Variable]bool)

	var topo func(node *Variable)
	topo = func(node *Variable) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.children {
			topo(child)
		}
		order = append(order, node)
	}
	topo(v)

	// grads maps each variable to its accumulated gradient
	grads := make(map[*Variable]*tensor.Tensor)
	grads[v] = grad

	// Process in reverse topological order (output → input)
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		g, ok := grads[node]
		if !ok {
			continue
		}

		// Accumulate into leaf .Grad
		if node.isLeaf && node.RequiresGrad {
			if node.Grad == nil {
				node.Grad = tensor.Zeros(node.Data.Shape()...)
			}
			node.Grad = tensor.Add(node.Grad, g)
			continue
		}

		// Call gradFn to get input gradients
		if node.gradFn == nil {
			continue
		}
		inputGrads := node.gradFn.Apply(g)

		for j, child := range node.children {
			if j >= len(inputGrads) || inputGrads[j] == nil {
				continue
			}
			if !child.RequiresGrad {
				continue
			}
			childGrad := inputGrads[j]
			// Accumulate
			if existing, ok2 := grads[child]; ok2 {
				grads[child] = tensor.Add(existing, childGrad)
			} else {
				grads[child] = childGrad
			}
		}
	}

	// Final accumulation for leaf nodes that appeared in grads but were
	// not processed (possible for nodes visited as leaves).
	for node, g := range grads {
		if node.isLeaf && node.RequiresGrad && node.Grad == nil {
			node.Grad = tensor.Zeros(node.Data.Shape()...)
			node.Grad = tensor.Add(node.Grad, g)
		}
	}
}

// unbroadcastGrad reduces a gradient back to the target shape,
// summing over broadcast axes (the reverse of broadcastTo).
func unbroadcastGrad(grad *tensor.Tensor, targetShape []int) *tensor.Tensor {
	g := grad
	// If ndim differs, sum over leading axes
	for g.Ndim() > len(targetShape) {
		g = tensor.Sum(g, 0, false)
	}
	// Sum over axes where targetShape[i] == 1
	for i, d := range targetShape {
		if d == 1 && g.Shape()[i] > 1 {
			g = tensor.Sum(g, i, true)
		}
	}
	return g
}
