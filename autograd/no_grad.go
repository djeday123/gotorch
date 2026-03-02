package autograd

import "sync/atomic"

// gradEnabled is 1 when autograd is active (default), 0 when disabled.
var gradEnabled int32 = 1

// IsGradEnabled reports whether gradient computation is currently enabled.
func IsGradEnabled() bool { return atomic.LoadInt32(&gradEnabled) == 1 }

// NoGrad runs fn with gradient computation disabled.
// Nested calls are safe: the outer state is restored when fn returns.
//
//	autograd.NoGrad(func() {
//	    pred := model.Forward(x)   // no graph built
//	    fmt.Println(pred.Data())
//	})
func NoGrad(fn func()) {
	old := atomic.SwapInt32(&gradEnabled, 0)
	defer atomic.StoreInt32(&gradEnabled, old)
	fn()
}

// EnableGrad runs fn with gradient computation enabled.
// Useful to re-enable grad inside a NoGrad block.
func EnableGrad(fn func()) {
	old := atomic.SwapInt32(&gradEnabled, 1)
	defer atomic.StoreInt32(&gradEnabled, old)
	fn()
}
