package autograd

import "sync/atomic"

// noGradDepth counts the number of currently-active NoGrad scopes (process-wide).
// Zero means gradient computation is enabled (the default); any positive value
// means at least one NoGrad block is active and grad tracking should be off.
//
// Using a counter rather than a boolean fixes a nesting race that the previous
// SwapInt32/restore approach had: when two goroutines entered NoGrad in
// interleaved order, the outer goroutine could observe its NoGrad scope
// silently expire because the inner goroutine restored the older value.
//
// NOTE: this remains a process-wide flag — it is not goroutine-local. If you
// need per-goroutine grad state, build it on top by passing a context.
var noGradDepth int32

// enableGradDepth is the symmetric counter for explicit EnableGrad scopes
// inside an outer NoGrad. While > 0, IsGradEnabled returns true regardless of
// noGradDepth — i.e. EnableGrad temporarily wins over NoGrad.
var enableGradDepth int32

// IsGradEnabled reports whether gradient computation is currently enabled.
func IsGradEnabled() bool {
	if atomic.LoadInt32(&enableGradDepth) > 0 {
		return true
	}
	return atomic.LoadInt32(&noGradDepth) == 0
}

// NoGrad runs fn with gradient computation disabled. Nested NoGrad calls are
// safe — including from multiple goroutines — because we count entries
// instead of overwriting a boolean.
//
//	autograd.NoGrad(func() {
//	    pred := model.Forward(x)   // no graph built
//	    fmt.Println(pred.Data())
//	})
func NoGrad(fn func()) {
	atomic.AddInt32(&noGradDepth, 1)
	defer atomic.AddInt32(&noGradDepth, -1)
	fn()
}

// EnableGrad runs fn with gradient computation enabled. Useful to re-enable
// grad inside an outer NoGrad block.
func EnableGrad(fn func()) {
	atomic.AddInt32(&enableGradDepth, 1)
	defer atomic.AddInt32(&enableGradDepth, -1)
	fn()
}
