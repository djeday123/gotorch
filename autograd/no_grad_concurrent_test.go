package autograd

import (
	"sync"
	"sync/atomic"
	"testing"
)

// TestNoGradConcurrentNesting exercises the previous broken behaviour:
// two goroutines entering NoGrad in interleaved order used to clobber each
// other's restore value, leaving one of them with grad accidentally
// re-enabled mid-scope. The counter-based implementation must report grad
// as disabled for both goroutines while either NoGrad block is active.
func TestNoGradConcurrentNesting(t *testing.T) {
	const iters = 200
	var wg sync.WaitGroup
	var leaks atomic.Int64

	check := func() {
		// While inside NoGrad, IsGradEnabled must be false.
		NoGrad(func() {
			for i := 0; i < 50; i++ {
				if IsGradEnabled() {
					leaks.Add(1)
				}
			}
		})
	}

	for i := 0; i < iters; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); check() }()
		go func() { defer wg.Done(); check() }()
	}
	wg.Wait()

	if leaks.Load() != 0 {
		t.Fatalf("IsGradEnabled returned true %d times inside NoGrad (must be 0)", leaks.Load())
	}
	if !IsGradEnabled() {
		t.Fatal("grad must be re-enabled after all NoGrad scopes exit")
	}
}

// TestEnableGradOverridesNoGrad verifies that an inner EnableGrad scope wins
// over an outer NoGrad scope, and that order is restored correctly on exit.
func TestEnableGradOverridesNoGrad(t *testing.T) {
	if !IsGradEnabled() {
		t.Fatal("grad must start enabled")
	}
	NoGrad(func() {
		if IsGradEnabled() {
			t.Fatal("grad must be disabled inside NoGrad")
		}
		EnableGrad(func() {
			if !IsGradEnabled() {
				t.Fatal("grad must be re-enabled by inner EnableGrad")
			}
			NoGrad(func() {
				// Inner NoGrad inside EnableGrad inside NoGrad:
				// EnableGrad still wins (it is the deepest enable).
				if !IsGradEnabled() {
					t.Fatal("EnableGrad must still dominate inner NoGrad")
				}
			})
		})
		if IsGradEnabled() {
			t.Fatal("grad must be disabled again after EnableGrad exits")
		}
	})
	if !IsGradEnabled() {
		t.Fatal("grad must be re-enabled at top level")
	}
}
