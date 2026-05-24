package tensor

import (
	"math/bits"
	"sync"
)

// ─────────────────────────────────────────────────────────────────────────────
// Slice pool for tensor storage
//
// In long training loops the dominant GC pressure comes from short-lived
// []float64 / []float32 backing storage produced by tensor ops. We bucket
// allocations by next-power-of-two size and stash freed slices in sync.Pool
// to be picked up by the next request of the same bucket.
//
// Usage:
//
//	// Replace `make([]float64, n)` in hot paths:
//	buf := tensor.AllocFloat64(n)   // length n, capped to bucket size
//	defer tensor.FreeFloat64(buf)   // return to pool when done
//
//	// On a Tensor created via Zeros/Ones/Rand/... that owns its storage:
//	t := tensor.Zeros(1024, 1024)
//	// ... use t ...
//	t.Release()                       // give the storage back to the pool
//
// After Release(), the tensor MUST NOT be used. Views/transposes of a
// released tensor read garbage. This is an explicit opt-in fast-path —
// callers are responsible for lifetime tracking. Tensors that are NOT
// released are simply GC'd as before.
// ─────────────────────────────────────────────────────────────────────────────

const (
	minBucketLog2 = 6  // 64 elements = 512 bytes (Float64)
	maxBucketLog2 = 24 // 16M elements = 128 MB; bigger allocs skip the pool
)

var (
	f64Pools [maxBucketLog2 + 1]sync.Pool
	f32Pools [maxBucketLog2 + 1]sync.Pool
)

func init() {
	for i := minBucketLog2; i <= maxBucketLog2; i++ {
		size := 1 << i
		f64Pools[i].New = func() interface{} {
			s := make([]float64, size)
			return &s
		}
		f32Pools[i].New = func() interface{} {
			s := make([]float32, size)
			return &s
		}
	}
}

// bucketOf returns the pool bucket for length n, or -1 if n is outside the
// pooled range. n is the requested length; we return a bucket whose
// capacity is at least n.
func bucketOf(n int) int {
	if n <= 0 {
		return -1
	}
	if n <= (1 << minBucketLog2) {
		return minBucketLog2
	}
	// Ceil log2 of n.
	b := bits.Len(uint(n - 1))
	if b > maxBucketLog2 {
		return -1
	}
	return b
}

// AllocFloat64 returns a []float64 of length exactly n. The underlying
// array is drawn from a pool when possible; the returned slice is zeroed.
func AllocFloat64(n int) []float64 {
	b := bucketOf(n)
	if b < 0 {
		return make([]float64, n)
	}
	sp := f64Pools[b].Get().(*[]float64)
	s := (*sp)[:n]
	for i := range s {
		s[i] = 0
	}
	return s
}

// FreeFloat64 returns a slice (previously obtained from AllocFloat64 or any
// slice whose cap is a power of two within the pool range) to the pool.
// The caller must not use the slice afterwards.
func FreeFloat64(s []float64) {
	c := cap(s)
	if c < (1<<minBucketLog2) || c > (1<<maxBucketLog2) {
		return
	}
	if c&(c-1) != 0 {
		return // capacity is not a power of two — not from our pool
	}
	b := bits.TrailingZeros(uint(c))
	full := s[:c]
	f64Pools[b].Put(&full)
}

// AllocFloat32 / FreeFloat32 — same pattern for the Float32 dtype path.
func AllocFloat32(n int) []float32 {
	b := bucketOf(n)
	if b < 0 {
		return make([]float32, n)
	}
	sp := f32Pools[b].Get().(*[]float32)
	s := (*sp)[:n]
	for i := range s {
		s[i] = 0
	}
	return s
}

func FreeFloat32(s []float32) {
	c := cap(s)
	if c < (1<<minBucketLog2) || c > (1<<maxBucketLog2) {
		return
	}
	if c&(c-1) != 0 {
		return
	}
	b := bits.TrailingZeros(uint(c))
	full := s[:c]
	f32Pools[b].Put(&full)
}

// Release returns the tensor's backing storage to the pool. Caller MUST NOT
// use the tensor or any view of it after Release.
//
// No-op for tensors that are views/non-contiguous (they don't own the full
// storage) and for scalar tensors below the pool minimum bucket size.
func (t *Tensor) Release() {
	if t == nil {
		return
	}
	if !t.isContiguous() {
		// View into shared storage — releasing would corrupt the parent.
		// Caller should release the parent tensor instead.
		return
	}
	if t.dtype == Float32 {
		FreeFloat32(t.f32)
		t.f32 = nil
	} else {
		FreeFloat64(t.data)
		t.data = nil
	}
}
