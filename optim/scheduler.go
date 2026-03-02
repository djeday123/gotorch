package optim

import "math"

// Optimizer is the minimal interface schedulers need to adjust LR.
type Optimizer interface {
	Step()
	ZeroGrad()
	SetLR(lr float64)
	GetLR() float64
}

// ---------------------------------------------------------------------------
// StepLR — multiply LR by gamma every step_size epochs
// ---------------------------------------------------------------------------

// StepLR decays the LR by gamma every stepSize steps.
//
//	sched := optim.NewStepLR(opt, 10, 0.1)
//	// every 10 steps: lr *= 0.1
type StepLR struct {
	opt      Optimizer
	stepSize int
	gamma    float64
	lastStep int
	baseLR   float64
}

func NewStepLR(opt Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{opt: opt, stepSize: stepSize, gamma: gamma, baseLR: opt.GetLR()}
}

func (s *StepLR) Step() {
	s.lastStep++
	if s.lastStep%s.stepSize == 0 {
		s.opt.SetLR(s.opt.GetLR() * s.gamma)
	}
}

func (s *StepLR) GetLR() float64 { return s.opt.GetLR() }

// ---------------------------------------------------------------------------
// CosineAnnealingLR — lr follows a cosine curve from base_lr to eta_min
// ---------------------------------------------------------------------------

// CosineAnnealingLR anneals the LR following a cosine curve over T_max steps.
//
//	sched := optim.NewCosineAnnealingLR(opt, 100, 1e-6)
type CosineAnnealingLR struct {
	opt    Optimizer
	tMax   int
	etaMin float64
	baseLR float64
	t      int
}

func NewCosineAnnealingLR(opt Optimizer, tMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{opt: opt, tMax: tMax, etaMin: etaMin, baseLR: opt.GetLR()}
}

func (c *CosineAnnealingLR) Step() {
	c.t++
	if c.t > c.tMax {
		c.t = c.tMax
	}
	lr := c.etaMin + (c.baseLR-c.etaMin)*(1+math.Cos(math.Pi*float64(c.t)/float64(c.tMax)))/2
	c.opt.SetLR(lr)
}

func (c *CosineAnnealingLR) GetLR() float64 { return c.opt.GetLR() }

// ---------------------------------------------------------------------------
// LinearWarmup — linearly ramp LR from 0 to base_lr over warmup_steps
// ---------------------------------------------------------------------------

// LinearWarmup linearly increases LR from 0 to base_lr over warmupSteps,
// then holds LR constant. Combine with another scheduler after warmup.
//
//	sched := optim.NewLinearWarmup(opt, 1000)
type LinearWarmup struct {
	opt         Optimizer
	warmupSteps int
	baseLR      float64
	t           int
}

func NewLinearWarmup(opt Optimizer, warmupSteps int) *LinearWarmup {
	baseLR := opt.GetLR()
	opt.SetLR(0)
	return &LinearWarmup{opt: opt, warmupSteps: warmupSteps, baseLR: baseLR}
}

func (w *LinearWarmup) Step() {
	w.t++
	if w.t <= w.warmupSteps {
		w.opt.SetLR(w.baseLR * float64(w.t) / float64(w.warmupSteps))
	}
}

func (w *LinearWarmup) GetLR() float64 { return w.opt.GetLR() }
