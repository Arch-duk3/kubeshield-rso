// Package engine implements the KRSI Kubernetes simulation engine.
// It models workload dynamics, failure injection, adversarial events,
// energy consumption, and scheduling policy effects.
//
// WHY a separate package: the original main.go coupled HTTP handling with
// simulation logic, making it impossible to unit-test the physics model
// independently. Separating into engine/ allows pure table-driven Go tests
// without spinning up an HTTP server.
package engine

import (
	"math"
	"math/rand"
	"sync"

	"krsi-simulator/models"
	"krsi-simulator/pkg/logger"
)

// Config holds all tunable simulation parameters.
// These are loaded from the YAML config to eliminate hardcoded constants.
type Config struct {
	Seed                    int64
	InitialReplicas         int
	InitialNodes            int
	InitialThetaScale       float64
	BaseRisk                float64
	AlphaRisk               float64
	BetaRisk                float64
	AdversarialProb         float64
	AdversarialRecoveryProb float64
	BurstProb               float64
	BurstMagnitude          float64
}

// DefaultConfig returns production-calibrated defaults matching the paper spec.
func DefaultConfig() Config {
	return Config{
		Seed:                    42,
		InitialReplicas:         3,
		InitialNodes:            3,
		InitialThetaScale:       0.8,
		BaseRisk:                0.01,
		AlphaRisk:               0.05,
		BetaRisk:                0.02,
		AdversarialProb:         0.02,
		AdversarialRecoveryProb: 0.10,
		BurstProb:               0.10,
		BurstMagnitude:          200.0,
	}
}

// Engine is the stateful simulation engine.
// It is thread-safe via an internal mutex, allowing future parallel episode
// execution or multi-threaded benchmark scenarios.
type Engine struct {
	mu    sync.Mutex
	state models.SimulatorState
	cfg   Config
	rng   *rand.Rand
	log   *logger.Logger
}

// New creates and initialises a new simulation Engine.
func New(cfg Config, log *logger.Logger) *Engine {
	e := &Engine{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
		log: log,
	}
	e.reset()
	return e
}

// reset initialises the engine state to the beginning of an episode.
func (e *Engine) reset() {
	e.state = models.SimulatorState{
		TimeStep:    0,
		NRep:        e.cfg.InitialReplicas,
		NNodes:      e.cfg.InitialNodes,
		PSched:      0,
		ThetaScale:  e.cfg.InitialThetaScale,
		Workload:    100.0,
		Failures:    0,
		Adversarial: 0,
		TotalTime:   0.0,
		Downtime:    0.0,
		SimCycleID:  e.state.SimCycleID + 1,
	}
}

// Reset resets the simulation state and seeds the RNG for reproducibility.
// WHY reseed on reset: each episode must start with a deterministic RNG state
// so that comparing two policies across the same seed yields valid comparisons.
func (e *Engine) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.rng = rand.New(rand.NewSource(e.cfg.Seed + e.state.SimCycleID))
	e.reset()

	e.log.Info(logger.EventSimReset, "Simulation episode reset", e.state.SimCycleID, 0, map[string]interface{}{
		"initial_replicas": e.cfg.InitialReplicas,
		"initial_nodes":    e.cfg.InitialNodes,
		"seed":             e.cfg.Seed + e.state.SimCycleID,
	})
}

// Step advances the simulation by one timestep, applying the given action.
// It returns the complete metrics payload consumed by the Python controller.
func (e *Engine) Step(action int) models.MetricsResponse {
	e.mu.Lock()
	defer e.mu.Unlock()

	// 1. Apply action and mutate state
	e.applyAction(action)

	e.state.TimeStep++

	// 2. Inject stochastic workload
	wasBurst := e.injectWorkload()

	// 3. Compute failure dynamics
	newFailures := e.computeFailures()

	// 4. Compute metrics from state
	resp := e.computeMetrics(newFailures)

	// 5. Emit structured step log
	actionName, _ := models.ActionMap[action]
	data := map[string]interface{}{
		"action":      action,
		"action_name": actionName,
		"n_rep":       e.state.NRep,
		"n_nodes":     e.state.NNodes,
		"p_sched":     e.state.PSched,
		"failures":    e.state.Failures,
		"adversarial": e.state.Adversarial,
		"workload":    e.state.Workload,
		"latency_ms":  resp.State.L,
		"energy_w":    resp.State.E,
		"was_burst":   wasBurst,
	}
	e.log.Debug(logger.EventSimStep, "Simulation step completed", e.state.SimCycleID, e.state.TimeStep, data)

	return resp
}

// applyAction maps the integer action code to state mutations.
// WHY this separation: isolating mutation makes the action logic
// independently testable and auditable for correctness.
func (e *Engine) applyAction(action int) {
	prevSched := e.state.PSched
	prevNNodes := e.state.NNodes
	prevNRep := e.state.NRep

	switch action {
	case 1:
		e.state.NRep++
		e.log.Info(logger.EventReplicaScaled, "Replica scaled out", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"from": prevNRep, "to": e.state.NRep})
	case 2:
		if e.state.NRep > 1 {
			e.state.NRep--
			e.log.Info(logger.EventReplicaScaled, "Replica scaled in", e.state.SimCycleID, e.state.TimeStep,
				map[string]interface{}{"from": prevNRep, "to": e.state.NRep})
		}
	case 3:
		e.state.NNodes++
		e.log.Info(logger.EventNodeAdded, "Node added", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"from": prevNNodes, "to": e.state.NNodes})
	case 4:
		if e.state.NNodes > 1 {
			e.state.NNodes--
			e.log.Info(logger.EventNodeRemoved, "Node removed", e.state.SimCycleID, e.state.TimeStep,
				map[string]interface{}{"from": prevNNodes, "to": e.state.NNodes})
		}
	case 5:
		e.state.PSched = 0
		if prevSched != 0 {
			e.log.Info(logger.EventSchedulerChange, "Scheduler changed to BINPACK", e.state.SimCycleID, e.state.TimeStep, nil)
		}
	case 6:
		e.state.PSched = 1
		if prevSched != 1 {
			e.log.Info(logger.EventSchedulerChange, "Scheduler changed to SPREAD", e.state.SimCycleID, e.state.TimeStep, nil)
		}
	case 7:
		e.state.ThetaScale = math.Min(1.0, e.state.ThetaScale+0.05)
	case 8:
		e.state.ThetaScale = math.Max(0.1, e.state.ThetaScale-0.05)
	}
}

// injectWorkload models sinusoidal workload with stochastic burst events.
// WHY sinusoidal + burst: mirrors real production traffic patterns (diurnal cycles
// + flash crowds) better than uniform random, producing more ecologically valid RL policies.
func (e *Engine) injectWorkload() bool {
	base := 100.0 + 50.0*math.Sin(float64(e.state.TimeStep)/10.0)
	wasBurst := false
	if e.rng.Float64() < e.cfg.BurstProb {
		base += e.cfg.BurstMagnitude
		wasBurst = true
		e.log.Warn(logger.EventWorkloadBurst, "Workload burst injected", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"burst_magnitude": e.cfg.BurstMagnitude, "total_workload": base})
	}
	e.state.Workload = base
	return wasBurst
}

// computeFailures models stochastic failures with scheduling policy effects
// and cascading recovery dynamics.
// WHY coupling failure risk to replica density: dense scheduling (binpack)
// increases blast radius when a node fails, while spread scheduling isolates faults.
func (e *Engine) computeFailures() int {
	reqsPerRep := e.state.Workload / float64(e.state.NRep)
	replicaDensity := float64(e.state.NRep) / float64(e.state.NNodes)

	failureRisk := e.cfg.BaseRisk +
		e.cfg.AlphaRisk*(reqsPerRep/50.0) +
		e.cfg.BetaRisk*replicaDensity

	// Scheduling policy modifies failure risk (key insight from the KRSI paper)
	if e.state.PSched == 0 { // binpack: higher utilization, higher blast radius
		reqsPerRep *= 1.3
		failureRisk *= 1.5
	} else { // spread: distributes load, reduces correlated failure risk
		reqsPerRep *= 0.8
		failureRisk *= 0.5
	}
	_ = reqsPerRep // consumed by scheduling effect above

	newFailures := 0
	if e.rng.Float64() < failureRisk {
		e.state.Failures++
		newFailures++
		e.log.Warn(logger.EventFailureInjected, "Pod failure injected", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{
				"total_failures": e.state.Failures,
				"failure_risk":   failureRisk,
				"scheduling":     e.state.PSched,
			})
	}

	// Adversarial events (e.g., privilege escalation, lateral movement)
	if e.rng.Float64() < e.cfg.AdversarialProb {
		e.state.Adversarial++
		e.log.Warn(logger.EventAdversarialEvent, "Adversarial event injected", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"total_adversarial": e.state.Adversarial})
	}

	// Recovery dynamics: more replicas per node accelerates recovery
	recoveryProb := 0.2 + 0.05*(float64(e.state.NRep)/float64(e.state.NNodes))
	if e.state.Failures > 0 && e.rng.Float64() < recoveryProb {
		e.state.Failures--
		e.log.Info(logger.EventFailureRecovered, "Failure recovered", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"remaining_failures": e.state.Failures})
	}
	if e.state.Adversarial > 0 && e.rng.Float64() < e.cfg.AdversarialRecoveryProb {
		e.state.Adversarial--
	}

	return newFailures
}

// computeMetrics derives all observable metrics from the current simulation state.
// These are the values sent to the Python controller and logged per step.
func (e *Engine) computeMetrics(newFailures int) models.MetricsResponse {
	reqsPerRep := e.state.Workload / float64(e.state.NRep)
	if e.state.PSched == 0 {
		reqsPerRep *= 1.3
	} else {
		reqsPerRep *= 0.8
	}

	capacity := float64(e.state.NNodes) * 100.0
	uCPU := math.Min(1.0, reqsPerRep/50.0)
	uMem := uCPU * 0.8
	uSto := 0.5

	// Exponential latency model: latency spikes sharply above capacity
	baseLatency := 10.0 + 50.0*math.Exp(math.Max(0.0, (reqsPerRep/50.0)-1.0))
	alpha := 0.2
	latency := baseLatency * (1.0 + (math.Exp(alpha*float64(e.state.Failures))-1.0)/(1.0+float64(e.state.Failures)))

	// Throughput degrades under failures
	W := math.Min(e.state.Workload, capacity)
	W = W * math.Max(0.1, 1.0-0.15*float64(e.state.Failures))

	// Recovery time is proportional to active failure count
	tRec := 5.0 + float64(e.state.Failures)*2.0

	// Uptime accounting
	e.state.TotalTime += 100.0
	if newFailures > 0 {
		e.state.Downtime += tRec * float64(newFailures)
	}
	tUp := math.Max(0.0, e.state.TotalTime-e.state.Downtime)

	// Linear energy model: PUE-aware power draw
	pIdle := 50.0 * float64(e.state.NNodes)
	pMax := 200.0 * float64(e.state.NNodes)
	power := pIdle + (pMax-pIdle)*uCPU

	// Carbon intensity: sinusoidal to model renewable energy availability variation
	ci := 0.4 + 0.2*math.Sin(float64(e.state.TimeStep)/20.0)

	// Log SLA violations
	if latency > 50.0 {
		e.log.Warn(logger.EventSLAViolation, "SLA latency threshold exceeded", e.state.SimCycleID, e.state.TimeStep,
			map[string]interface{}{"latency_ms": latency, "threshold_ms": 50.0})
	}

	return models.MetricsResponse{
		SimCycleID: e.state.SimCycleID,
		Sustainability: models.SustainabilityMetrics{
			ETotal:     power,
			W:          W,
			CI:         ci,
			ERenewable: power * 0.3,
			UCpu:       uCPU,
			UMem:       uMem,
			USto:       uSto,
			UTarget:    e.state.ThetaScale,
		},
		Resilience: models.ResilienceMetrics{
			TUp:           tUp,
			TObs:          math.Max(e.state.TotalTime, 1.0),
			Nf:            e.state.Failures,
			TRec:          tRec,
			TRecMax:       20.0,
			PMin:          math.Min(1.0, capacity/math.Max(1.0, e.state.Workload)),
			PBaseline:     1.0,
			PPost:         0.9,
			WDisruption:   math.Max(0, e.state.Workload-W),
			WBaseline:     e.state.Workload,
			TDet:          float64(e.state.Adversarial) * 5.0,
			TDetMax:       20.0,
			TSec:          float64(e.state.Adversarial) * 10.0,
			TSecMax:       40.0,
			NAffected:     e.state.Adversarial,
			NTotal:        e.state.NNodes,
			SeverityScore: math.Min(1.0, 0.2*float64(e.state.Adversarial)),
			OConf:         0.95,
		},
		State: models.StateMetrics{
			UCpu:       uCPU,
			UMem:       uMem,
			USto:       uSto,
			E:          power,
			L:          latency,
			F:          e.state.Failures,
			NRep:       e.state.NRep,
			NNodes:     e.state.NNodes,
			PSched:     e.state.PSched,
			ThetaScale: e.state.ThetaScale,
		},
	}
}

// GetState returns a snapshot of the current simulation state (for testing).
func (e *Engine) GetState() models.SimulatorState {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.state
}
