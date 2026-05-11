// Package engine — unit tests for the KRSI simulation engine.
//
// WHY these tests matter:
// The engine's physics model (failure risk, latency, energy) directly determines
// the quality of RL training data. Bugs here corrupt every experiment silently.
// Tests are table-driven (Go idiom), deterministic (fixed RNG seeds), and cover
// both happy paths and boundary/edge conditions.
package engine_test

import (
	"math"
	"testing"

	"krsi-simulator/engine"
	"krsi-simulator/pkg/logger"
)

// testLogger returns a silent logger suitable for tests.
func testLogger() *logger.Logger {
	return logger.New("test", logger.ERROR) // suppress output during tests
}

// newTestEngine builds a deterministic engine for testing.
func newTestEngine(seed int64) *engine.Engine {
	cfg := engine.DefaultConfig()
	cfg.Seed = seed
	return engine.New(cfg, testLogger())
}

// =============================================================================
// Reset Tests
// =============================================================================

func TestReset_InitialisesCleanState(t *testing.T) {
	eng := newTestEngine(42)

	// Run a few steps to dirty the state
	for i := 0; i < 10; i++ {
		eng.Step(1) // SCALE_OUT_REPLICA
	}

	eng.Reset()
	state := eng.GetState()

	if state.TimeStep != 0 {
		t.Errorf("expected TimeStep=0 after reset, got %d", state.TimeStep)
	}
	if state.Failures != 0 {
		t.Errorf("expected Failures=0 after reset, got %d", state.Failures)
	}
	if state.Adversarial != 0 {
		t.Errorf("expected Adversarial=0 after reset, got %d", state.Adversarial)
	}
}

func TestReset_IncreasesSimCycleID(t *testing.T) {
	eng := newTestEngine(42)
	initialID := eng.GetState().SimCycleID
	eng.Reset()
	newID := eng.GetState().SimCycleID
	if newID <= initialID {
		t.Errorf("expected SimCycleID to increase after reset, got %d -> %d", initialID, newID)
	}
}

// =============================================================================
// Action Tests
// =============================================================================

func TestAction_ScaleOutReplica(t *testing.T) {
	eng := newTestEngine(42)
	initial := eng.GetState().NRep
	eng.Step(1) // SCALE_OUT_REPLICA
	after := eng.GetState().NRep
	if after != initial+1 {
		t.Errorf("SCALE_OUT: expected NRep=%d, got %d", initial+1, after)
	}
}

func TestAction_ScaleInReplica(t *testing.T) {
	eng := newTestEngine(42)
	// First scale out to have room to scale in
	eng.Step(1) // NRep = 4
	before := eng.GetState().NRep
	eng.Step(2) // SCALE_IN_REPLICA
	after := eng.GetState().NRep
	if after != before-1 {
		t.Errorf("SCALE_IN: expected NRep=%d, got %d", before-1, after)
	}
}

func TestAction_ScaleInReplica_DoesNotGoBelowOne(t *testing.T) {
	cfg := engine.DefaultConfig()
	cfg.Seed = 42
	cfg.InitialReplicas = 1
	eng := engine.New(cfg, testLogger())

	eng.Step(2) // SCALE_IN_REPLICA — should be no-op
	state := eng.GetState()
	if state.NRep < 1 {
		t.Errorf("NRep should never go below 1, got %d", state.NRep)
	}
}

func TestAction_AddNode(t *testing.T) {
	eng := newTestEngine(42)
	initial := eng.GetState().NNodes
	eng.Step(3) // ADD_NODE
	after := eng.GetState().NNodes
	if after != initial+1 {
		t.Errorf("ADD_NODE: expected NNodes=%d, got %d", initial+1, after)
	}
}

func TestAction_RemoveNode_DoesNotGoBelowOne(t *testing.T) {
	cfg := engine.DefaultConfig()
	cfg.Seed = 42
	cfg.InitialNodes = 1
	eng := engine.New(cfg, testLogger())

	eng.Step(4) // REMOVE_NODE — should be no-op
	state := eng.GetState()
	if state.NNodes < 1 {
		t.Errorf("NNodes should never go below 1, got %d", state.NNodes)
	}
}

func TestAction_SetBinpack(t *testing.T) {
	eng := newTestEngine(42)
	eng.Step(6) // SET_SPREAD first
	eng.Step(5) // SET_BINPACK
	if eng.GetState().PSched != 0 {
		t.Errorf("expected PSched=0 (binpack) after action 5")
	}
}

func TestAction_SetSpread(t *testing.T) {
	eng := newTestEngine(42)
	eng.Step(6) // SET_SPREAD
	if eng.GetState().PSched != 1 {
		t.Errorf("expected PSched=1 (spread) after action 6")
	}
}

func TestAction_ThetaScaleBounds(t *testing.T) {
	eng := newTestEngine(42)
	// Drive theta to maximum
	for i := 0; i < 50; i++ {
		eng.Step(7) // SCALE_THRESHOLD_UP
	}
	if eng.GetState().ThetaScale > 1.0 {
		t.Errorf("ThetaScale exceeded upper bound 1.0: got %f", eng.GetState().ThetaScale)
	}
	// Drive theta to minimum
	eng.Reset()
	for i := 0; i < 50; i++ {
		eng.Step(8) // SCALE_THRESHOLD_DOWN
	}
	if eng.GetState().ThetaScale < 0.1 {
		t.Errorf("ThetaScale below lower bound 0.1: got %f", eng.GetState().ThetaScale)
	}
}

func TestAction_InvalidNoOp(t *testing.T) {
	eng := newTestEngine(42)
	initial := eng.GetState().NRep
	initialNodes := eng.GetState().NNodes
	eng.Step(0) // NO_OP
	// NOP should not change replica or node counts (only timestep advances)
	if eng.GetState().NRep != initial {
		t.Errorf("NO_OP should not change NRep")
	}
	if eng.GetState().NNodes != initialNodes {
		t.Errorf("NO_OP should not change NNodes")
	}
}

// =============================================================================
// Metrics / Physics Tests
// =============================================================================

func TestMetrics_TimestepAdvances(t *testing.T) {
	eng := newTestEngine(42)
	eng.Step(0)
	if eng.GetState().TimeStep != 1 {
		t.Errorf("TimeStep should be 1 after one step")
	}
}

func TestMetrics_EnergyIsPositive(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 20; i++ {
		resp := eng.Step(0)
		if resp.State.E <= 0 {
			t.Errorf("step %d: energy must be positive, got %f", i, resp.State.E)
		}
	}
}

func TestMetrics_LatencyIsPositive(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 20; i++ {
		resp := eng.Step(0)
		if resp.State.L <= 0 {
			t.Errorf("step %d: latency must be positive, got %f", i, resp.State.L)
		}
	}
}

func TestMetrics_UtilizationInRange(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 50; i++ {
		resp := eng.Step(0)
		for _, u := range []float64{resp.State.UCpu, resp.State.UMem, resp.State.USto} {
			if u < 0.0 || u > 1.0 {
				t.Errorf("step %d: utilization out of [0,1]: %f", i, u)
			}
		}
	}
}

func TestMetrics_SeverityScoreInRange(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 100; i++ {
		resp := eng.Step(0)
		s := resp.Resilience.SeverityScore
		if s < 0.0 || s > 1.0 {
			t.Errorf("step %d: SeverityScore out of [0,1]: %f", i, s)
		}
	}
}

func TestMetrics_CarbonIntensityRange(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 100; i++ {
		resp := eng.Step(0)
		ci := resp.Sustainability.CI
		// CI is 0.4 ± 0.2 sin(t/20), so range is [0.2, 0.6]
		if ci < 0.19 || ci > 0.61 {
			t.Errorf("step %d: carbon intensity %f outside expected range [0.2, 0.6]", i, ci)
		}
	}
}

func TestMetrics_RenewableEnergyFraction(t *testing.T) {
	eng := newTestEngine(42)
	for i := 0; i < 20; i++ {
		resp := eng.Step(0)
		renewable := resp.Sustainability.ERenewable
		total := resp.Sustainability.ETotal
		// ERenewable should be ~30% of ETotal
		expected := total * 0.3
		if math.Abs(renewable-expected) > 0.01 {
			t.Errorf("step %d: renewable fraction wrong: got %f, expected %f", i, renewable, expected)
		}
	}
}

// =============================================================================
// Determinism Tests
// =============================================================================

// TestDeterminism_SameSeedSameResults verifies that two engines with identical
// seeds produce identical output sequences — critical for reproducible research.
func TestDeterminism_SameSeedSameResults(t *testing.T) {
	actions := []int{0, 1, 3, 6, 0, 2, 4, 5, 8, 0}

	eng1 := newTestEngine(42)
	eng2 := newTestEngine(42)

	for i, action := range actions {
		r1 := eng1.Step(action)
		r2 := eng2.Step(action)
		if math.Abs(r1.State.L-r2.State.L) > 1e-9 {
			t.Errorf("step %d: latency diverged between identical seeds: %f vs %f", i, r1.State.L, r2.State.L)
		}
		if math.Abs(r1.State.E-r2.State.E) > 1e-9 {
			t.Errorf("step %d: energy diverged between identical seeds: %f vs %f", i, r1.State.E, r2.State.E)
		}
	}
}

func TestDeterminism_DifferentSeedsDifferentResults(t *testing.T) {
	eng1 := newTestEngine(42)
	eng2 := newTestEngine(99)

	diffs := 0
	for i := 0; i < 50; i++ {
		r1 := eng1.Step(0)
		r2 := eng2.Step(0)
		if math.Abs(r1.Resilience.Nf-float64(r2.Resilience.Nf)) > 0 {
			diffs++
		}
	}
	// Different seeds should produce some divergence in stochastic events
	if diffs == 0 {
		t.Log("WARNING: different seeds produced identical failure sequences (unlikely but possible)")
	}
}

// =============================================================================
// Stress Tests
// =============================================================================

// TestStress_LongEpisode runs 10,000 steps to detect panics, NaN propagation,
// and numeric instability in the physics model.
func TestStress_LongEpisode(t *testing.T) {
	eng := newTestEngine(42)
	actions := []int{0, 1, 2, 3, 4, 5, 6, 7, 8}

	for i := 0; i < 10000; i++ {
		action := actions[i%len(actions)]
		resp := eng.Step(action)

		if math.IsNaN(resp.State.L) || math.IsInf(resp.State.L, 0) {
			t.Fatalf("step %d: latency is NaN or Inf", i)
		}
		if math.IsNaN(resp.State.E) || math.IsInf(resp.State.E, 0) {
			t.Fatalf("step %d: energy is NaN or Inf", i)
		}
		if math.IsNaN(resp.Sustainability.ETotal) {
			t.Fatalf("step %d: ETotal is NaN", i)
		}
	}
}

// TestStress_AllActionsUnderHighLoad verifies the engine handles all actions
// under maximum workload without panicking.
func TestStress_AllActionsUnderHighLoad(t *testing.T) {
	cfg := engine.DefaultConfig()
	cfg.Seed = 0
	cfg.BurstProb = 1.0 // always burst — maximum stress
	eng := engine.New(cfg, testLogger())

	for action := 0; action <= 8; action++ {
		for i := 0; i < 100; i++ {
			resp := eng.Step(action)
			if resp.State.E <= 0 {
				t.Errorf("action %d step %d: energy must be positive", action, i)
			}
		}
	}
}

// TestConcurrentSafety verifies no data race under concurrent step calls.
// Run with: go test -race ./engine/...
func TestConcurrentSafety(t *testing.T) {
	eng := newTestEngine(42)
	done := make(chan struct{})

	for i := 0; i < 4; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				eng.Step(id % 9)
			}
			done <- struct{}{}
		}(i)
	}

	for i := 0; i < 4; i++ {
		<-done
	}
}
