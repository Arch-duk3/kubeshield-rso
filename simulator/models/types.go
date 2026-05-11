// Package models defines all domain types for the KRSI simulator.
// These are the shared data structures used across the API, engine, and test layers.
package models

// StepRequest is the inbound action from the RL controller.
type StepRequest struct {
	Action int `json:"action"`
}

// MetricsResponse is the simulator's complete outbound payload per step.
// It includes sustainability, resilience, and raw state data — all consumed
// by the Python KRSICalculator to derive S, R, and KRSI.
type MetricsResponse struct {
	Sustainability SustainabilityMetrics `json:"sustainability"`
	Resilience     ResilienceMetrics     `json:"resilience"`
	State          StateMetrics          `json:"state"`
	SimCycleID     int64                 `json:"sim_cycle_id"`
}

// SustainabilityMetrics captures energy, utilization, and carbon intensity signals.
type SustainabilityMetrics struct {
	ETotal     float64 `json:"e_total"`     // Total power draw (W)
	W          float64 `json:"w"`           // Useful work processed (req/s)
	CI         float64 `json:"ci"`          // Carbon intensity (gCO2/kWh)
	ERenewable float64 `json:"e_renewable"` // Renewable energy fraction (W)
	UCpu       float64 `json:"u_cpu"`       // CPU utilization [0,1]
	UMem       float64 `json:"u_mem"`       // Memory utilization [0,1]
	USto       float64 `json:"u_sto"`       // Storage utilization [0,1]
	UTarget    float64 `json:"u_target"`    // Target utilization setpoint (theta_scale)
}

// ResilienceMetrics captures all resilience sub-domain signals.
type ResilienceMetrics struct {
	TUp           float64 `json:"t_up"`           // Total uptime units
	TObs          float64 `json:"t_obs"`          // Total observed time units
	Nf            int     `json:"n_f"`            // Active failure count
	TRec          float64 `json:"t_rec"`          // Current recovery time estimate
	TRecMax       float64 `json:"t_rec_max"`      // Maximum acceptable recovery time
	PMin          float64 `json:"p_min"`          // Minimum performance during disruption
	PBaseline     float64 `json:"p_baseline"`     // Baseline performance
	PPost         float64 `json:"p_post"`         // Post-recovery performance
	WDisruption   float64 `json:"w_disruption"`   // Workload shed during disruption
	WBaseline     float64 `json:"w_baseline"`     // Baseline workload
	TDet          float64 `json:"t_det"`          // Attack detection time
	TDetMax       float64 `json:"t_det_max"`      // Maximum detection time threshold
	TSec          float64 `json:"t_sec"`          // Time to remediate security event
	TSecMax       float64 `json:"t_sec_max"`      // Max acceptable remediation time
	NAffected     int     `json:"n_affected"`     // Nodes affected by adversarial events
	NTotal        int     `json:"n_total"`        // Total node count
	SeverityScore float64 `json:"severity_score"` // Normalized attack severity [0,1]
	OConf         float64 `json:"o_conf"`         // Operational confidence factor
}

// StateMetrics is the raw observable state vector surfaced to the RL agent.
type StateMetrics struct {
	UCpu       float64 `json:"u_cpu"`
	UMem       float64 `json:"u_mem"`
	USto       float64 `json:"u_sto"`
	E          float64 `json:"e"`
	L          float64 `json:"l"`           // Latency (ms)
	F          int     `json:"f"`           // Active failure count
	NRep       int     `json:"n_rep"`       // Replica count
	NNodes     int     `json:"n_nodes"`     // Node count
	PSched     int     `json:"p_sched"`     // Scheduler policy: 0=binpack, 1=spread
	ThetaScale float64 `json:"theta_scale"` // Utilization target setpoint
}

// SimulatorState is the mutable internal state of the simulation engine.
// It is NOT exposed directly via the API — only StateMetrics is surfaced.
type SimulatorState struct {
	TimeStep    int
	NRep        int
	NNodes      int
	PSched      int
	ThetaScale  float64
	Workload    float64
	Failures    int
	Adversarial int
	TotalTime   float64
	Downtime    float64
	SimCycleID  int64
}

// ActionMap maps action integer codes to human-readable descriptions.
// Used in structured logs and observability tooling.
var ActionMap = map[int]string{
	0: "NO_OP",
	1: "SCALE_OUT_REPLICA",
	2: "SCALE_IN_REPLICA",
	3: "ADD_NODE",
	4: "REMOVE_NODE",
	5: "SET_BINPACK",
	6: "SET_SPREAD",
	7: "SCALE_THRESHOLD_UP",
	8: "SCALE_THRESHOLD_DOWN",
}
