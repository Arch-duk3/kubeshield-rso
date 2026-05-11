package main

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sync"
	//"time"
)

// Task 7: Structural Improvements
type Controller interface {
	Decide(state StateMetrics) int
}

type State struct {
	sync.Mutex
	TimeStep    int
	NRep        int
	NNodes      int
	PSched      int
	ThetaScale  float64
	Workload    float64
	Failures    int
	Adversarial int

	UCpu float64
	UMem float64
	USto float64
	E    float64
	L    float64 // latency
	F    int     // current failures
	
	TotalTime float64 // Task 4: Uptime Model
	Downtime  float64 // Task 4: Uptime Model
}

type StepRequest struct {
	Action int `json:"action"`
}

type MetricsResponse struct {
	Sustainability SustainabilityMetrics `json:"sustainability"`
	Resilience     ResilienceMetrics     `json:"resilience"`
	State          StateMetrics          `json:"state"`
}

type SustainabilityMetrics struct {
	ETotal     float64 `json:"e_total"`
	W          float64 `json:"w"`
	CI         float64 `json:"ci"`
	ERenewable float64 `json:"e_renewable"`
	UCpu       float64 `json:"u_cpu"`
	UMem       float64 `json:"u_mem"`
	USto       float64 `json:"u_sto"`
	UTarget    float64 `json:"u_target"`
}

type ResilienceMetrics struct {
	TUp           float64 `json:"t_up"`
	TObs          float64 `json:"t_obs"`
	Nf            int     `json:"n_f"`
	TRec          float64 `json:"t_rec"`
	TRecMax       float64 `json:"t_rec_max"`
	PMin          float64 `json:"p_min"`
	PBaseline     float64 `json:"p_baseline"`
	PPost         float64 `json:"p_post"`
	WDisruption   float64 `json:"w_disruption"`
	WBaseline     float64 `json:"w_baseline"`
	TDet          float64 `json:"t_det"`
	TDetMax       float64 `json:"t_det_max"`
	TSec          float64 `json:"t_sec"`
	TSecMax       float64 `json:"t_sec_max"`
	NAffected     int     `json:"n_affected"`
	NTotal        int     `json:"n_total"`
	SeverityScore float64 `json:"severity_score"`
	OConf         float64 `json:"o_conf"`
}

type StateMetrics struct {
	UCpu       float64 `json:"u_cpu"`
	UMem       float64 `json:"u_mem"`
	USto       float64 `json:"u_sto"`
	E          float64 `json:"e"`
	L          float64 `json:"l"`
	F          int     `json:"f"`
	NRep       int     `json:"n_rep"`
	NNodes     int     `json:"n_nodes"`
	PSched     int     `json:"p_sched"`
	ThetaScale float64 `json:"theta_scale"`
}

var simState *State

func init() {
	simState = &State{
		NRep:       3,
		NNodes:     3,
		PSched:     0,
		ThetaScale: 0.8,
	}
	// Task 5: Add reproducibility
	rand.Seed(42)
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	simState.Lock()
	defer simState.Unlock()
	simState.TimeStep = 0
	simState.NRep = 3
	simState.NNodes = 3
	simState.PSched = 0
	simState.ThetaScale = 0.8
	simState.Workload = 100.0
	simState.Failures = 0
	simState.Adversarial = 0
	simState.TotalTime = 0.0
	simState.Downtime = 0.0
	w.WriteHeader(http.StatusOK)
}

func handleStep(w http.ResponseWriter, r *http.Request) {
	var req StepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	simState.Lock()
	defer simState.Unlock()

	// 1. Apply Action
	switch req.Action {
	case 1:
		simState.NRep++
	case 2:
		if simState.NRep > 1 {
			simState.NRep--
		}
	case 3:
		simState.NNodes++
	case 4:
		if simState.NNodes > 1 {
			simState.NNodes--
		}
	case 5:
		simState.PSched = 0
	case 6:
		simState.PSched = 1
	case 7:
		simState.ThetaScale += 0.05
	case 8:
		simState.ThetaScale -= 0.05
	}
	// constrain
	if simState.ThetaScale > 1.0 {
		simState.ThetaScale = 1.0
	}
	if simState.ThetaScale < 0.1 {
		simState.ThetaScale = 0.1
	}

	simState.TimeStep++

	// 2. Inject Workload (bursty/mixed)
	baseWorkload := 100.0 + 50.0*math.Sin(float64(simState.TimeStep)/10.0)
	if rand.Float64() < 0.1 { // burst
		baseWorkload += 200.0
	}
	simState.Workload = baseWorkload

	// 3. Setup failures context & load based on scheduling
	capacity := float64(simState.NNodes) * 100.0
	reqsPerRep := simState.Workload / float64(simState.NRep)

	// Task 5: Add failure cascade effect
	baseRisk := 0.01
	alphaRisk := 0.05
	betaRisk := 0.02
	replica_density := float64(simState.NRep) / float64(simState.NNodes)
	failureRisk := baseRisk + alphaRisk*(reqsPerRep/50.0) + betaRisk*replica_density

	// Task 5: Scheduling effect stronger
	if simState.PSched == 0 { // binpack -> higher utilization, higher failure risk
		reqsPerRep *= 1.3
		failureRisk *= 1.5
	} else { // spread -> lower utilization, more resilience
		reqsPerRep *= 0.8
		failureRisk *= 0.5
	}

	newFailures := 0
	if rand.Float64() < failureRisk {
		simState.Failures++
		newFailures++
	}
	if rand.Float64() < 0.02 {
		simState.Adversarial++
	}
	// recovery
	recoveryProb := 0.2 + 0.05*(float64(simState.NRep)/float64(simState.NNodes))
	if simState.Failures > 0 && rand.Float64() < recoveryProb {
		simState.Failures--
	}
	if simState.Adversarial > 0 && rand.Float64() < 0.1 {
		simState.Adversarial--
	}

	// 4. Compute Metrics
	simState.UCpu = math.Min(1.0, reqsPerRep/50.0)
	simState.UMem = simState.UCpu * 0.8
	simState.USto = 0.5

	// Task 5: Failures affect performance
	baseLatency := 10.0 + 50.0*math.Exp(math.Max(0.0, (reqsPerRep/50.0)-1.0))
	alpha := 0.2
	L := baseLatency * (1.0 + (math.Exp(alpha*float64(simState.Failures))-1.0)/(1.0+float64(simState.Failures)))

	// Throughput (Work processed) decreases with failures
	W := math.Min(simState.Workload, capacity)
	W = W * math.Max(0.1, 1.0-0.15*float64(simState.Failures))

	// Recovery time based on failures
	TRec := 5.0 + float64(simState.Failures)*2.0

	// Task 4: Uptime Model Update
	simState.TotalTime += 100.0 // say each timestep represents 100 observation units
	if newFailures > 0 {
		simState.Downtime += TRec * float64(newFailures)
	}
	TUp := simState.TotalTime - simState.Downtime
	if TUp < 0 {
		TUp = 0
	}

	// Energy Model
	P_idle := 50.0 * float64(simState.NNodes)
	P_max := 200.0 * float64(simState.NNodes)
	P := P_idle + (P_max-P_idle)*simState.UCpu
	simState.E = P
	simState.L = L
	simState.F = simState.Failures

	ci := 0.4 + 0.2*math.Sin(float64(simState.TimeStep)/20.0)

	resp := MetricsResponse{
		Sustainability: SustainabilityMetrics{
			ETotal:     P,
			W:          W,
			CI:         ci,
			ERenewable: P * 0.3, // 30% renewable
			UCpu:       simState.UCpu,
			UMem:       simState.UMem,
			USto:       simState.USto,
			UTarget:    simState.ThetaScale,
		},
		Resilience: ResilienceMetrics{
			TUp:           TUp,
			TObs:          math.Max(simState.TotalTime, 1.0),
			Nf:            simState.Failures,
			TRec:          TRec,
			TRecMax:       20.0,
			PMin:          math.Min(1.0, capacity/math.Max(1.0, simState.Workload)),
			PBaseline:     1.0,
			PPost:         0.9,
			WDisruption:   math.Max(0, simState.Workload-W),
			WBaseline:     simState.Workload,
			TDet:          float64(simState.Adversarial) * 5.0,
			TDetMax:       20.0,
			TSec:          float64(simState.Adversarial) * 10.0,
			TSecMax:       40.0,
			NAffected:     simState.Adversarial,
			NTotal:        simState.NNodes,
			SeverityScore: 0.2 * float64(simState.Adversarial),
			OConf:         0.95,
		},
		State: StateMetrics{
			UCpu:       simState.UCpu,
			UMem:       simState.UMem,
			USto:       simState.USto,
			E:          simState.E,
			L:          simState.L,
			F:          simState.F,
			NRep:       simState.NRep,
			NNodes:     simState.NNodes,
			PSched:     simState.PSched,
			ThetaScale: simState.ThetaScale,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func main() {
	http.HandleFunc("/step", handleStep)
	http.HandleFunc("/reset", handleReset)
	log.Println("KRSI Go Simulator starting on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
