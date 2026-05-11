// Package api implements the HTTP server for the KRSI simulator.
// It exposes /step and /reset endpoints consumed by the Python RL controller.
//
// WHY a separate API package: HTTP routing, request validation, and response
// serialization are distinct concerns from simulation physics. Separation
// enables mocking the API layer in integration tests without touching engine logic.
package api

import (
	"encoding/json"
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"krsi-simulator/engine"
	"krsi-simulator/models"
	"krsi-simulator/pkg/logger"
	"krsi-simulator/pkg/metrics"
)

// Server wraps the simulation engine and exposes it over HTTP.
type Server struct {
	eng *engine.Engine
	log *logger.Logger
}

// New creates an API server bound to the given engine.
func New(eng *engine.Engine, log *logger.Logger) *Server {
	return &Server{eng: eng, log: log}
}

// RegisterRoutes registers all API routes on the given mux.
// Using an explicit mux parameter (rather than http.DefaultServeMux) allows
// test code to create isolated servers without global state interference.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/step", s.handleStep)
	mux.HandleFunc("/reset", s.handleReset)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/state", s.handleState)
	mux.Handle("/metrics", promhttp.Handler())
}

// handleReset resets the simulation to the start of a new episode.
func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.eng.Reset()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "reset"})
}

// handleStep applies an action and returns the next state metrics.
// Input validation is performed before touching the engine to prevent
// invalid actions from corrupting simulation state.
func (s *Server) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req models.StepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.log.Warn("INVALID_REQUEST", "Failed to decode step request", 0, 0,
			map[string]interface{}{"error": err.Error()})
		http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate action range [0, 8]
	if req.Action < 0 || req.Action > 8 {
		s.log.Warn("INVALID_ACTION", "Action out of valid range [0,8]", 0, 0,
			map[string]interface{}{"action": req.Action})
		http.Error(w, "action must be in range [0, 8]", http.StatusBadRequest)
		return
	}

	resp := s.eng.Step(req.Action)
	metrics.RecordMetrics(resp)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHealth implements a liveness probe for Docker/Kubernetes health checks.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

// handleState returns the current internal simulator state (for debugging/observability).
func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	state := s.eng.GetState()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}
