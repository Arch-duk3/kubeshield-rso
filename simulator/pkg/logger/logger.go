// Package logger provides structured, JSON-formatted logging for the KRSI simulator.
// Every log entry carries simulation context (cycle ID, timestep, component) to
// enable forensic debugging, LLM-assisted analysis, and experiment reproducibility.
package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Level represents log severity.
type Level int

const (
	DEBUG   Level = iota
	INFO
	WARNING
	ERROR
)

func (l Level) String() string {
	switch l {
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case WARNING:
		return "WARNING"
	case ERROR:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// LogEntry is the canonical structured log record.
// Every field maps directly to the log schema in docs/log_schema.md.
type LogEntry struct {
	Timestamp   string                 `json:"timestamp"`
	Level       string                 `json:"level"`
	Component   string                 `json:"component"`
	EventType   string                 `json:"event_type"`
	SimCycleID  int64                  `json:"sim_cycle_id,omitempty"`
	TimeStep    int                    `json:"timestep,omitempty"`
	Message     string                 `json:"message"`
	Data        map[string]interface{} `json:"data,omitempty"`
}

// Logger is the central logger for a named component.
type Logger struct {
	component string
	level     Level
	out       *os.File
}

// New creates a Logger for the given component at the specified minimum level.
func New(component string, level Level) *Logger {
	return &Logger{
		component: component,
		level:     level,
		out:       os.Stdout,
	}
}

// log emits a structured JSON log entry.
func (l *Logger) log(lvl Level, eventType string, msg string, cycleID int64, timestep int, data map[string]interface{}) {
	if lvl < l.level {
		return
	}
	entry := LogEntry{
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		Level:      lvl.String(),
		Component:  l.component,
		EventType:  eventType,
		SimCycleID: cycleID,
		TimeStep:   timestep,
		Message:    msg,
		Data:       data,
	}
	b, err := json.Marshal(entry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "logger marshal error: %v\n", err)
		return
	}
	fmt.Fprintln(l.out, string(b))
}

// Info logs an informational event.
func (l *Logger) Info(eventType, msg string, cycleID int64, timestep int, data map[string]interface{}) {
	l.log(INFO, eventType, msg, cycleID, timestep, data)
}

// Debug logs a debug-level event.
func (l *Logger) Debug(eventType, msg string, cycleID int64, timestep int, data map[string]interface{}) {
	l.log(DEBUG, eventType, msg, cycleID, timestep, data)
}

// Warn logs a warning event.
func (l *Logger) Warn(eventType, msg string, cycleID int64, timestep int, data map[string]interface{}) {
	l.log(WARNING, eventType, msg, cycleID, timestep, data)
}

// Error logs an error event.
func (l *Logger) Error(eventType, msg string, cycleID int64, timestep int, data map[string]interface{}) {
	l.log(ERROR, eventType, msg, cycleID, timestep, data)
}

// --- Event type constants for consistent event taxonomy ---
const (
	EventSimStep        = "SIM_STEP"
	EventSimReset       = "SIM_RESET"
	EventActionApplied  = "ACTION_APPLIED"
	EventFailureInjected = "FAILURE_INJECTED"
	EventFailureRecovered = "FAILURE_RECOVERED"
	EventAdversarialEvent = "ADVERSARIAL_EVENT"
	EventWorkloadBurst  = "WORKLOAD_BURST"
	EventStateTransition = "STATE_TRANSITION"
	EventResourcePressure = "RESOURCE_PRESSURE"
	EventSchedulerChange  = "SCHEDULER_CHANGE"
	EventNodeAdded      = "NODE_ADDED"
	EventNodeRemoved    = "NODE_REMOVED"
	EventReplicaScaled  = "REPLICA_SCALED"
	EventSLAViolation   = "SLA_VIOLATION"
)
