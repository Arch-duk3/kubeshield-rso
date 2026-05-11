// KRSI Simulator — Main entrypoint
// Wires together engine, API server, and structured logger.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"krsi-simulator/api"
	"krsi-simulator/engine"
	"krsi-simulator/pkg/logger"
)

func main() {
	// CLI flags — all simulator parameters are configurable at launch,
	// so container deployments can override via environment/entrypoint args.
	port := flag.Int("port", 8080, "HTTP port for the simulator API")
	seed := flag.Int64("seed", 42, "Random seed for reproducible simulation")
	logLevel := flag.String("log-level", "INFO", "Log level: DEBUG|INFO|WARNING|ERROR")
	flag.Parse()

	// Parse log level
	var level logger.Level
	switch *logLevel {
	case "DEBUG":
		level = logger.DEBUG
	case "WARNING":
		level = logger.WARNING
	case "ERROR":
		level = logger.ERROR
	default:
		level = logger.INFO
	}

	log := logger.New("simulator.main", level)

	cfg := engine.DefaultConfig()
	cfg.Seed = *seed

	eng := engine.New(cfg, logger.New("simulator.engine", level))
	srv := api.New(eng, log)

	mux := http.NewServeMux()
	srv.RegisterRoutes(mux)

	addr := fmt.Sprintf(":%d", *port)
	log.Info("SIM_STARTUP", "KRSI simulator starting", 0, 0, map[string]interface{}{
		"addr": addr,
		"seed": *seed,
	})

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Error("SIM_FATAL", "Simulator failed to start", 0, 0, map[string]interface{}{"error": err.Error()})
		log.Error("SIM_FATAL", err.Error(), 0, 0, nil)
		panic(err)
	}
}

// Ensure the standard log package doesn't interfere with structured output
func init() {
	log.SetFlags(0)
	log.SetPrefix("")
}
