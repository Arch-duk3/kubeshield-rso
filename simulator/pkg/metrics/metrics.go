// Package metrics provides Prometheus instrumentation for the KRSI simulator.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"krsi-simulator/models"
)

var (
	// Sustainability Metrics
	EnergyConsumption = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_energy_consumption_watts",
		Help: "Current power draw of the cluster in Watts",
	}, []string{"sim_cycle_id"})

	Throughput = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_throughput_requests_per_second",
		Help: "Work processed by the cluster",
	}, []string{"sim_cycle_id"})

	CarbonIntensity = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_carbon_intensity",
		Help: "Current carbon intensity of the energy grid",
	}, []string{"sim_cycle_id"})

	// Resilience Metrics
	ActiveFailures = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_active_failures_count",
		Help: "Number of currently active pod/node failures",
	}, []string{"sim_cycle_id"})

	AdversarialEvents = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "krsi_adversarial_events_total",
		Help: "Total number of adversarial events injected",
	}, []string{"sim_cycle_id"})

	Latency = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_latency_ms",
		Help: "Current service latency in milliseconds",
	}, []string{"sim_cycle_id"})

	// State Metrics
	ReplicaCount = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_replica_count",
		Help: "Current number of running replicas",
	}, []string{"sim_cycle_id"})

	NodeCount = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "krsi_node_count",
		Help: "Current number of worker nodes",
	}, []string{"sim_cycle_id"})
)

// RecordMetrics updates the Prometheus gauges with the current simulation state.
func RecordMetrics(m models.MetricsResponse) {
	cycleStr := string(m.SimCycleID)
	
	EnergyConsumption.WithLabelValues(cycleStr).Set(m.Sustainability.ETotal)
	Throughput.WithLabelValues(cycleStr).Set(m.Sustainability.W)
	CarbonIntensity.WithLabelValues(cycleStr).Set(m.Sustainability.CI)
	
	ActiveFailures.WithLabelValues(cycleStr).Set(float64(m.Resilience.Nf))
	Latency.WithLabelValues(cycleStr).Set(m.State.L)
	
	ReplicaCount.WithLabelValues(cycleStr).Set(float64(m.State.NRep))
	NodeCount.WithLabelValues(cycleStr).Set(float64(m.State.NNodes))
}
