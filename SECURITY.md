# Security Policy

## Supported Versions

The KRSI Framework team provides security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of this research framework seriously. If you discover any security issues, please do not open a public issue. Instead, report them via the following process:

1. Send an email to `security@krsi-research.org` (Placeholder).
2. Include a detailed description of the vulnerability, steps to reproduce, and potential impact.
3. We will acknowledge your report within 48 hours and provide a timeline for resolution.

## Security Architecture

The KRSI Framework implements several layers of security hardening:

### 1. Input Validation
The simulator API server performs strict range validation on all control actions to prevent out-of-bounds state corruption.

### 2. Container Hardening
Docker images are built using minimal base images (Alpine, Slim) to reduce the attack surface. Root privileges are minimized where possible.

### 3. Dependency Scanning
All Python and Go dependencies are pinned to specific versions to prevent supply chain attacks via unvetted updates.

### 4. Threat Model
- **Trust Boundary**: The boundary between the RL Controller (Python) and the Simulator (Go).
- **Asset**: Simulation integrity and experimental reproducibility.
- **Threat**: Malicious action injection or state corruption via API server exploitation.
- **Mitigation**: Schema validation and strict typing in both layers.

## Disclaimers
This is a **research simulation framework**. While we strive for engineering maturity, it is not intended to manage production Kubernetes clusters directly without further security auditing of the controller-to-cluster bridge (if implemented).
