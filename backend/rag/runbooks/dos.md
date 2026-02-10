# Denial of Service (DoS / DDoS) Runbook

## Overview
Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks attempt to disrupt the availability of a system, service, or network by overwhelming it with traffic or resource exhaustion.
Distributed Denial of Service (DDoS) is a cyberattack in which multiple systems flood a target with traffic, overwhelming resources and making services unavailable to legitimate users.

## Threat Description
- DoS: Single-source attack
- DDoS: Multi-source (botnet-based) attack
- Targets: Web servers, APIs, DNS, firewalls

## Common Attack Types
- SYN Flood
- UDP Flood
- ICMP Flood
- HTTP GET/POST Flood
- Slowloris
- Amplification attacks (DNS, NTP, SSDP)

## Detection Indicators
- Sudden spike in `spkts`, `dpkts`, `rate`
- Abnormally high `sload` or `dload`
- Consistent traffic with minimal payload
- TCP half-open connections
- Repeated requests to same endpoint
- Latency degradation across services

## ML Features Commonly Triggered
- rate
- spkts / dpkts
- sload / dload
- synack anomalies
- tcp flags imbalance

## Risk Classification
- MEDIUM: Short bursts, no service impact
- HIGH: Sustained traffic causing latency or downtime
- CRITICAL: Service outage or cascading failures

## Immediate Response (0–5 minutes)
1. Enable rate limiting on ingress firewall.
2. Block top offending IPs or CIDR ranges.
3. Enable SYN cookies (if TCP-based).
4. Notify SOC Tier-2.

## Short-Term Mitigation (5–30 minutes)
- Redirect traffic through DDoS protection service.
- Apply geo-blocking if attack origin is localized.
- Scale horizontally if cloud-based.

## Validation Steps
- Confirm legitimate traffic is not blocked.
- Compare traffic to historical baselines.
- Check upstream provider alerts.

## Recovery
- Gradually remove mitigation rules.
- Monitor for secondary attack waves.
- Validate system stability.

## Post-Incident Actions
- Update detection thresholds.
- Add attack samples to training data.
- Review capacity planning.

## False Positive Considerations
- Marketing campaigns
- Software updates
- Load testing

## Compliance Notes
DoS incidents affecting availability must be logged for SOC2 and ISO27001 audits.

## Escalation
- HIGH: SOC Tier-2 within 10 minutes
- CRITICAL: Incident Response Lead immediately
