# Reconnaissance Playbook

## Indicators
- Many destination ports or services in short time
- Low bytes, short durations, unusual TCP flags/state patterns

## Actions
- Correlate across IPs
- Identify targeted services
- Block if repeated and clearly malicious

## Evidence
- Firewall deny logs, port scan signatures
- Authentication failure bursts
