# DoS / DDoS Playbook

## Indicators
- Sudden spike in rate, spkts/dpkts, repeated flows
- High SYN volume, many short-lived connections

## Immediate actions
- Rate limit at edge
- Block abusive IPs / ASNs (carefully)
- Verify service health metrics

## Evidence
- Source distribution, top talkers
- Per-service traffic volume
- WAF/firewall logs
