# Incident Response Basics

## Severity
- HIGH: likely compromise / ongoing attack
- MEDIUM: suspicious activity; needs verification
- LOW: informational or benign anomaly

## Immediate actions
1) Confirm alert details: risk_level, threat_score, timestamps
2) Check source IP, destination service, port/protocol patterns
3) Correlate with recent events in last 5–15 minutes
4) If HIGH: isolate endpoint, block IP (if safe), escalate

## Evidence to collect
- Last 50 events for same src/dst/service
- Authentication logs around timestamp
- Firewall logs and IDS signatures
