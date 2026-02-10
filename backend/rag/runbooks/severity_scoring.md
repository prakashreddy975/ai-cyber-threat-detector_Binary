# Severity Scoring (LOW / MEDIUM / HIGH)

## LOW
- Low threat_score, weak indicators, likely benign anomaly
- Action: log, monitor, enrich context, no immediate containment

## MEDIUM
- Credible suspicious indicators (scan, brute force patterns, odd service state)
- Action: verify with logs, correlate in last 15m, consider block if repeated

## HIGH
- Strong indicators of compromise OR clear exploit / active attack pattern
- Action: isolate endpoint, block IP (if safe), escalate, capture evidence
