# False Positive Management Runbook

## Definition
An alert incorrectly flagged as malicious.

## Common Causes
- Internal scanners
- Load tests
- Legitimate automation
- New applications

## Handling Process
1. Confirm benign behavior.
2. Label as FALSE_POSITIVE.
3. Store for model retraining.
4. Adjust thresholds if required.

## Governance
False positives must be reviewed weekly.

## Model Improvement
High false-positive rates indicate feature imbalance or drift.
