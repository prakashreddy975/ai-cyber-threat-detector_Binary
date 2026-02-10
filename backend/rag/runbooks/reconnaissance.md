# Generic Network Anomalies Runbook

## Overview
Generic anomalies represent traffic patterns that do not match known benign behavior but are not clearly malicious.

## Examples
- Unusual protocol usage
- Traffic at odd hours
- Unexpected packet sizes

## ML Indicators
- Feature distribution drift
- Unseen categorical values
- Deviations from learned baseline

## Risk Classification
- Usually MEDIUM
- May escalate with repetition

## Response
- Monitor closely
- Correlate with system changes
- Validate business context

## Model Feedback
Generic anomalies are valuable for retraining and drift detection.
