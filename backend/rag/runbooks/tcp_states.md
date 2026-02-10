# TCP State Reference

## FIN
- FIN indicates a graceful connection close.
- If FIN appears with unusual byte/packet patterns, it may still be malicious depending on context.

## SYN
- SYN indicates connection initiation.
- Repeated SYN spikes can indicate SYN flood / scanning.

## RST
- RST indicates abrupt reset.
- High RST rates can indicate scanning, failed connections, or filtering.
