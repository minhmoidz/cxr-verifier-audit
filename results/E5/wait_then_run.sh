#!/bin/bash
cd "$(dirname "$0")/../.."
while pgrep -f "^python3 -u artifacts/paper_upgrade/phase5_pairing_audit/run_pairing_audit.py" >/dev/null; do sleep 120; done
echo "$(date +%F_%T) E3 finished; starting E5"
bash results/E5/commands.sh
