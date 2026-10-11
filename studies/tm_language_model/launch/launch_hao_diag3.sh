#!/usr/bin/env bash
# Wait (no CPU slot held) until every confirmatory R1 lane has finished, then run the H-AO rerun chain inside one CPU slot.
# Recovered 2026-09-28 from the saved record (original 2026-09-25 18:30 UTC); only the night directory changed.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=$LAUNCH_DIR
for L in common seed1 seed2 seed3 report; do
  until grep -q "CHAIN r1_confirm lane $L DONE" $N/chain_r1_confirm_$L.out 2>/dev/null || ! pgrep -f "chain_r1_confirm.sh $L" >/dev/null; do sleep 300; done
done
echo "confirmatory lanes finished $(date -u +%FT%TZ): taking a CPU slot for the H-AO rerun"
cd "$LAUNCH_DIR/.." && exec scripts/cpu_slot.sh 1 -- bash $N/chain_hao_diag3.sh
