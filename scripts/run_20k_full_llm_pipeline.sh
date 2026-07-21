#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
[deprecated] scripts/run_20k_full_llm_pipeline.sh was a legacy 20k scratch pipeline.
It is not part of the current paper protocol.

Use the final 20,000-APK full-LLM command:

python scripts/run_multiseed_experiments.py \
  --evidence data/processed/evidence_final_20000_balanced_20260706.jsonl \
  --behaviors data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl \
  --mamadroid-cache data/processed/mamadroid_features_final_20000_balanced_20260706.jsonl \
  --out-dir artifacts/optimized/full_llm_final_20000_balanced_multiseed_20260706 \
  --seeds 42,2026,2027 \
  --save-predictions
EOF
exit 1
