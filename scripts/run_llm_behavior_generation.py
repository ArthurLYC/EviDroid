from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from evidroid.analyzers.llm_analyzer import OpenAIBehaviorAnalyzer
from evidroid.io_utils import read_jsonl, write_jsonl
from evidroid.settings import load_llm_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM behavior records with resume and bounded concurrency.")
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--out", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--config", default="configs/deepseek.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--prompt-mode", choices=["default", "malware_focused", "risk_focused"], default=None)
    parser.add_argument("--evidence-budget-mode", choices=["legacy", "compact", "adaptive"], default=None)
    parser.add_argument("--view-budgets", default=None)
    parser.add_argument("--max-value-chars", type=int, default=None)
    parser.add_argument("--compact-evidence", action="store_true")
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    out_path = Path(args.out)
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    if out_path.exists() and not args.resume:
        raise SystemExit(f"{out_path} exists. Pass --resume or --overwrite.")

    completed_ids = set()
    if args.resume and out_path.exists():
        completed_ids = {str(row.get("sample_id")) for row in read_jsonl(out_path) if row.get("sample_id")}
    print(f"[llm] completed={len(completed_ids)} out={out_path}", flush=True)

    rows = []
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            if sample_id in completed_ids:
                continue
            rows.append(row)
            if args.limit is not None and len(rows) >= args.limit:
                break
    print(f"[llm] pending={len(rows)} workers={args.workers}", flush=True)
    if not rows:
        return

    llm_config = load_llm_config(args.config)
    if args.model:
        llm_config["model"] = args.model
    if args.request_timeout is not None:
        llm_config["request_timeout"] = args.request_timeout
    if args.prompt_mode:
        llm_config["prompt_mode"] = args.prompt_mode
    if args.evidence_budget_mode:
        llm_config["evidence_budget_mode"] = args.evidence_budget_mode
    if args.view_budgets:
        llm_config["view_budgets"] = parse_view_budgets(args.view_budgets)
    if args.max_value_chars is not None:
        llm_config["max_value_chars"] = args.max_value_chars
    if args.compact_evidence:
        llm_config["compact_evidence"] = True

    started = time.perf_counter()
    done = 0
    failed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(analyze_with_retry, row, llm_config, args.max_retries, args.retry_base_seconds): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            sample_id = str(row.get("sample_id"))
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                result = {
                    "sample_id": sample_id,
                    "label": row.get("label"),
                    "analyzer": f"llm:{llm_config.get('model', 'unknown')}",
                    "behaviors": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[llm] failed {sample_id}: {result['error']}", flush=True)
            write_jsonl(out_path, [result], append=out_path.exists())
            done += 1
            if done == 1 or done % 50 == 0:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"[llm] done={done}/{len(rows)} failed={failed} rate={rate:.3f}/s last={sample_id}", flush=True)
    print(f"[llm] wrote {out_path} done={done} failed={failed}", flush=True)


def analyze_with_retry(
    evidence_doc: dict[str, Any],
    llm_config: dict[str, Any],
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max(1, max_retries) + 1):
        try:
            analyzer = OpenAIBehaviorAnalyzer.from_config(llm_config)
            start = time.perf_counter()
            result = analyzer.analyze(evidence_doc)
            result["timing"] = {**result.get("timing", {}), "behavior_seconds": time.perf_counter() - start}
            return result
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_base_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


def parse_view_budgets(raw: str) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid --view-budgets item: {item!r}")
        view, value = item.split("=", 1)
        budgets[view.strip()] = int(value)
    return budgets


if __name__ == "__main__":
    main()
