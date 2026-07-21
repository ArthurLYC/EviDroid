from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from evidroid.analyzers.llm_analyzer import OpenAIBehaviorAnalyzer
from evidroid.io_utils import write_json
from evidroid.schemas import group_evidence_by_view


PRESETS: dict[str, dict[str, Any]] = {
    "legacy80": {
        "evidence_budget_mode": "legacy",
        "max_evidence_per_view": 80,
        "compact_evidence": False,
        "max_value_chars": None,
    },
    "compact60": {
        "evidence_budget_mode": "adaptive",
        "view_budgets": {"permission": 80, "api": 60, "component": 45, "string": 40},
        "compact_evidence": True,
        "max_value_chars": 160,
    },
    "compact40": {
        "evidence_budget_mode": "adaptive",
        "view_budgets": {"permission": 80, "api": 40, "component": 30, "string": 25},
        "compact_evidence": True,
        "max_value_chars": 120,
    },
    "compact30": {
        "evidence_budget_mode": "adaptive",
        "view_budgets": {"permission": 60, "api": 35, "component": 24, "string": 18},
        "compact_evidence": True,
        "max_value_chars": 96,
    },
    "compact20": {
        "evidence_budget_mode": "adaptive",
        "view_budgets": {"permission": 50, "api": 25, "component": 18, "string": 12},
        "compact_evidence": True,
        "max_value_chars": 80,
    },
    "compact12": {
        "evidence_budget_mode": "adaptive",
        "view_budgets": {"permission": 35, "api": 18, "component": 12, "string": 8},
        "compact_evidence": True,
        "max_value_chars": 72,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate LLM prompt token budgets without calling the LLM API.")
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--behaviors", default=None, help="optional behavior JSONL for evidence-retention proxy metrics")
    parser.add_argument("--out-dir", default="artifacts/analysis/final_20000_token_budget")
    parser.add_argument("--prompt-mode", choices=["default", "malware_focused", "risk_focused"], default="risk_focused")
    parser.add_argument("--configs", default="legacy80,compact60,compact40,compact30,compact20,compact12")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--expected-labels",
        default="benign,malware",
        help="labels expected in --limit-per-class fast sampling; empty string disables early stop",
    )
    parser.add_argument("--total-rows", type=int, default=None, help="dataset size for cost extrapolation")
    parser.add_argument("--output-tokens-per-apk", type=float, default=900.0)
    parser.add_argument("--input-price-per-m", type=float, default=0.14)
    parser.add_argument("--output-price-per-m", type=float, default=0.28)
    args = parser.parse_args()

    config_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in config_names if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown configs: {', '.join(unknown)}. Available: {', '.join(PRESETS)}")

    behaviors_by_id = load_behavior_index(Path(args.behaviors)) if args.behaviors else {}
    selected_docs, label_counts = collect_evidence_docs(
        Path(args.evidence),
        limit=args.limit,
        limit_per_class=args.limit_per_class,
        expected_labels={item.strip() for item in args.expected_labels.split(",") if item.strip()},
    )
    if not selected_docs:
        raise SystemExit("No evidence rows selected.")

    total_rows = args.total_rows or sum(label_counts.values())
    summaries: list[dict[str, Any]] = []
    for config_name in config_names:
        analyzer = OpenAIBehaviorAnalyzer(
            model="deepseek-v4-flash",
            prompt_mode=args.prompt_mode,
            **PRESETS[config_name],
        )
        summaries.append(
            evaluate_config(
                name=config_name,
                analyzer=analyzer,
                evidence_docs=selected_docs,
                behaviors_by_id=behaviors_by_id,
                total_rows=total_rows,
                output_tokens_per_apk=args.output_tokens_per_apk,
                input_price_per_m=args.input_price_per_m,
                output_price_per_m=args.output_price_per_m,
            )
        )

    legacy = next((row for row in summaries if row["config"] == "legacy80"), None)
    if legacy:
        baseline_tokens = legacy["prompt_tokens_char4_mean"]
        for row in summaries:
            row["prompt_token_reduction_vs_legacy"] = _safe_ratio(
                baseline_tokens - row["prompt_tokens_char4_mean"],
                baseline_tokens,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "evidence": args.evidence,
        "behaviors": args.behaviors,
        "prompt_mode": args.prompt_mode,
        "selected_rows": len(selected_docs),
        "total_rows_seen": total_rows,
        "label_counts_seen": label_counts,
        "pricing_usd_per_m_tokens": {
            "input_cache_miss": args.input_price_per_m,
            "output": args.output_price_per_m,
        },
        "output_tokens_per_apk_assumption": args.output_tokens_per_apk,
        "configs": summaries,
    }
    write_json(out_dir / "token_budget_summary.json", result)
    (out_dir / "token_budget_summary.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_console_summary(result))


def collect_evidence_docs(
    path: Path,
    *,
    limit: int | None,
    limit_per_class: int | None,
    expected_labels: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    selected_by_label: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("label", ""))
            label_counts[label] = label_counts.get(label, 0) + 1
            if limit_per_class is not None and selected_by_label.get(label, 0) >= limit_per_class:
                continue
            if limit is not None and len(selected) >= limit:
                continue
            selected.append(row)
            selected_by_label[label] = selected_by_label.get(label, 0) + 1
            if (
                limit_per_class is not None
                and expected_labels
                and all(selected_by_label.get(item, 0) >= limit_per_class for item in expected_labels)
            ):
                break
    return selected, label_counts


def load_behavior_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id:
                rows[str(sample_id)] = row
    return rows


def evaluate_config(
    *,
    name: str,
    analyzer: OpenAIBehaviorAnalyzer,
    evidence_docs: list[dict[str, Any]],
    behaviors_by_id: dict[str, dict[str, Any]],
    total_rows: int,
    output_tokens_per_apk: float,
    input_price_per_m: float,
    output_price_per_m: float,
) -> dict[str, Any]:
    prompt_chars: list[int] = []
    selected_counts: list[int] = []
    available_counts: list[int] = []
    proxy = ProxyAccumulator()

    system_chars = len(analyzer._system_prompt())
    for evidence_doc in evidence_docs:
        prompt_chars.append(system_chars + len(analyzer._build_prompt(evidence_doc)))
        selected_ids = selected_evidence_ids(analyzer, evidence_doc)
        selected_counts.append(len(selected_ids))
        available_counts.append(len(evidence_doc.get("evidence", [])))
        behavior_doc = behaviors_by_id.get(str(evidence_doc.get("sample_id")))
        if behavior_doc:
            proxy.add(behavior_doc, selected_ids)

    mean_prompt_tokens = mean(prompt_chars) / 4.0
    extrapolated_input_tokens = mean_prompt_tokens * total_rows
    extrapolated_output_tokens = output_tokens_per_apk * total_rows
    extrapolated_cost = (
        extrapolated_input_tokens / 1_000_000.0 * input_price_per_m
        + extrapolated_output_tokens / 1_000_000.0 * output_price_per_m
    )

    return {
        "config": name,
        "params": PRESETS[name],
        "rows_evaluated": len(evidence_docs),
        "prompt_chars_mean": mean(prompt_chars),
        "prompt_chars_median": median(prompt_chars),
        "prompt_chars_p90": percentile(prompt_chars, 90),
        "prompt_tokens_char4_mean": mean_prompt_tokens,
        "prompt_tokens_char3_mean": mean(prompt_chars) / 3.0,
        "selected_evidence_mean": mean(selected_counts),
        "available_evidence_mean": mean(available_counts),
        "selected_evidence_ratio_mean": _safe_ratio(mean(selected_counts), mean(available_counts)),
        "extrapolated_input_tokens_char4": extrapolated_input_tokens,
        "extrapolated_output_tokens": extrapolated_output_tokens,
        "extrapolated_total_cost_usd": extrapolated_cost,
        "behavior_proxy": proxy.summary(),
    }


def selected_evidence_ids(analyzer: OpenAIBehaviorAnalyzer, evidence_doc: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    for view, items in group_evidence_by_view(evidence_doc).items():
        for item in analyzer._select_evidence_items(view, items):
            evidence_id = item.get("id")
            if evidence_id:
                selected.add(str(evidence_id))
    return selected


class ProxyAccumulator:
    def __init__(self) -> None:
        self.docs = 0
        self.docs_with_behaviors = 0
        self.behaviors = 0
        self.behaviors_at_least_one = 0
        self.behaviors_at_least_two = 0
        self.behaviors_all_refs = 0
        self.refs = 0
        self.refs_covered = 0

    def add(self, behavior_doc: dict[str, Any], selected_ids: set[str]) -> None:
        self.docs += 1
        behaviors = behavior_doc.get("behaviors", [])
        if behaviors:
            self.docs_with_behaviors += 1
        for behavior in behaviors:
            refs = [str(item) for item in behavior.get("evidence_ids", []) if item]
            if not refs:
                continue
            covered = sum(1 for ref in refs if ref in selected_ids)
            self.behaviors += 1
            self.refs += len(refs)
            self.refs_covered += covered
            if covered >= 1:
                self.behaviors_at_least_one += 1
            if covered >= min(2, len(refs)):
                self.behaviors_at_least_two += 1
            if covered == len(refs):
                self.behaviors_all_refs += 1

    def summary(self) -> dict[str, Any]:
        return {
            "docs_compared": self.docs,
            "docs_with_behaviors": self.docs_with_behaviors,
            "behavior_records": self.behaviors,
            "reference_coverage": _safe_ratio(self.refs_covered, self.refs),
            "behavior_at_least_one_ref": _safe_ratio(self.behaviors_at_least_one, self.behaviors),
            "behavior_at_least_two_refs": _safe_ratio(self.behaviors_at_least_two, self.behaviors),
            "behavior_all_refs": _safe_ratio(self.behaviors_all_refs, self.behaviors),
        }


def percentile(values: list[int], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def render_console_summary(result: dict[str, Any]) -> str:
    lines = [
        f"rows evaluated: {result['selected_rows']} / rows seen: {result['total_rows_seen']}",
        "config\tmean_prompt_tokens(char/4)\tcost_usd\tref_coverage\tbehavior>=1ref\treduction",
    ]
    for row in result["configs"]:
        proxy = row["behavior_proxy"]
        lines.append(
            "\t".join(
                [
                    row["config"],
                    f"{row['prompt_tokens_char4_mean']:.1f}",
                    f"{row['extrapolated_total_cost_usd']:.2f}",
                    f"{proxy['reference_coverage']:.4f}",
                    f"{proxy['behavior_at_least_one_ref']:.4f}",
                    f"{row.get('prompt_token_reduction_vs_legacy', 0.0):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LLM Token Budget Sweep",
        "",
        f"- Evidence: `{result['evidence']}`",
        f"- Behavior proxy: `{result['behaviors']}`",
        f"- Prompt mode: `{result['prompt_mode']}`",
        f"- Evaluated rows: {result['selected_rows']} of {result['total_rows_seen']} rows seen",
        f"- Output-token assumption: {result['output_tokens_per_apk_assumption']:.0f} tokens/APK",
        "",
        "| Config | Mean prompt tokens | Token reduction | Mean selected evidence | Est. full cost (USD) | Ref coverage | Behavior >=1 ref | Behavior >=2 refs | All refs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["configs"]:
        proxy = row["behavior_proxy"]
        lines.append(
            "| {config} | {tokens:.1f} | {reduction:.1%} | {selected:.1f} | ${cost:.2f} | {ref:.1%} | {one:.1%} | {two:.1%} | {all_refs:.1%} |".format(
                config=row["config"],
                tokens=row["prompt_tokens_char4_mean"],
                reduction=row.get("prompt_token_reduction_vs_legacy", 0.0),
                selected=row["selected_evidence_mean"],
                cost=row["extrapolated_total_cost_usd"],
                ref=proxy["reference_coverage"],
                one=proxy["behavior_at_least_one_ref"],
                two=proxy["behavior_at_least_two_refs"],
                all_refs=proxy["behavior_all_refs"],
            )
        )
    lines.extend(
        [
            "",
            "Reference coverage is an offline proxy against the existing behavior file. It measures whether evidence IDs used by current behavior records would still be visible to the LLM after compression; it is not a substitute for the final real-LLM classification experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
