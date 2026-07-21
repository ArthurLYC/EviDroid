from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evidroid.ablation import run_ablation
from evidroid.analyzers.llm_analyzer import OpenAIBehaviorAnalyzer
from evidroid.classifier_selection import DEFAULT_CLASSIFIERS, available_classifiers, run_classifier_selection
from evidroid.efficiency import run_efficiency_analysis
from evidroid.extractors.androguard_extractor import AndroguardEvidenceExtractor
from evidroid.io_utils import read_jsonl, write_json, write_jsonl
from evidroid.modeling import train_and_evaluate
from evidroid.reporting import generate_llm_analyst_report, predict_detection
from evidroid.settings import load_llm_config
from evidroid.template_reporting import generate_template_report, prediction_for_template_report


def main() -> None:
    parser = argparse.ArgumentParser(description="EviDroid research prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="extract evidence from APK files")
    extract_parser.add_argument("--raw-dir", default="data/raw")
    extract_parser.add_argument("--out", default="data/processed/evidence.jsonl")
    extract_parser.add_argument("--limit-per-class", type=int, default=None)
    extract_parser.add_argument("--limit", type=int, default=None)
    extract_parser.add_argument("--workers", type=int, default=1)
    extract_parser.add_argument("--overwrite", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="infer behavior findings")
    analyze_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    analyze_parser.add_argument("--out", default="data/processed/behaviors.jsonl")
    analyze_parser.add_argument("--analyzer", choices=["llm"], default="llm")
    analyze_parser.add_argument("--config", default=None, help="LLM config JSON, defaults to configs/deepseek.json")
    analyze_parser.add_argument("--model", default=None, help="Override the model configured in the LLM config file")
    analyze_parser.add_argument(
        "--prompt-mode",
        choices=["default", "malware_focused", "risk_focused"],
        default=None,
        help="Override the LLM behavior-inference prompt mode.",
    )
    analyze_parser.add_argument("--overwrite", action="store_true")
    analyze_parser.add_argument("--resume", action="store_true", help="skip sample IDs already present in --out and append missing rows")
    analyze_parser.add_argument("--limit", type=int, default=None, help="analyze at most this many evidence rows")
    analyze_parser.add_argument("--limit-per-class", type=int, default=None, help="analyze at most this many rows per class")
    analyze_parser.add_argument(
        "--expected-labels",
        default="benign,malware",
        help="labels expected by --limit-per-class for early stopping; empty string disables early stop",
    )
    analyze_parser.add_argument(
        "--evidence-budget-mode",
        choices=["legacy", "compact", "adaptive"],
        default=None,
        help="LLM evidence budget strategy; legacy preserves the original prompt behavior.",
    )
    analyze_parser.add_argument(
        "--view-budgets",
        default=None,
        help="comma-separated LLM evidence budgets, e.g. permission=80,api=35,component=30,string=25",
    )
    analyze_parser.add_argument("--max-value-chars", type=int, default=None, help="truncate evidence values in LLM prompts")
    analyze_parser.add_argument("--compact-evidence", action="store_true", help="send evidence as [id,value] pairs to reduce tokens")

    train_parser = subparsers.add_parser("train", help="train and evaluate classifier")
    train_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    train_parser.add_argument("--behaviors", default="data/processed/behaviors.jsonl")
    train_parser.add_argument("--out-dir", default="artifacts")
    train_parser.add_argument("--mode", choices=["static", "behavior", "fusion", "all"], default="fusion")
    train_parser.add_argument("--test-size", type=float, default=0.2)

    report_parser = subparsers.add_parser("report", help="generate a markdown report for one sample")
    report_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    report_parser.add_argument("--behaviors", default="data/processed/behaviors.jsonl")
    report_parser.add_argument("--sample-id", default=None)
    report_parser.add_argument("--out", default="reports/sample_report.md")
    report_parser.add_argument("--model-path", default=None, help="optional classifier model for detection result")
    report_parser.add_argument("--config", default=None, help="LLM config JSON, defaults to configs/deepseek.json")
    report_parser.add_argument("--llm-model", default=None, help="override configured LLM model for report generation")
    report_parser.add_argument("--language", default="zh-CN")
    report_parser.add_argument("--max-behaviors", type=int, default=4)
    report_parser.add_argument("--max-evidence-per-behavior", type=int, default=4)
    report_parser.add_argument("--min-consistency", type=float, default=0.0)
    report_parser.add_argument("--min-support-views", type=int, default=1)
    report_parser.add_argument("--top-k-behaviors", type=int, default=None)
    report_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")

    report_batch_parser = subparsers.add_parser("report-batch", help="generate per-sample analyst reports")
    report_batch_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    report_batch_parser.add_argument("--behaviors", default="data/processed/behaviors.jsonl")
    report_batch_parser.add_argument("--out-dir", default="reports/analyst")
    report_batch_parser.add_argument("--model-path", default=None, help="optional classifier model for detection result")
    report_batch_parser.add_argument("--config", default=None, help="LLM config JSON, defaults to configs/deepseek.json")
    report_batch_parser.add_argument("--llm-model", default=None, help="override configured LLM model for report generation")
    report_batch_parser.add_argument("--language", default="zh-CN")
    report_batch_parser.add_argument("--max-behaviors", type=int, default=4)
    report_batch_parser.add_argument("--max-evidence-per-behavior", type=int, default=4)
    report_batch_parser.add_argument("--limit", type=int, default=None)
    report_batch_parser.add_argument("--resume", action="store_true")
    report_batch_parser.add_argument("--min-consistency", type=float, default=0.0)
    report_batch_parser.add_argument("--min-support-views", type=int, default=1)
    report_batch_parser.add_argument("--top-k-behaviors", type=int, default=None)
    report_batch_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")

    template_report_parser = subparsers.add_parser("report-template", help="generate a markdown report without calling an LLM")
    template_report_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    template_report_parser.add_argument("--behaviors", default="data/processed/behaviors.jsonl")
    template_report_parser.add_argument("--sample-id", default=None)
    template_report_parser.add_argument("--out", default="reports/template_report.md")
    template_report_parser.add_argument("--model-path", default=None, help="optional classifier model for detection result")
    template_report_parser.add_argument("--max-behaviors", type=int, default=6)
    template_report_parser.add_argument("--max-evidence-per-behavior", type=int, default=5)
    template_report_parser.add_argument("--min-consistency", type=float, default=0.0)
    template_report_parser.add_argument("--min-support-views", type=int, default=1)
    template_report_parser.add_argument("--top-k-behaviors", type=int, default=None)
    template_report_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")

    template_batch_parser = subparsers.add_parser("report-template-batch", help="generate per-sample template reports without calling an LLM")
    template_batch_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    template_batch_parser.add_argument("--behaviors", default="data/processed/behaviors.jsonl")
    template_batch_parser.add_argument("--out-dir", default="reports/template")
    template_batch_parser.add_argument("--model-path", default=None, help="optional classifier model for detection result")
    template_batch_parser.add_argument("--max-behaviors", type=int, default=6)
    template_batch_parser.add_argument("--max-evidence-per-behavior", type=int, default=5)
    template_batch_parser.add_argument("--limit", type=int, default=None)
    template_batch_parser.add_argument("--resume", action="store_true")
    template_batch_parser.add_argument("--min-consistency", type=float, default=0.0)
    template_batch_parser.add_argument("--min-support-views", type=int, default=1)
    template_batch_parser.add_argument("--top-k-behaviors", type=int, default=None)
    template_batch_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")

    pipeline_parser = subparsers.add_parser("pipeline", help="run extract, analyze, and train")
    pipeline_parser.add_argument("--raw-dir", default="data/raw")
    pipeline_parser.add_argument("--limit-per-class", type=int, default=5)
    pipeline_parser.add_argument("--workers", type=int, default=1)

    ablate_parser = subparsers.add_parser("ablate", help="run A0-A3 ablation experiments")
    ablate_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    ablate_parser.add_argument("--behaviors", default="data/processed/behaviors_deepseek.jsonl")
    ablate_parser.add_argument("--out-dir", default="artifacts/ablation")
    ablate_parser.add_argument("--test-size", type=float, default=0.2)
    ablate_parser.add_argument("--random-state", type=int, default=42)
    ablate_parser.add_argument("--classifier", default="random_forest")
    ablate_parser.add_argument("--min-consistency", type=float, default=0.0)
    ablate_parser.add_argument("--min-support-views", type=int, default=1)
    ablate_parser.add_argument("--top-k-behaviors", type=int, default=None)
    ablate_parser.add_argument("--select-k-best", type=int, default=0)
    ablate_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")
    ablate_parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1")

    classifier_parser = subparsers.add_parser("select-classifier", help="run classifier selection experiments")
    classifier_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    classifier_parser.add_argument("--behaviors", default="data/processed/behaviors_deepseek.jsonl")
    classifier_parser.add_argument("--out-dir", default="artifacts/classifier_selection")
    classifier_parser.add_argument("--classifiers", default=",".join(DEFAULT_CLASSIFIERS))
    classifier_parser.add_argument("--test-size", type=float, default=0.2)
    classifier_parser.add_argument("--random-state", type=int, default=42)
    classifier_parser.add_argument("--min-consistency", type=float, default=0.0)
    classifier_parser.add_argument("--min-support-views", type=int, default=1)
    classifier_parser.add_argument("--top-k-behaviors", type=int, default=None)
    classifier_parser.add_argument("--select-k-best", type=int, default=0)
    classifier_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")
    classifier_parser.add_argument("--list-classifiers", action="store_true")

    efficiency_parser = subparsers.add_parser("efficiency", help="run efficiency analysis")
    efficiency_parser.add_argument("--raw-dir", default="data/raw")
    efficiency_parser.add_argument("--evidence", default="data/processed/evidence.jsonl")
    efficiency_parser.add_argument("--behaviors", default="data/processed/behaviors_deepseek.jsonl")
    efficiency_parser.add_argument("--out-dir", default="artifacts/efficiency")
    efficiency_parser.add_argument("--limit-per-class", type=int, default=None)
    efficiency_parser.add_argument("--limit", type=int, default=None)
    efficiency_parser.add_argument("--measure-extract", action="store_true")
    efficiency_parser.add_argument("--behavior-analyzer", choices=["existing", "llm"], default="existing")
    efficiency_parser.add_argument("--config", default=None, help="LLM config JSON for --behavior-analyzer llm")
    efficiency_parser.add_argument("--model", default=None, help="Override configured LLM model")
    efficiency_parser.add_argument("--classifier", default="random_forest")
    efficiency_parser.add_argument("--test-size", type=float, default=0.2)
    efficiency_parser.add_argument("--random-state", type=int, default=42)
    efficiency_parser.add_argument("--min-consistency", type=float, default=0.0)
    efficiency_parser.add_argument("--min-support-views", type=int, default=1)
    efficiency_parser.add_argument("--top-k-behaviors", type=int, default=None)
    efficiency_parser.add_argument("--select-k-best", type=int, default=0)
    efficiency_parser.add_argument("--static-profile", choices=["basic", "drebin", "compact"], default="basic")

    args = parser.parse_args()
    if args.command == "extract":
        command_extract(args)
    elif args.command == "analyze":
        command_analyze(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "report":
        command_report(args)
    elif args.command == "report-batch":
        command_report_batch(args)
    elif args.command == "report-template":
        command_report_template(args)
    elif args.command == "report-template-batch":
        command_report_template_batch(args)
    elif args.command == "pipeline":
        command_pipeline(args)
    elif args.command == "ablate":
        command_ablate(args)
    elif args.command == "select-classifier":
        command_select_classifier(args)
    elif args.command == "efficiency":
        command_efficiency(args)


def command_extract(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists. Pass --overwrite to replace it.")
    apk_rows = collect_apks(Path(args.raw_dir), args.limit_per_class, args.limit)
    print(f"[extract] APK files: {len(apk_rows)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, doc in enumerate(executor.map(_extract_one, apk_rows), start=1):
                print(f"[extract] {idx}/{len(apk_rows)} {doc.get('label')} {doc.get('sample_id')}")
                write_jsonl(out, [doc], append=out.exists())
    else:
        for idx, row in enumerate(apk_rows, start=1):
            print(f"[extract] {idx}/{len(apk_rows)} {row[1]} {row[0].name}")
            write_jsonl(out, [_extract_one(row)], append=out.exists())
    print(f"[extract] wrote {out}")


def command_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    completed_ids: set[str] = set()
    if out.exists() and getattr(args, "resume", False):
        completed_ids = {str(row.get("sample_id")) for row in read_jsonl(out) if row.get("sample_id")}
        print(f"[analyze] resume enabled: {len(completed_ids)} existing rows in {out}")
    elif out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists. Pass --overwrite to replace it.")
    elif out.exists() and args.overwrite:
        out.unlink()
    config_path = args.config or "configs/deepseek.json"
    llm_config = load_llm_config(config_path)
    if args.model:
        llm_config["model"] = args.model
    if args.prompt_mode:
        llm_config["prompt_mode"] = args.prompt_mode
    if args.evidence_budget_mode:
        llm_config["evidence_budget_mode"] = args.evidence_budget_mode
    if args.view_budgets:
        llm_config["view_budgets"] = _parse_view_budgets(args.view_budgets)
    if args.max_value_chars is not None:
        llm_config["max_value_chars"] = args.max_value_chars
    if args.compact_evidence:
        llm_config["compact_evidence"] = True
    analyzer = OpenAIBehaviorAnalyzer.from_config(llm_config)
    evidence_rows = _iter_limited_jsonl(
        Path(args.evidence),
        limit=args.limit,
        limit_per_class=args.limit_per_class,
        expected_labels={item.strip() for item in args.expected_labels.split(",") if item.strip()},
    )
    for idx, evidence_doc in enumerate(evidence_rows, start=1):
        sample_id = str(evidence_doc["sample_id"])
        if sample_id in completed_ids:
            print(f"[analyze] {idx} {sample_id} skipped")
            continue
        print(f"[analyze] {idx} {sample_id}")
        start = time.perf_counter()
        behavior_doc = analyzer.analyze(evidence_doc)
        elapsed = time.perf_counter() - start
        behavior_doc["timing"] = {**behavior_doc.get("timing", {}), "behavior_seconds": elapsed}
        write_jsonl(out, [behavior_doc], append=out.exists())
    print(f"[analyze] wrote {out}")


def _parse_view_budgets(raw: str) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"Invalid --view-budgets item: {part!r}")
        view, value = part.split("=", 1)
        try:
            budgets[view.strip()] = int(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid budget value in --view-budgets item: {part!r}") from exc
    return budgets


def _iter_limited_jsonl(
    path: Path,
    *,
    limit: int | None = None,
    limit_per_class: int | None = None,
    expected_labels: set[str] | None = None,
):
    counts: dict[str, int] = {}
    yielded = 0
    expected_labels = expected_labels or set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = read_jsonl_line(line)
            label = str(row.get("label", ""))
            if limit_per_class is not None and counts.get(label, 0) >= limit_per_class:
                continue
            if limit is not None and yielded >= limit:
                break
            counts[label] = counts.get(label, 0) + 1
            yielded += 1
            yield row
            if (
                limit_per_class is not None
                and expected_labels
                and all(counts.get(item, 0) >= limit_per_class for item in expected_labels)
            ):
                break


def read_jsonl_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def command_train(args: argparse.Namespace) -> None:
    modes = ["static", "behavior", "fusion"] if args.mode == "all" else [args.mode]
    metrics_rows = []
    for mode in modes:
        print(f"[train] mode={mode}")
        metrics = train_and_evaluate(
            evidence_path=args.evidence,
            behavior_path=args.behaviors,
            out_dir=args.out_dir,
            mode=mode,
            test_size=args.test_size,
        )
        metrics_rows.append(metrics)
        print(
            f"[train] {mode}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )
    if len(metrics_rows) > 1:
        write_json(Path(args.out_dir) / "ablation_metrics.json", {"metrics": metrics_rows})


def command_report(args: argparse.Namespace) -> None:
    evidence_rows = read_jsonl(args.evidence)
    behavior_rows = {row["sample_id"]: row for row in read_jsonl(args.behaviors)}
    evidence_doc = None
    if args.sample_id:
        evidence_doc = next((row for row in evidence_rows if row["sample_id"] == args.sample_id), None)
    elif evidence_rows:
        evidence_doc = evidence_rows[0]
    if not evidence_doc:
        raise SystemExit("No matching evidence row found.")
    behavior_doc = behavior_rows.get(evidence_doc["sample_id"], {"sample_id": evidence_doc["sample_id"], "behaviors": []})
    prediction_doc = _prediction_for_report(args, evidence_doc, behavior_doc)
    llm_config = _report_llm_config(args)
    generate_llm_analyst_report(
        evidence_doc,
        behavior_doc,
        args.out,
        llm_config,
        prediction_doc=prediction_doc,
        language=getattr(args, "language", "zh-CN"),
        max_behaviors=getattr(args, "max_behaviors", 4),
        max_evidence_per_behavior=getattr(args, "max_evidence_per_behavior", 4),
    )
    print(f"[report] wrote {args.out}")


def command_report_batch(args: argparse.Namespace) -> None:
    evidence_rows = read_jsonl(args.evidence)
    behavior_rows = {row["sample_id"]: row for row in read_jsonl(args.behaviors)}
    if args.limit is not None:
        evidence_rows = evidence_rows[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_config = _report_llm_config(args)
    written = 0
    for idx, evidence_doc in enumerate(evidence_rows, start=1):
        sample_id = evidence_doc["sample_id"]
        out_path = out_dir / f"{sample_id}.md"
        if args.resume and out_path.exists():
            print(f"[report-batch] {idx}/{len(evidence_rows)} {sample_id} skipped")
            continue
        behavior_doc = behavior_rows.get(sample_id, {"sample_id": sample_id, "behaviors": []})
        prediction_doc = _prediction_for_report(args, evidence_doc, behavior_doc)
        print(f"[report-batch] {idx}/{len(evidence_rows)} {sample_id}")
        generate_llm_analyst_report(
            evidence_doc,
            behavior_doc,
            out_path,
            llm_config,
            prediction_doc=prediction_doc,
            language=args.language,
            max_behaviors=args.max_behaviors,
            max_evidence_per_behavior=args.max_evidence_per_behavior,
        )
        written += 1
    print(f"[report-batch] wrote {written} reports to {out_dir}")


def command_report_template(args: argparse.Namespace) -> None:
    evidence_rows = read_jsonl(args.evidence)
    behavior_rows = {row["sample_id"]: row for row in read_jsonl(args.behaviors)}
    evidence_doc = None
    if args.sample_id:
        evidence_doc = next((row for row in evidence_rows if row["sample_id"] == args.sample_id), None)
    elif evidence_rows:
        evidence_doc = evidence_rows[0]
    if not evidence_doc:
        raise SystemExit("No matching evidence row found.")
    behavior_doc = behavior_rows.get(evidence_doc["sample_id"], {"sample_id": evidence_doc["sample_id"], "behaviors": []})
    prediction_doc = _prediction_for_template_report(args, evidence_doc, behavior_doc)
    generate_template_report(
        evidence_doc=evidence_doc,
        behavior_doc=behavior_doc,
        out_path=args.out,
        prediction_doc=prediction_doc,
        max_behaviors=args.max_behaviors,
        max_evidence_per_behavior=args.max_evidence_per_behavior,
    )
    print(f"[report-template] wrote {args.out}")


def command_report_template_batch(args: argparse.Namespace) -> None:
    evidence_rows = read_jsonl(args.evidence)
    behavior_rows = {row["sample_id"]: row for row in read_jsonl(args.behaviors)}
    if args.limit is not None:
        evidence_rows = evidence_rows[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, evidence_doc in enumerate(evidence_rows, start=1):
        sample_id = evidence_doc["sample_id"]
        out_path = out_dir / f"{sample_id}.md"
        if args.resume and out_path.exists():
            print(f"[report-template-batch] {idx}/{len(evidence_rows)} {sample_id} skipped")
            continue
        behavior_doc = behavior_rows.get(sample_id, {"sample_id": sample_id, "behaviors": []})
        prediction_doc = _prediction_for_template_report(args, evidence_doc, behavior_doc)
        print(f"[report-template-batch] {idx}/{len(evidence_rows)} {sample_id}")
        generate_template_report(
            evidence_doc=evidence_doc,
            behavior_doc=behavior_doc,
            out_path=out_path,
            prediction_doc=prediction_doc,
            max_behaviors=args.max_behaviors,
            max_evidence_per_behavior=args.max_evidence_per_behavior,
        )
        written += 1
    print(f"[report-template-batch] wrote {written} reports to {out_dir}")


def _prediction_for_report(
    args: argparse.Namespace,
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
) -> dict[str, Any] | None:
    if not getattr(args, "model_path", None):
        return None
    return predict_detection(
        evidence_doc,
        behavior_doc,
        args.model_path,
        min_consistency=getattr(args, "min_consistency", 0.0),
        min_support_views=getattr(args, "min_support_views", 1),
        top_k_behaviors=getattr(args, "top_k_behaviors", None),
        static_profile=getattr(args, "static_profile", "basic"),
    )


def _prediction_for_template_report(
    args: argparse.Namespace,
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
) -> dict[str, Any] | None:
    return prediction_for_template_report(
        evidence_doc=evidence_doc,
        behavior_doc=behavior_doc,
        model_path=getattr(args, "model_path", None),
        min_consistency=getattr(args, "min_consistency", 0.0),
        min_support_views=getattr(args, "min_support_views", 1),
        top_k_behaviors=getattr(args, "top_k_behaviors", None),
        static_profile=getattr(args, "static_profile", "basic"),
    )


def _report_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    llm_config = load_llm_config(getattr(args, "config", None) or "configs/deepseek.json")
    if getattr(args, "llm_model", None):
        llm_config["model"] = args.llm_model
    return llm_config


def command_pipeline(args: argparse.Namespace) -> None:
    evidence = "data/processed/evidence.jsonl"
    behaviors = "data/processed/behaviors.jsonl"
    command_extract(
        argparse.Namespace(
            raw_dir=args.raw_dir,
            out=evidence,
            limit_per_class=args.limit_per_class,
            limit=None,
            workers=args.workers,
            overwrite=True,
        )
    )
    command_analyze(
        argparse.Namespace(
            evidence=evidence,
            out=behaviors,
            analyzer="llm",
            config=None,
            model=None,
            prompt_mode=None,
            evidence_budget_mode=None,
            view_budgets=None,
            max_value_chars=None,
            compact_evidence=False,
            overwrite=True,
            resume=False,
            limit=None,
            limit_per_class=None,
            expected_labels="benign,malware",
        )
    )
    command_train(
        argparse.Namespace(
            evidence=evidence,
            behaviors=behaviors,
            out_dir="artifacts",
            mode="fusion",
            test_size=0.2,
        )
    )
    print("[pipeline] report generation is a separate LLM step; run `report` when needed.")


def command_ablate(args: argparse.Namespace) -> None:
    summary = run_ablation(
        evidence_path=args.evidence,
        behavior_path=args.behaviors,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        classifier=args.classifier,
        min_consistency=args.min_consistency,
        min_support_views=args.min_support_views,
        top_k_behaviors=args.top_k_behaviors,
        select_k_best=args.select_k_best,
        static_profile=args.static_profile,
        feature_version=args.feature_version,
    )
    for metrics in summary["metrics"]:
        print(
            f"[ablate] {metrics['variant_id']} {metrics['variant_name']}: "
            f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']}"
        )
    print(f"[ablate] wrote {Path(args.out_dir) / 'ablation_metrics.json'}")


def command_select_classifier(args: argparse.Namespace) -> None:
    if args.list_classifiers:
        availability = available_classifiers()
        for name, is_available in availability.items():
            print(f"{name}: {'available' if is_available else 'unavailable'}")
        return

    classifiers = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    summary = run_classifier_selection(
        evidence_path=args.evidence,
        behavior_path=args.behaviors,
        out_dir=args.out_dir,
        classifiers=classifiers,
        test_size=args.test_size,
        random_state=args.random_state,
        min_consistency=args.min_consistency,
        min_support_views=args.min_support_views,
        top_k_behaviors=args.top_k_behaviors,
        select_k_best=args.select_k_best,
        static_profile=args.static_profile,
    )
    for metrics in summary["metrics"]:
        if metrics.get("status") != "ok":
            print(
                f"[select-classifier] {metrics['classifier']}: "
                f"{metrics['status']} ({metrics.get('reason', '')})"
            )
            continue
        print(
            f"[select-classifier] {metrics['classifier']}: "
            f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']}, fit_seconds={metrics['fit_seconds']:.4f}"
        )
    print(f"[select-classifier] wrote {Path(args.out_dir) / 'classifier_selection_metrics.json'}")


def command_efficiency(args: argparse.Namespace) -> None:
    summary = run_efficiency_analysis(
        raw_dir=args.raw_dir,
        evidence_path=args.evidence,
        behavior_path=args.behaviors,
        out_dir=args.out_dir,
        limit_per_class=args.limit_per_class,
        limit=args.limit,
        measure_extract=args.measure_extract,
        behavior_analyzer=args.behavior_analyzer,
        llm_config_path=args.config,
        model=args.model,
        classifier=args.classifier,
        test_size=args.test_size,
        random_state=args.random_state,
        min_consistency=args.min_consistency,
        min_support_views=args.min_support_views,
        top_k_behaviors=args.top_k_behaviors,
        select_k_best=args.select_k_best,
        static_profile=args.static_profile,
    )
    extraction = summary["extraction"]
    behavior = summary["behavior"]
    feature = summary["feature_construction"]
    classification = summary["classification"]
    if extraction.get("seconds"):
        print(f"[efficiency] extraction_mean_seconds={extraction['seconds']['mean']:.4f}")
    else:
        print("[efficiency] extraction timing reused existing evidence")
    if behavior.get("seconds"):
        print(f"[efficiency] behavior_mean_seconds={behavior['seconds']['mean']:.4f}")
    else:
        print("[efficiency] behavior timing reused existing behavior file")
    print(f"[efficiency] feature_mean_seconds={feature['seconds']['mean']:.6f}")
    if classification.get("status") == "skipped":
        print(f"[efficiency] classifier skipped: {classification.get('reason')}")
    else:
        print(
            f"[efficiency] classifier={classification['classifier']} "
            f"fit_seconds={classification['fit_seconds']:.4f} "
            f"predict_seconds={classification['predict_seconds']:.4f}"
        )
    print(f"[efficiency] wrote {Path(args.out_dir) / 'efficiency_metrics.json'}")
    print(f"[efficiency] wrote {Path(args.out_dir) / 'efficiency_report.md'}")


def collect_apks(
    raw_dir: Path,
    limit_per_class: int | None = None,
    limit: int | None = None,
) -> list[tuple[Path, str | None]]:
    rows: list[tuple[Path, str | None]] = []
    class_dirs = [(raw_dir / "benign", "benign"), (raw_dir / "malware", "malware")]
    if all(path.exists() for path, _label in class_dirs):
        for path, label in class_dirs:
            files = sorted(path.rglob("*.apk"))
            if limit_per_class is not None:
                files = files[:limit_per_class]
            rows.extend((item, label) for item in files)
    else:
        files = sorted(raw_dir.rglob("*.apk"))
        if limit is not None:
            files = files[:limit]
        rows.extend((item, None) for item in files)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _extract_one(row: tuple[Path, str | None]) -> dict[str, Any]:
    apk_path, label = row
    extractor = AndroguardEvidenceExtractor()
    start = time.perf_counter()
    doc = extractor.extract(apk_path, label=label)
    elapsed = time.perf_counter() - start
    doc["timing"] = {**doc.get("timing", {}), "extract_seconds": elapsed}
    return doc


if __name__ == "__main__":
    main()
