from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from evidroid.baselines import load_sample_index
from evidroid.dynamic_weights import learn_view_weight_spec
from evidroid.features import build_ablation_feature_parts
from evidroid.io_utils import read_jsonl, write_json
from run_multiseed_experiments import (
    ABLATION_VARIANTS,
    METRIC_KEYS,
    SAMPLE_ID_RE,
    compose_variant_features,
    make_split,
    run_encoder_fusion_variant,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check A0-A3 validity for multiple EviDroid fusion heads.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--behaviors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="42,2026,2027")
    parser.add_argument("--heads", default="extra_trees,gradient_boosting,adaboost,xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--evidroid-alpha", type=float, default=0.75)
    parser.add_argument("--evidroid-shrinkage-samples", type=float, default=10.0)
    parser.add_argument("--evidroid-dynamic-consistency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidroid-augment-fixed-consistency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablation-static-profile", choices=["basic", "drebin", "compact"], default="compact")
    parser.add_argument("--ablation-feature-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--fusion-folds", type=int, default=5)
    parser.add_argument("--fusion-static-classifier", default="logistic_regression_sgd")
    parser.add_argument("--fusion-behavior-classifier", default="random_forest")
    parser.add_argument("--fusion-behavior-evidence-classifier", default="xgboost")
    parser.add_argument("--fusion-consistency-classifier", default="xgboost")
    parser.add_argument("--fusion-static-select-k", type=int, default=12000)
    parser.add_argument("--fusion-behavior-select-k", type=int, default=15000)
    parser.add_argument("--fusion-behavior-evidence-select-k", type=int, default=None)
    parser.add_argument("--fusion-consistency-select-k", type=int, default=0)
    parser.add_argument("--fusion-score-mode", choices=["classifier", "weighted_mean"], default="classifier")
    parser.add_argument(
        "--fusion-threshold-mode",
        choices=["train_f1", "train_fbeta", "train_recall_at_precision", "train_f1_recall_floor", "fixed"],
        default="train_recall_at_precision",
    )
    parser.add_argument("--fusion-min-train-precision", type=float, default=0.92)
    parser.add_argument("--fusion-min-train-recall", type=float, default=0.0)
    parser.add_argument("--fusion-threshold-beta", type=float, default=1.5)
    parser.add_argument("--fusion-fixed-threshold", type=float, default=0.5)
    parser.add_argument("--fusion-static-weight", type=float, default=0.4)
    parser.add_argument("--fusion-behavior-weight", type=float, default=0.4)
    parser.add_argument("--fusion-consistency-weight", type=float, default=0.2)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one fusion head is required.")
    return values


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    heads = parse_csv_strings(args.heads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = Path(args.evidence)
    behavior_by_id = {row["sample_id"]: row for row in read_jsonl(args.behaviors)}
    sample_rows = load_sample_index(evidence_path)
    sample_ids = [row["sample_id"] for row in sample_rows]
    labels = [row["label_int"] for row in sample_rows]
    labels_by_id = {row["sample_id"]: row["label_int"] for row in sample_rows}
    setup = {
        "evidence_path": str(evidence_path),
        "behavior_path": args.behaviors,
        "sample_count": len(sample_rows),
        "label_counts": dict(Counter(row["label"] for row in sample_rows)),
        "seeds": seeds,
        "heads": heads,
        "fusion_threshold_mode": args.fusion_threshold_mode,
        "fusion_min_train_precision": args.fusion_min_train_precision,
    }
    write_json(out_dir / "fusion_head_ablation_setup.json", setup)

    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for seed in seeds:
        print(f"[seed {seed}] start", flush=True)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split = make_split(sample_ids, labels, test_size=args.test_size, random_state=seed)
        write_json(seed_dir / "split.json", split)
        train_ids = split["train_sample_ids"]
        test_ids = split["test_sample_ids"]
        y_train = [labels_by_id[sample_id] for sample_id in train_ids]
        y_test = [labels_by_id[sample_id] for sample_id in test_ids]
        encoder_features_by_id = build_ablation_encoder_features(
            seed=seed,
            evidence_path=evidence_path,
            behavior_by_id=behavior_by_id,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            args=args,
            out_dir=seed_dir,
        )
        branch_score_cache: dict[tuple[str, str, int], tuple[list[float], list[float], dict[str, Any]]] = {}
        for head in heads:
            args.fusion_head_classifier = head
            for variant in ABLATION_VARIANTS:
                print(f"[seed {seed}][{head}] {variant['id']}", flush=True)
                metrics = run_encoder_fusion_variant(
                    seed=seed,
                    encoder_features_by_id=encoder_features_by_id,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    out_dir=seed_dir / head / variant["id"],
                    args=args,
                    variant_id=variant["id"],
                    display_name=variant["name"],
                    method_name=variant["id"],
                    branch_score_cache=branch_score_cache,
                )
                metrics.update(
                    {
                        "seed": seed,
                        "head": head,
                        "variant_id": variant["id"],
                        "variant_name": variant["name"],
                        "use_behavior_semantics": bool(variant["use_behavior_semantics"]),
                        "use_consistency": bool(variant["use_consistency"]),
                    }
                )
                rows.append(metrics)
                write_json(seed_dir / head / f"{variant['id']}_metrics.json", metrics)
                gc.collect()

    summary = summarize_by_head_variant(rows)
    validity = validity_by_head(summary)
    result = {
        **setup,
        "elapsed_seconds": time.perf_counter() - start_all,
        "rows": rows,
        "summary": summary,
        "validity": validity,
    }
    write_json(out_dir / "fusion_head_ablation_summary.json", result)
    write_summary_csv(out_dir / "fusion_head_ablation_summary.csv", summary, validity)


def build_ablation_encoder_features(
    seed: int,
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    wanted = set(train_ids) | set(test_ids)
    view_weight_spec = None
    if args.evidroid_dynamic_consistency:
        train_behavior_docs = [
            behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []}) for sample_id in train_ids
        ]
        view_weight_spec = learn_view_weight_spec(
            train_behavior_docs,
            y_train,
            mode="behavior",
            alpha=args.evidroid_alpha,
            min_label_samples=5,
            score_method="chi2",
            shrinkage_samples=args.evidroid_shrinkage_samples,
        )
        view_weight_spec["augment_fixed"] = bool(args.evidroid_augment_fixed_consistency)
        write_json(out_dir / "ablation_weight_spec.json", view_weight_spec)

    static_by_id: dict[str, dict[str, float]] = {}
    behavior_consistency_by_id: dict[str, dict[str, float]] = {}
    behavior_evidence_by_id: dict[str, dict[str, float]] = {}
    extras_by_variant: dict[str, dict[str, dict[str, float]]] = {"A1": {}, "A2": {}, "A3": {}}
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_match = SAMPLE_ID_RE.search(line)
            if not sample_match:
                continue
            sample_id = sample_match.group(1)
            if sample_id not in wanted:
                continue
            evidence_doc = json.loads(line)
            behavior_doc = behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []})
            parts = build_ablation_feature_parts(
                evidence_doc,
                behavior_doc,
                static_profile=args.ablation_static_profile,
                feature_version=args.ablation_feature_version,
                view_weight_spec=view_weight_spec,
            )
            static_by_id[sample_id] = parts["static"]
            behavior_consistency_by_id[sample_id] = parts["behavior_consistency"]
            extras_by_variant["A1"][sample_id] = parts["behavior"]
            extras_by_variant["A2"][sample_id] = parts["behavior_consistency"]
            behavior_evidence_by_id[sample_id] = compose_variant_features(parts["behavior"], parts["behavior_consistency"])
            extras_by_variant["A3"][sample_id] = behavior_evidence_by_id[sample_id]
            if len(static_by_id) % 2000 == 0:
                print(f"[seed {seed}] built {len(static_by_id)}/{len(wanted)} ablation rows", flush=True)
    missing = wanted - set(static_by_id)
    if missing:
        raise ValueError(f"Missing ablation features for {len(missing)} samples.")
    return {
        "static": static_by_id,
        "behavior": extras_by_variant["A1"],
        "consistency": extras_by_variant["A2"],
        "behavior_consistency": behavior_consistency_by_id,
        "behavior_evidence": behavior_evidence_by_id,
        "behavior_evidence_rf": behavior_evidence_by_id,
        "behavior_evidence_xgb": behavior_evidence_by_id,
    }


def summarize_by_head_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in sorted({row["head"] for row in rows}):
        summary[head] = {}
        for variant in [row["id"] for row in ABLATION_VARIANTS]:
            variant_rows = [row for row in rows if row["head"] == head and row["variant_id"] == variant]
            summary[head][variant] = summarize_rows(variant_rows, METRIC_KEYS)
    return summary


def validity_by_head(summary: dict[str, Any]) -> dict[str, Any]:
    validity: dict[str, Any] = {}
    for head, variants in summary.items():
        checks = {}
        for metric in METRIC_KEYS:
            a3 = variants["A3"][metric]["mean"]
            checks[metric] = all(a3 is not None and a3 >= variants[variant][metric]["mean"] for variant in ("A0", "A1", "A2"))
        validity[head] = {"a3_all_metrics_highest": all(checks.values()), "metric_checks": checks}
    return validity


def write_summary_csv(path: Path, summary: dict[str, Any], validity: dict[str, Any]) -> None:
    fields = ["head", "variant", "accuracy", "precision", "recall", "f1", "roc_auc", "a3_all_metrics_highest"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for head, variants in summary.items():
            for variant, metrics in variants.items():
                row = {"head": head, "variant": variant, "a3_all_metrics_highest": validity[head]["a3_all_metrics_highest"]}
                for metric in METRIC_KEYS:
                    row[metric] = metrics[metric]["mean"]
                writer.writerow(row)


if __name__ == "__main__":
    main()
