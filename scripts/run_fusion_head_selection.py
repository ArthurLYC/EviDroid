from __future__ import annotations

import argparse
import csv
import json
import statistics
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

from evidroid.dynamic_weights import learn_view_weight_spec
from evidroid.io_utils import read_jsonl, write_json
from evidroid.baselines import load_sample_index
from run_multiseed_experiments import (
    METRIC_KEYS,
    build_encoder_features_for_ids,
    make_split,
    run_encoder_fusion_variant,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple fusion-head classifiers on fixed EviDroid branches.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--behaviors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="42,2026,2027")
    parser.add_argument(
        "--heads",
        default="decision_tree,random_forest,extra_trees,gradient_boosting,adaboost,xgboost",
        help="Comma-separated classifier names accepted by make_classifier_pipeline.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--evidroid-alpha", type=float, default=0.75)
    parser.add_argument("--evidroid-shrinkage-samples", type=float, default=10.0)
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
    behavior_path = Path(args.behaviors)
    behavior_by_id = {row["sample_id"]: row for row in read_jsonl(behavior_path)}
    sample_rows = load_sample_index(evidence_path)
    sample_ids = [row["sample_id"] for row in sample_rows]
    labels = [row["label_int"] for row in sample_rows]
    labels_by_id = {row["sample_id"]: row["label_int"] for row in sample_rows}

    setup = {
        "evidence_path": str(evidence_path),
        "behavior_path": str(behavior_path),
        "sample_count": len(sample_rows),
        "label_counts": dict(Counter(row["label"] for row in sample_rows)),
        "seeds": seeds,
        "heads": heads,
        "test_size": args.test_size,
        "fusion_threshold_mode": args.fusion_threshold_mode,
        "fusion_min_train_precision": args.fusion_min_train_precision,
        "fusion_branch_classifiers": {
            "static": args.fusion_static_classifier,
            "behavior": args.fusion_behavior_classifier,
            "behavior_evidence_rf": args.fusion_behavior_classifier,
            "behavior_evidence_xgb": args.fusion_behavior_evidence_classifier,
        },
        "fusion_branch_select_k": {
            "static": args.fusion_static_select_k,
            "behavior": args.fusion_behavior_select_k,
            "behavior_evidence_rf": args.fusion_behavior_select_k,
            "behavior_evidence_xgb": args.fusion_behavior_evidence_select_k or args.fusion_behavior_select_k,
        },
    }
    write_json(out_dir / "fusion_head_selection_setup.json", setup)

    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for seed in seeds:
        seed_start = time.perf_counter()
        print(f"[seed {seed}] start", flush=True)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split = make_split(sample_ids, labels, test_size=args.test_size, random_state=seed)
        write_json(seed_dir / "split.json", split)
        train_ids = split["train_sample_ids"]
        test_ids = split["test_sample_ids"]
        y_train = [labels_by_id[sample_id] for sample_id in train_ids]
        y_test = [labels_by_id[sample_id] for sample_id in test_ids]

        train_behavior_docs = [
            behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []}) for sample_id in train_ids
        ]
        weight_spec = learn_view_weight_spec(
            train_behavior_docs,
            y_train,
            mode="behavior",
            alpha=args.evidroid_alpha,
            min_label_samples=5,
            score_method="chi2",
            shrinkage_samples=args.evidroid_shrinkage_samples,
        )
        weight_spec["augment_fixed"] = False
        write_json(seed_dir / "view_weight_spec.json", weight_spec)

        encoder_features_by_id = build_encoder_features_for_ids(
            evidence_path=evidence_path,
            behavior_by_id=behavior_by_id,
            wanted_ids=set(train_ids) | set(test_ids),
            static_profile=args.ablation_static_profile,
            feature_version=args.ablation_feature_version,
            view_weight_spec=weight_spec,
        )
        branch_score_cache: dict[tuple[str, str, int], tuple[list[float], list[float], dict[str, Any]]] = {}
        for head in heads:
            print(f"[seed {seed}] fusion head {head}", flush=True)
            args.fusion_head_classifier = head
            metrics = run_encoder_fusion_variant(
                seed=seed,
                encoder_features_by_id=encoder_features_by_id,
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                y_test=y_test,
                out_dir=seed_dir / head,
                args=args,
                variant_id="A3",
                display_name=head,
                method_name=head,
                branch_score_cache=branch_score_cache,
            )
            metrics["seed"] = seed
            metrics["head"] = head
            rows.append(metrics)
            write_json(seed_dir / f"{head}_metrics.json", metrics)
        print(f"[seed {seed}] finished in {time.perf_counter() - seed_start:.1f}s", flush=True)

    summary = summarize_by_head(rows)
    result = {
        **setup,
        "elapsed_seconds": time.perf_counter() - start_all,
        "rows": rows,
        "summary": summary,
    }
    write_json(out_dir / "fusion_head_selection_summary.json", result)
    write_summary_csv(out_dir / "fusion_head_selection_summary.csv", summary)
    write_by_seed_csv(out_dir / "fusion_head_selection_by_seed.csv", rows)


def summarize_by_head(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in sorted({row["head"] for row in rows}):
        head_rows = [row for row in rows if row["head"] == head]
        summary[head] = summarize_rows(head_rows, METRIC_KEYS)
        summary[head]["thresholds"] = [float(row["fusion_threshold"]) for row in head_rows]
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "head",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "n",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for head, metrics in summary.items():
            row = {"head": head, "n": metrics["n"]}
            for metric in METRIC_KEYS:
                row[f"{metric}_mean"] = metrics[metric]["mean"]
                row[f"{metric}_std"] = metrics[metric]["std"]
            writer.writerow(row)


def write_by_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["head", "seed", "accuracy", "precision", "recall", "f1", "roc_auc", "fusion_threshold"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    main()
