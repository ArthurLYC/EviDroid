from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from evidroid.baselines import (
    METHODS,
    TORCH_METHODS,
    build_deep_inputs,
    build_drebin_features,
    build_droidapiminer_features,
    convert_mamadroid_package_cache_to_family,
    load_mamadroid_cache,
    load_sample_index,
    require_torch,
    run_api_transformer,
    run_apppoet_like,
    run_prebuilt_sklearn_method,
    run_streamed_sklearn_method,
)
from evidroid.dynamic_weights import learn_view_weight_spec
from evidroid.features import build_evidroid_feature_dict
from evidroid.io_utils import read_jsonl, write_json
from evidroid.modeling import _adjust_test_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main SOTA comparison table for EviDroid.")
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--behaviors", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--mamadroid-cache", default="data/processed/mamadroid_features_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--out-dir", default="artifacts/optimized/sota_current")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--reuse-split", default="", help="Reuse an existing split.json for comparable reruns.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--select-k-best", type=int, default=20000)
    parser.add_argument("--drebin-classifier", default="linear_svm_sgd")
    parser.add_argument("--drebin-select-k-best", type=int, default=0)
    parser.add_argument("--droidapiminer-classifier", default="linear_svm_sgd")
    parser.add_argument("--droidapiminer-select-k-best", type=int, default=20000)
    parser.add_argument("--mamadroid-classifier", default="random_forest")
    parser.add_argument("--mamadroid-select-k-best", type=int, default=0)
    parser.add_argument("--mamadroid-abstraction", choices=["package", "family_from_package"], default="family_from_package")
    parser.add_argument("--max-api-len", type=int, default=256)
    parser.add_argument("--max-api-vocab", type=int, default=8000)
    parser.add_argument("--max-appoet-vocab", type=int, default=12000)
    parser.add_argument(
        "--apppoet-include-behavior",
        action="store_true",
        help="Include EviDroid behavior labels in AppPoet-like tokens. Disabled by default for original-baseline runs.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--evidroid-alpha", type=float, default=0.75)
    parser.add_argument("--evidroid-shrinkage-samples", type=float, default=10.0)
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    behavior_path = Path(args.behaviors)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if set(methods) & TORCH_METHODS:
        require_torch()
    if args.torch_threads > 0:
        require_torch().set_num_threads(args.torch_threads)

    behavior_by_id = {row["sample_id"]: row for row in read_jsonl(behavior_path)}
    sample_rows = load_sample_index(evidence_path)
    sample_ids = [row["sample_id"] for row in sample_rows]
    labels = [row["label_int"] for row in sample_rows]
    if args.reuse_split:
        labels_by_id = {row["sample_id"]: row["label_int"] for row in sample_rows}
        reused_split = json.loads(Path(args.reuse_split).read_text(encoding="utf-8"))
        train_ids = [sample_id for sample_id in reused_split["train_sample_ids"] if sample_id in labels_by_id]
        test_ids = [sample_id for sample_id in reused_split["test_sample_ids"] if sample_id in labels_by_id]
        y_train = [labels_by_id[sample_id] for sample_id in train_ids]
        y_test = [labels_by_id[sample_id] for sample_id in test_ids]
        split = {
            "train_sample_ids": train_ids,
            "test_sample_ids": test_ids,
            "random_state": reused_split.get("random_state", args.random_state),
            "test_size": reused_split.get("test_size", args.test_size),
            "reused_from": str(args.reuse_split),
            "dropped_train_ids": len(reused_split["train_sample_ids"]) - len(train_ids),
            "dropped_test_ids": len(reused_split["test_sample_ids"]) - len(test_ids),
        }
    else:
        test_size = _adjust_test_size(args.test_size, len(labels), len(set(labels)))
        train_ids, test_ids, y_train, y_test = train_test_split(
            sample_ids,
            labels,
            test_size=test_size,
            random_state=args.random_state,
            stratify=labels,
        )
        split = {
            "train_sample_ids": train_ids,
            "test_sample_ids": test_ids,
            "random_state": args.random_state,
            "test_size": test_size,
        }
    write_json(out_dir / "split.json", split)

    summary: dict[str, Any] = {
        "evidence_path": str(evidence_path),
        "behavior_path": str(behavior_path),
        "mamadroid_cache": str(args.mamadroid_cache),
        "sample_count": len(sample_rows),
        "label_counts": dict(Counter(row["label"] for row in sample_rows)),
        "methods": methods,
        "split": split,
        "settings": {
            "select_k_best": args.select_k_best,
            "drebin_classifier": args.drebin_classifier,
            "drebin_select_k_best": args.drebin_select_k_best,
            "droidapiminer_classifier": args.droidapiminer_classifier,
            "droidapiminer_select_k_best": args.droidapiminer_select_k_best,
            "mamadroid_classifier": args.mamadroid_classifier,
            "mamadroid_select_k_best": args.mamadroid_select_k_best,
            "mamadroid_abstraction": args.mamadroid_abstraction,
            "apppoet_include_behavior": args.apppoet_include_behavior,
            "max_api_len": args.max_api_len,
            "max_api_vocab": args.max_api_vocab,
            "max_appoet_vocab": args.max_appoet_vocab,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "metrics": [],
    }

    if "drebin" in methods:
        metrics = run_streamed_sklearn_method(
            evidence_path=evidence_path,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            feature_builder=build_drebin_features,
            classifier=args.drebin_classifier,
            select_k_best=args.drebin_select_k_best,
            random_state=args.random_state,
            out_dir=out_dir,
            model_name="drebin",
            display_name="Drebin (original)",
            feature_type="Original static feature groups",
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "drebin_metrics.json", metrics)
        gc.collect()

    if "droidapiminer" in methods:
        metrics = run_streamed_sklearn_method(
            evidence_path=evidence_path,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            feature_builder=build_droidapiminer_features,
            classifier=args.droidapiminer_classifier,
            select_k_best=args.droidapiminer_select_k_best,
            random_state=args.random_state,
            out_dir=out_dir,
            model_name="droidapiminer",
            display_name="DroidAPIMiner-style",
            feature_type="API and permission mining",
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "droidapiminer_metrics.json", metrics)
        gc.collect()

    if "mamadroid" in methods:
        mamadroid_features = load_mamadroid_cache(Path(args.mamadroid_cache))
        if args.mamadroid_abstraction == "family_from_package":
            mamadroid_features = convert_mamadroid_package_cache_to_family(mamadroid_features)
        mamadroid_display = "MaMaDroid (family)" if args.mamadroid_abstraction == "family_from_package" else "MaMaDroid (package)"
        metrics = run_prebuilt_sklearn_method(
            feature_by_id=mamadroid_features,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            classifier=args.mamadroid_classifier,
            select_k_best=args.mamadroid_select_k_best,
            random_state=args.random_state,
            out_dir=out_dir,
            model_name="mamadroid",
            display_name=mamadroid_display,
            feature_type="API Markov chain",
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "mamadroid_metrics.json", metrics)
        del mamadroid_features
        gc.collect()

    deep_cache: dict[str, dict[str, Any]] | None = None
    if "apppoet" in methods or "api_transformer" in methods:
        deep_cache = build_deep_inputs(
            evidence_path=evidence_path,
            behavior_by_id=behavior_by_id,
            wanted_ids=set(train_ids) | set(test_ids),
            max_api_len=args.max_api_len,
            include_behavior_in_apppoet=args.apppoet_include_behavior,
        )

    if "apppoet" in methods:
        assert deep_cache is not None
        metrics = run_apppoet_like(
            deep_cache=deep_cache,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            max_vocab=args.max_appoet_vocab,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.random_state,
            out_dir=out_dir,
            include_behavior=args.apppoet_include_behavior,
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "apppoet_metrics.json", metrics)
        gc.collect()

    if "api_transformer" in methods:
        assert deep_cache is not None
        metrics = run_api_transformer(
            deep_cache=deep_cache,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            max_vocab=args.max_api_vocab,
            max_len=args.max_api_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.random_state,
            out_dir=out_dir,
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "api_transformer_metrics.json", metrics)
        gc.collect()

    if "evidroid" in methods:
        train_behavior_docs = [behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []}) for sample_id in train_ids]
        weight_spec = learn_view_weight_spec(
            train_behavior_docs,
            y_train,
            mode="behavior",
            alpha=args.evidroid_alpha,
            min_label_samples=5,
            score_method="chi2",
            shrinkage_samples=args.evidroid_shrinkage_samples,
        )
        weight_spec["augment_fixed"] = True
        write_json(out_dir / "evidroid_weight_spec.json", weight_spec)
        metrics = run_streamed_sklearn_method(
            evidence_path=evidence_path,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            y_test=y_test,
            feature_builder=lambda evidence_doc: build_evidroid_feature_dict(
                evidence_doc,
                behavior_by_id.get(evidence_doc["sample_id"], {"sample_id": evidence_doc["sample_id"], "behaviors": []}),
                view_weight_spec=weight_spec,
            ),
            classifier="random_forest",
            select_k_best=args.select_k_best,
            random_state=args.random_state,
            out_dir=out_dir,
            model_name="evidroid",
            display_name="EviDroid",
            feature_type="Static + behavior + consistency + behavior-constrained evidence",
        )
        summary["metrics"].append(metrics)
        write_json(out_dir / "evidroid_metrics.json", metrics)
        gc.collect()

    write_json(out_dir / "sota_comparison_metrics.json", summary)
    write_markdown_report(out_dir / "sota_comparison_report.md", summary)


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SOTA Baseline Comparison",
        "",
        f"- Evidence: `{summary['evidence_path']}`",
        f"- Behaviors: `{summary['behavior_path']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Label counts: `{summary['label_counts']}`",
        "",
        "| Method | Feature Type | Backbone | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit seconds | Predict seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("metrics", []):
        lines.append(
            f"| {row.get('display_name', row.get('name'))} | {row.get('feature_type')} | {row.get('classifier')} | "
            f"{fmt(row.get('accuracy'))} | {fmt(row.get('precision'))} | {fmt(row.get('recall'))} | "
            f"{fmt(row.get('f1'))} | {fmt(row.get('roc_auc'))} | {fmt(row.get('fit_seconds'))} | "
            f"{fmt(row.get('predict_seconds'))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
