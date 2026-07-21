from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    decision_scores,
    iter_jsonl,
    load_mamadroid_cache,
    load_sample_index,
    require_torch,
    run_api_transformer,
    run_apppoet_like,
    run_prebuilt_sklearn_method,
    run_streamed_sklearn_method,
)
from evidroid.classifier_selection import feature_selection_metadata, make_classifier_pipeline
from evidroid.dynamic_weights import learn_view_weight_spec
from evidroid.features import build_ablation_feature_dict, build_ablation_feature_parts
from evidroid.io_utils import read_jsonl, write_json
from evidroid.modeling import _adjust_test_size


SAMPLE_ID_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
ABLATION_VARIANTS = [
    {
        "id": "A0",
        "name": "Static",
        "use_behavior_semantics": False,
        "use_consistency": False,
        "dynamic_weights": False,
    },
    {
        "id": "A1",
        "name": "Static+Behavior",
        "use_behavior_semantics": True,
        "use_consistency": False,
        "dynamic_weights": False,
    },
    {
        "id": "A2",
        "name": "Static+Consistency",
        "use_behavior_semantics": False,
        "use_consistency": True,
        "dynamic_weights": True,
    },
    {
        "id": "A3",
        "name": "Static+Behavior+Consistency",
        "use_behavior_semantics": True,
        "use_consistency": True,
        "dynamic_weights": True,
    },
]
FUSION_VARIANT_ENCODERS = {
    "A0": ("static",),
    "A1": ("static", "behavior"),
    "A2": ("static", "behavior_consistency"),
    "A3": ("static", "behavior", "behavior_evidence_rf", "behavior_evidence_xgb"),
}
FUSION_ENCODER_OFFSETS = {
    "static": 101,
    "behavior": 211,
    "consistency": 307,
    "behavior_consistency": 419,
    "behavior_evidence": 521,
    "behavior_evidence_rf": 521,
    "behavior_evidence_xgb": 631,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-random-seed EviDroid experiments.")
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--behaviors", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--mamadroid-cache", default="data/processed/mamadroid_features_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--mamadroid-abstraction", choices=["package", "family_from_package"], default="family_from_package")
    parser.add_argument("--out-dir", default="artifacts/optimized/multiseed_run")
    parser.add_argument("--seeds", default="42,2026,2027")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--select-k-best", type=int, default=20000)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Accepted for compatibility; dynamic-weight experiments were removed from the cleaned project.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--max-api-len", type=int, default=256)
    parser.add_argument("--max-api-vocab", type=int, default=8000)
    parser.add_argument("--max-appoet-vocab", type=int, default=12000)
    parser.add_argument(
        "--apppoet-include-behavior",
        action="store_true",
        help="Include EviDroid behavior labels in AppPoet-like tokens. Disabled by default for original-baseline runs.",
    )
    parser.add_argument("--evidroid-alpha", type=float, default=0.75)
    parser.add_argument("--evidroid-shrinkage-samples", type=float, default=10.0)
    parser.add_argument("--evidroid-dynamic-consistency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidroid-augment-fixed-consistency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ablation-static-profile", choices=["basic", "drebin", "compact"], default="compact")
    parser.add_argument("--ablation-feature-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--ablation-groupwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ablation-static-share", type=float, default=0.0008)
    parser.add_argument("--ablation-behavior-share", type=float, default=0.75)
    parser.add_argument("--a3-stacking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--a3-stack-source", choices=["extras", "full"], default="extras")
    parser.add_argument("--a3-stack-folds", type=int, default=5)
    parser.add_argument("--a3-stack-classifier", default="logistic_regression")
    parser.add_argument("--a3-stack-select-k", type=int, default=5000)
    parser.add_argument("--a3-prediction-mode", choices=["classifier", "stack", "blend"], default="classifier")
    parser.add_argument("--a3-stack-weight", type=float, default=0.5)
    parser.add_argument("--ablation-variants", default="A0,A1,A2,A3")
    parser.add_argument("--ablation-classifier", default="random_forest")
    parser.add_argument("--encoder-fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion-folds", type=int, default=5)
    parser.add_argument("--fusion-static-classifier", default="logistic_regression_sgd")
    parser.add_argument("--fusion-behavior-classifier", default="random_forest")
    parser.add_argument(
        "--fusion-behavior-evidence-classifier",
        default="xgboost",
        help="Classifier for the A3 behavior+consistency evidence branch.",
    )
    parser.add_argument("--fusion-consistency-classifier", default="xgboost")
    parser.add_argument("--fusion-head-classifier", default="logistic_regression")
    parser.add_argument("--fusion-score-mode", choices=["classifier", "weighted_mean"], default="classifier")
    parser.add_argument(
        "--fusion-threshold-mode",
        choices=["train_f1", "train_fbeta", "train_recall_at_precision", "train_f1_recall_floor", "fixed"],
        default="train_f1",
    )
    parser.add_argument("--fusion-fixed-threshold", type=float, default=0.5)
    parser.add_argument(
        "--fusion-threshold-beta",
        type=float,
        default=1.5,
        help="Beta for --fusion-threshold-mode train_fbeta; beta > 1 favors recall.",
    )
    parser.add_argument(
        "--fusion-min-train-precision",
        type=float,
        default=0.88,
        help="Minimum train precision for --fusion-threshold-mode train_recall_at_precision.",
    )
    parser.add_argument(
        "--fusion-min-train-recall",
        type=float,
        default=0.0,
        help="Minimum train recall for --fusion-threshold-mode train_f1_recall_floor.",
    )
    parser.add_argument("--fusion-static-weight", type=float, default=0.4)
    parser.add_argument("--fusion-behavior-weight", type=float, default=0.4)
    parser.add_argument("--fusion-consistency-weight", type=float, default=0.2)
    parser.add_argument("--fusion-static-select-k", type=int, default=12000)
    parser.add_argument("--fusion-behavior-select-k", type=int, default=15000)
    parser.add_argument(
        "--fusion-behavior-evidence-select-k",
        type=int,
        default=None,
        help="SelectK budget for the A3 behavior+consistency evidence branch; defaults to --fusion-behavior-select-k.",
    )
    parser.add_argument("--fusion-consistency-select-k", type=int, default=0)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Store per-sample test predictions in metric JSON files for failure-case analysis.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    ablation_variant_ids = parse_variant_ids(args.ablation_variants)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not args.skip_main and set(methods) & TORCH_METHODS:
        require_torch()
    if not args.skip_main and args.torch_threads > 0:
        require_torch().set_num_threads(args.torch_threads)
    grouped_select_k_best = (
        make_ablation_group_budget(
            args.select_k_best,
            static_share=args.ablation_static_share,
            behavior_share=args.ablation_behavior_share,
        )
        if args.ablation_groupwise and not args.encoder_fusion
        else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = Path(args.evidence)
    behavior_path = Path(args.behaviors)

    behavior_by_id = {row["sample_id"]: row for row in read_jsonl(behavior_path)}
    sample_rows = load_sample_index(evidence_path)
    sample_ids = [row["sample_id"] for row in sample_rows]
    labels = [row["label_int"] for row in sample_rows]
    labels_by_id = {row["sample_id"]: row["label_int"] for row in sample_rows}

    summary: dict[str, Any] = {
        "evidence_path": str(evidence_path),
        "behavior_path": str(behavior_path),
        "mamadroid_cache": str(args.mamadroid_cache),
        "mamadroid_abstraction": args.mamadroid_abstraction,
        "apppoet_include_behavior": args.apppoet_include_behavior,
        "sample_count": len(sample_rows),
        "label_counts": dict(Counter(row["label"] for row in sample_rows)),
        "seeds": seeds,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "select_k_best": args.select_k_best,
        "evidroid_dynamic_consistency": bool(args.evidroid_dynamic_consistency),
        "evidroid_augment_fixed_consistency": bool(args.evidroid_augment_fixed_consistency),
        "evidroid_alpha": args.evidroid_alpha,
        "evidroid_shrinkage_samples": args.evidroid_shrinkage_samples,
        "ablation_static_profile": args.ablation_static_profile,
        "ablation_feature_version": args.ablation_feature_version,
        "ablation_grouped_select_k_best": grouped_select_k_best,
        "ablation_a3_consistency_scope": "label_specific",
        "ablation_a3_stacking": bool(args.a3_stacking and not args.encoder_fusion),
        "ablation_a3_stack_source": args.a3_stack_source,
        "ablation_a3_stack_folds": args.a3_stack_folds,
        "ablation_a3_stack_classifier": args.a3_stack_classifier,
        "ablation_a3_stack_select_k": args.a3_stack_select_k,
        "ablation_a3_prediction_mode": args.a3_prediction_mode,
        "ablation_a3_stack_weight": args.a3_stack_weight,
        "ablation_variant_ids": ablation_variant_ids,
        "ablation_classifier": "encoder_fusion" if args.encoder_fusion else args.ablation_classifier,
        "encoder_fusion": args.encoder_fusion,
        "fusion_folds": args.fusion_folds,
        "fusion_branch_classifiers": {
            "static": args.fusion_static_classifier,
            "behavior": args.fusion_behavior_classifier,
            "consistency": args.fusion_consistency_classifier,
            "behavior_consistency": args.fusion_consistency_classifier,
            "behavior_evidence": fusion_behavior_evidence_classifier(args),
            "behavior_evidence_rf": args.fusion_behavior_classifier,
            "behavior_evidence_xgb": fusion_behavior_evidence_classifier(args),
        },
        "fusion_branch_select_k": {
            "static": args.fusion_static_select_k,
            "behavior": args.fusion_behavior_select_k,
            "consistency": args.fusion_consistency_select_k,
            "behavior_consistency": args.fusion_consistency_select_k,
            "behavior_evidence": fusion_behavior_evidence_select_k(args),
            "behavior_evidence_rf": args.fusion_behavior_select_k,
            "behavior_evidence_xgb": fusion_behavior_evidence_select_k(args),
        },
        "fusion_head_classifier": args.fusion_head_classifier,
        "fusion_score_mode": args.fusion_score_mode,
        "fusion_threshold_mode": args.fusion_threshold_mode,
        "fusion_fixed_threshold": args.fusion_fixed_threshold,
        "fusion_threshold_beta": args.fusion_threshold_beta,
        "fusion_min_train_precision": args.fusion_min_train_precision,
        "fusion_min_train_recall": args.fusion_min_train_recall,
        "fusion_score_weights": {
            "static": args.fusion_static_weight,
            "behavior": args.fusion_behavior_weight,
            "consistency": args.fusion_consistency_weight,
            "behavior_evidence": args.fusion_behavior_weight,
            "behavior_evidence_rf": args.fusion_behavior_weight,
            "behavior_evidence_xgb": args.fusion_behavior_weight,
        },
        "save_predictions": bool(args.save_predictions),
        "main_runs": [],
        "ablation_runs": [],
    }
    write_json(out_dir / "multiseed_setup.json", summary)

    mamadroid_features: dict[str, dict[str, float]] | None = None
    if not args.skip_main and "mamadroid" in methods:
        mamadroid_features = load_mamadroid_cache(Path(args.mamadroid_cache))
        if args.mamadroid_abstraction == "family_from_package":
            mamadroid_features = convert_mamadroid_package_cache_to_family(mamadroid_features)

    for seed in seeds:
        seed_start = time.perf_counter()
        print(f"[seed {seed}] starting", flush=True)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split = make_split(sample_ids, labels, test_size=args.test_size, random_state=seed)
        train_ids = split["train_sample_ids"]
        test_ids = split["test_sample_ids"]
        y_train = [labels_by_id[sample_id] for sample_id in train_ids]
        y_test = [labels_by_id[sample_id] for sample_id in test_ids]
        write_json(seed_dir / "split.json", split)

        if not args.skip_main:
            main_result = run_main_for_seed(
                seed=seed,
                evidence_path=evidence_path,
                behavior_by_id=behavior_by_id,
                mamadroid_features=mamadroid_features,
                methods=methods,
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                y_test=y_test,
                select_k_best=args.select_k_best,
                out_dir=seed_dir / "main",
                args=args,
            )
            summary["main_runs"].append(main_result)
            write_json(out_dir / "multiseed_partial.json", summary)
            gc.collect()

        if not args.skip_ablation:
            ablation_result = run_ablation_for_seed(
                seed=seed,
                evidence_path=evidence_path,
                behavior_by_id=behavior_by_id,
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                y_test=y_test,
                select_k_best=args.select_k_best,
                out_dir=seed_dir / "ablation",
                static_profile=args.ablation_static_profile,
                feature_version=args.ablation_feature_version,
                grouped_select_k_best=grouped_select_k_best,
                args=args,
                a3_stacking=args.a3_stacking,
                a3_stack_source=args.a3_stack_source,
                a3_stack_folds=args.a3_stack_folds,
                a3_stack_classifier=args.a3_stack_classifier,
                a3_stack_select_k=args.a3_stack_select_k,
                a3_prediction_mode=args.a3_prediction_mode,
                a3_stack_weight=args.a3_stack_weight,
                variant_ids=ablation_variant_ids,
                ablation_classifier=args.ablation_classifier,
            )
            summary["ablation_runs"].append(ablation_result)
            write_json(out_dir / "multiseed_partial.json", summary)
            gc.collect()

        print(f"[seed {seed}] finished in {time.perf_counter() - seed_start:.1f}s", flush=True)

    summary["main_summary"] = summarize_main(summary["main_runs"])
    summary["ablation_summary"] = summarize_ablation(summary["ablation_runs"])
    write_json(out_dir / "multiseed_summary.json", summary)
    write_markdown_report(out_dir / "multiseed_report.md", summary)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_variant_ids(raw: str) -> list[str]:
    ids = [item.strip().upper() for item in raw.split(",") if item.strip()]
    known = {row["id"] for row in ABLATION_VARIANTS}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}")
    return ids or [row["id"] for row in ABLATION_VARIANTS]


def fusion_behavior_evidence_classifier(args: argparse.Namespace) -> str:
    return str(args.fusion_behavior_evidence_classifier or args.fusion_behavior_classifier)


def fusion_behavior_evidence_select_k(args: argparse.Namespace) -> int:
    if args.fusion_behavior_evidence_select_k is None:
        return int(args.fusion_behavior_select_k)
    return int(args.fusion_behavior_evidence_select_k)


def make_ablation_group_budget(
    select_k_best: int,
    static_share: float = 0.6,
    behavior_share: float = 0.2,
) -> dict[str, int] | None:
    if select_k_best <= 0:
        return None
    if static_share <= 0 or behavior_share <= 0 or static_share + behavior_share >= 1.0:
        raise ValueError("Ablation group shares must be positive and leave room for consistency features.")
    static_k = max(1, int(round(select_k_best * static_share)))
    behavior_k = max(1, int(round(select_k_best * behavior_share)))
    consistency_k = max(1, select_k_best - static_k - behavior_k)
    return {
        "static": static_k,
        "behavior": behavior_k,
        "consistency": consistency_k,
        "other": 0,
    }


def make_split(sample_ids: list[str], labels: list[int], test_size: float, random_state: int) -> dict[str, Any]:
    adjusted = _adjust_test_size(test_size, len(labels), len(set(labels)))
    train_ids, test_ids, _y_train, _y_test = train_test_split(
        sample_ids,
        labels,
        test_size=adjusted,
        random_state=random_state,
        stratify=labels,
    )
    return {
        "train_sample_ids": train_ids,
        "test_sample_ids": test_ids,
        "random_state": random_state,
        "test_size": adjusted,
    }


def run_main_for_seed(
    seed: int,
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    mamadroid_features: dict[str, dict[str, float]] | None,
    methods: list[str],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    select_k_best: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if "drebin" in methods:
        print(f"[seed {seed}][main] drebin", flush=True)
        rows.append(
            cleanup_model(
                run_streamed_sklearn_method(
                    evidence_path=evidence_path,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    feature_builder=build_drebin_features,
                    classifier="linear_svm_sgd",
                    select_k_best=0,
                    random_state=seed,
                    out_dir=out_dir,
                    model_name="drebin",
                    display_name="Drebin (original)",
                    feature_type="Original static feature groups",
                )
            )
        )

    if "droidapiminer" in methods:
        print(f"[seed {seed}][main] droidapiminer", flush=True)
        rows.append(
            cleanup_model(
                run_streamed_sklearn_method(
                    evidence_path=evidence_path,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    feature_builder=build_droidapiminer_features,
                    classifier="linear_svm_sgd",
                    select_k_best=select_k_best,
                    random_state=seed,
                    out_dir=out_dir,
                    model_name="droidapiminer",
                    display_name="DroidAPIMiner-style",
                    feature_type="API and permission mining",
                )
            )
        )

    if "mamadroid" in methods:
        print(f"[seed {seed}][main] mamadroid", flush=True)
        if mamadroid_features is None:
            raise ValueError("MaMaDroid features were not loaded.")
        mamadroid_display = "MaMaDroid (family)" if args.mamadroid_abstraction == "family_from_package" else "MaMaDroid (package)"
        rows.append(
            cleanup_model(
                run_prebuilt_sklearn_method(
                    feature_by_id=mamadroid_features,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    classifier="random_forest",
                    select_k_best=0,
                    random_state=seed,
                    out_dir=out_dir,
                    model_name="mamadroid",
                    display_name=mamadroid_display,
                    feature_type="API Markov chain",
                )
            )
        )

    deep_cache: dict[str, dict[str, Any]] | None = None
    if "apppoet" in methods or "api_transformer" in methods:
        print(f"[seed {seed}][main] deep inputs", flush=True)
        deep_cache = build_deep_inputs(
            evidence_path=evidence_path,
            behavior_by_id=behavior_by_id,
            wanted_ids=set(train_ids) | set(test_ids),
            max_api_len=args.max_api_len,
            include_behavior_in_apppoet=args.apppoet_include_behavior,
        )

    if "apppoet" in methods:
        print(f"[seed {seed}][main] apppoet-like", flush=True)
        assert deep_cache is not None
        rows.append(
            cleanup_model(
                run_apppoet_like(
                    deep_cache=deep_cache,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    max_vocab=args.max_appoet_vocab,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    random_state=seed,
                    out_dir=out_dir,
                    include_behavior=args.apppoet_include_behavior,
                )
            )
        )

    if "api_transformer" in methods:
        print(f"[seed {seed}][main] api-transformer", flush=True)
        assert deep_cache is not None
        rows.append(
            cleanup_model(
                run_api_transformer(
                    deep_cache=deep_cache,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    max_vocab=args.max_api_vocab,
                    max_len=args.max_api_len,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    random_state=seed,
                    out_dir=out_dir,
                )
            )
        )

    if "evidroid" in methods:
        print(f"[seed {seed}][main] evidroid", flush=True)
        weight_spec = None
        weight_spec_path = None
        if args.evidroid_dynamic_consistency:
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
            weight_spec["augment_fixed"] = bool(args.evidroid_augment_fixed_consistency)
            weight_spec_path = out_dir / "evidroid_weight_spec.json"
            write_json(weight_spec_path, weight_spec)
        if args.encoder_fusion:
            rows.append(
                run_encoder_fusion_for_ids(
                    seed=seed,
                    evidence_path=evidence_path,
                    behavior_by_id=behavior_by_id,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    y_test=y_test,
                    out_dir=out_dir / "evidroid_encoder_fusion",
                    args=args,
                    variant_id="A3",
                    display_name="EviDroid",
                    method_name="evidroid",
                    view_weight_spec=weight_spec,
                    view_weight_spec_path=str(weight_spec_path) if weight_spec_path else None,
                )
            )
            result = {
                "seed": seed,
                "metrics": rows,
            }
            write_json(out_dir / "main_metrics.json", result)
            return result
        metrics = cleanup_model(
            run_streamed_sklearn_method(
                evidence_path=evidence_path,
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                y_test=y_test,
                feature_builder=lambda evidence_doc: build_ablation_feature_dict(
                    evidence_doc,
                    behavior_by_id.get(
                        evidence_doc["sample_id"], {"sample_id": evidence_doc["sample_id"], "behaviors": []}
                    ),
                    use_behavior_semantics=True,
                    use_consistency=True,
                    view_weight_spec=weight_spec,
                ),
                classifier="random_forest",
                select_k_best=select_k_best,
                random_state=seed,
                out_dir=out_dir,
                model_name="evidroid",
                display_name="EviDroid",
                feature_type="Static + Behavior + Consistency + BehaviorEvidence",
            )
        )
        metrics["dynamic_weights"] = bool(weight_spec)
        metrics["view_weight_spec_path"] = str(weight_spec_path) if weight_spec_path else None
        metrics["adaptive_consistency_augment_fixed"] = bool(weight_spec and weight_spec.get("augment_fixed"))
        rows.append(metrics)

    result = {
        "seed": seed,
        "metrics": rows,
    }
    write_json(out_dir / "main_metrics.json", result)
    return result


def cleanup_model(metrics: dict[str, Any]) -> dict[str, Any]:
    model_path = metrics.get("model_path")
    if model_path:
        path = Path(model_path)
        if path.exists():
            path.unlink()
        metrics["model_path"] = None
        metrics["model_deleted_after_run"] = True
    return metrics


def run_ablation_for_seed(
    seed: int,
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    select_k_best: int,
    out_dir: Path,
    static_profile: str,
    feature_version: str,
    grouped_select_k_best: dict[str, int] | None,
    args: argparse.Namespace,
    a3_stacking: bool,
    a3_stack_source: str,
    a3_stack_folds: int,
    a3_stack_classifier: str,
    a3_stack_select_k: int,
    a3_prediction_mode: str,
    a3_stack_weight: float,
    variant_ids: list[str],
    ablation_classifier: str,
) -> dict[str, Any]:
    print(f"[seed {seed}][ablation] building component features", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(train_ids) | set(test_ids)
    dynamic_consistency_variants = [
        row for row in ABLATION_VARIANTS if row["id"] in variant_ids and row["use_consistency"]
    ]
    view_weight_spec = None
    view_weight_spec_path = None
    if args.evidroid_dynamic_consistency and dynamic_consistency_variants:
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
        view_weight_spec_path = out_dir / "ablation_weight_spec.json"
        write_json(view_weight_spec_path, view_weight_spec)
    static_by_id: dict[str, dict[str, float]] = {}
    behavior_consistency_by_id: dict[str, dict[str, float]] = {}
    behavior_evidence_by_id: dict[str, dict[str, float]] = {}
    extras_by_variant: dict[str, dict[str, dict[str, float]]] = {row["id"]: {} for row in ABLATION_VARIANTS if row["id"] != "A0"}
    start = time.perf_counter()
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
                static_profile=static_profile,
                feature_version=feature_version,
                view_weight_spec=view_weight_spec,
            )
            static_features = parts["static"]
            static_by_id[sample_id] = static_features
            behavior_consistency_by_id[sample_id] = parts["behavior_consistency"]
            extras_by_variant["A1"][sample_id] = parts["behavior"]
            extras_by_variant["A2"][sample_id] = parts["behavior_consistency"]
            a3_extra_features = compose_variant_features(
                parts["behavior"],
                parts["behavior_consistency"],
            )
            extras_by_variant["A3"][sample_id] = a3_extra_features
            behavior_evidence_by_id[sample_id] = a3_extra_features
            if len(static_by_id) % 2000 == 0:
                print(f"[seed {seed}][ablation] built {len(static_by_id)}/{len(wanted)} rows", flush=True)
    missing = wanted - set(static_by_id)
    if missing:
        raise ValueError(f"Missing ablation features for {len(missing)} samples.")
    if args.encoder_fusion:
        encoder_features_by_id = {
            "static": static_by_id,
            "behavior": extras_by_variant["A1"],
            "consistency": extras_by_variant["A2"],
            "behavior_consistency": behavior_consistency_by_id,
            "behavior_evidence": behavior_evidence_by_id,
            "behavior_evidence_rf": behavior_evidence_by_id,
            "behavior_evidence_xgb": behavior_evidence_by_id,
        }
        rows = []
        branch_score_cache: dict[tuple[str, str, int], tuple[list[float], list[float], dict[str, Any]]] = {}
        for variant in ABLATION_VARIANTS:
            if variant["id"] not in variant_ids:
                continue
            print(f"[seed {seed}][ablation] encoder fusion {variant['id']}", flush=True)
            metrics = run_encoder_fusion_variant(
                seed=seed,
                encoder_features_by_id=encoder_features_by_id,
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                y_test=y_test,
                out_dir=out_dir / variant["id"],
                args=args,
                variant_id=variant["id"],
                display_name=variant["name"],
                method_name=variant["id"],
                branch_score_cache=branch_score_cache,
            )
            metrics.update(
                {
                    "seed": seed,
                    "variant_id": variant["id"],
                    "variant_name": variant["name"],
                    "use_behavior_semantics": bool(variant["use_behavior_semantics"]),
                    "use_consistency": bool(variant["use_consistency"]),
                    "dynamic_weights": bool(view_weight_spec and variant["use_consistency"]),
                    "view_weight_spec_path": str(view_weight_spec_path) if (view_weight_spec and variant["use_consistency"]) else None,
                    "adaptive_consistency_augment_fixed": bool(
                        view_weight_spec
                        and variant["use_consistency"]
                        and view_weight_spec.get("augment_fixed")
                    ),
                    "static_profile": static_profile,
                    "feature_version": feature_version,
                }
            )
            rows.append(metrics)
            write_json(out_dir / f"{variant['id']}_metrics.json", metrics)
            gc.collect()
        result = {
            "seed": seed,
            "feature_build_seconds": float(time.perf_counter() - start),
            "metrics": rows,
        }
        write_json(out_dir / "ablation_metrics.json", result)
        return result
    stack_result: dict[str, Any] | None = None
    if a3_stacking:
        stack_result = add_a3_stacked_features(
            seed=seed,
            static_by_id=static_by_id,
            extras_by_variant=extras_by_variant,
            train_ids=train_ids,
            test_ids=test_ids,
            y_train=y_train,
            source=a3_stack_source,
            folds=a3_stack_folds,
            classifier_name=a3_stack_classifier,
            select_k_best=a3_stack_select_k,
        )
        write_json(out_dir / "A3_stack_metadata.json", stack_result["metadata"])

    rows = []
    for variant in ABLATION_VARIANTS:
        if variant["id"] not in variant_ids:
            continue
        print(f"[seed {seed}][ablation] training {variant['id']}", flush=True)
        x_train = [
            compose_variant_features(static_by_id[sample_id], extras_by_variant.get(variant["id"], {}).get(sample_id))
            for sample_id in train_ids
        ]
        x_test = [
            compose_variant_features(static_by_id[sample_id], extras_by_variant.get(variant["id"], {}).get(sample_id))
            for sample_id in test_ids
        ]
        model = make_classifier_pipeline(
            ablation_classifier,
            random_state=seed,
            select_k_best=0 if grouped_select_k_best else select_k_best,
            grouped_select_k_best=grouped_select_k_best,
        )
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        predictions = model.predict(x_test)
        scores = decision_scores(model, x_test)
        prediction_mode = "classifier"
        if variant["id"] == "A3" and stack_result and a3_prediction_mode in {"stack", "blend"}:
            stack_scores = [float(stack_result["test_scores_by_id"][sample_id]) for sample_id in test_ids]
            threshold = float(stack_result["metadata"]["best_oof_threshold"])
            if a3_prediction_mode == "stack":
                scores = stack_scores
                predictions = [1 if score >= threshold else 0 for score in scores]
                prediction_mode = "stack"
            else:
                weight = min(1.0, max(0.0, float(a3_stack_weight)))
                scores = [
                    (weight * float(stack_score)) + ((1.0 - weight) * float(classifier_score))
                    for stack_score, classifier_score in zip(stack_scores, _normalize_scores(scores))
                ]
                predictions = [1 if score >= 0.5 else 0 for score in scores]
                prediction_mode = f"blend_{weight:.2f}"
        predict_seconds = time.perf_counter() - predict_start
        metrics = evaluate_metric_row(
            y_test=y_test,
            predictions=predictions,
            scores=scores,
            test_ids=test_ids,
            save_predictions=args.save_predictions,
            extra={
                "seed": seed,
                "variant_id": variant["id"],
                "variant_name": variant["name"],
                "use_behavior_semantics": bool(variant["use_behavior_semantics"]),
                "use_consistency": bool(variant["use_consistency"]),
                "dynamic_weights": bool(view_weight_spec and variant["use_consistency"]),
                "view_weight_spec_path": str(view_weight_spec_path) if (view_weight_spec and variant["use_consistency"]) else None,
                "adaptive_consistency_augment_fixed": bool(
                    view_weight_spec
                    and variant["use_consistency"]
                    and view_weight_spec.get("augment_fixed")
                ),
                "classifier": ablation_classifier,
                "static_profile": static_profile,
                "feature_version": feature_version,
                "grouped_select_k_best": grouped_select_k_best,
                "a3_stacking": bool(a3_stacking and variant["id"] == "A3"),
                "a3_prediction_mode": prediction_mode,
                "a3_stack_metadata": stack_result["metadata"] if (stack_result and variant["id"] == "A3") else None,
                **feature_selection_metadata(model),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
            },
        )
        rows.append(metrics)
        write_json(out_dir / f"{variant['id']}_metrics.json", metrics)
        del x_train, x_test, model
        gc.collect()

    result = {
        "seed": seed,
        "feature_build_seconds": float(time.perf_counter() - start),
        "metrics": rows,
    }
    write_json(out_dir / "ablation_metrics.json", result)
    return result


def compose_variant_features(
    static_features: dict[str, float],
    extra_features: dict[str, float] | None,
) -> dict[str, float]:
    if not extra_features:
        return dict(static_features)
    features = dict(static_features)
    features.update(extra_features)
    return features


def run_encoder_fusion_for_ids(
    seed: int,
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    out_dir: Path,
    args: argparse.Namespace,
    variant_id: str,
    display_name: str,
    method_name: str,
    view_weight_spec: dict[str, Any] | None = None,
    view_weight_spec_path: str | None = None,
) -> dict[str, Any]:
    wanted = set(train_ids) | set(test_ids)
    encoder_features_by_id = build_encoder_features_for_ids(
        evidence_path=evidence_path,
        behavior_by_id=behavior_by_id,
        wanted_ids=wanted,
        static_profile=args.ablation_static_profile,
        feature_version=args.ablation_feature_version,
        view_weight_spec=view_weight_spec,
    )
    metrics = run_encoder_fusion_variant(
        seed=seed,
        encoder_features_by_id=encoder_features_by_id,
        train_ids=train_ids,
        test_ids=test_ids,
        y_train=y_train,
        y_test=y_test,
        out_dir=out_dir,
        args=args,
        variant_id=variant_id,
        display_name=display_name,
        method_name=method_name,
    )
    if view_weight_spec is not None:
        metrics["dynamic_weights"] = True
        metrics["view_weight_spec_path"] = view_weight_spec_path
        metrics["adaptive_consistency_augment_fixed"] = bool(view_weight_spec.get("augment_fixed"))
        write_json(out_dir / "encoder_fusion_metrics.json", metrics)
    return metrics


def build_encoder_features_for_ids(
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    wanted_ids: set[str],
    static_profile: str,
    feature_version: str,
    view_weight_spec: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    encoder_features_by_id: dict[str, dict[str, dict[str, float]]] = {
        "static": {},
        "behavior": {},
        "consistency": {},
        "behavior_consistency": {},
        "behavior_evidence": {},
        "behavior_evidence_rf": {},
        "behavior_evidence_xgb": {},
    }
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_match = SAMPLE_ID_RE.search(line)
            if not sample_match:
                continue
            sample_id = sample_match.group(1)
            if sample_id not in wanted_ids:
                continue
            evidence_doc = json.loads(line)
            behavior_doc = behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []})
            parts = build_ablation_feature_parts(
                evidence_doc,
                behavior_doc,
                static_profile=static_profile,
                feature_version=feature_version,
                view_weight_spec=view_weight_spec,
            )
            encoder_features_by_id["static"][sample_id] = parts["static"]
            encoder_features_by_id["behavior"][sample_id] = parts["behavior"]
            encoder_features_by_id["consistency"][sample_id] = parts["consistency"]
            encoder_features_by_id["behavior_consistency"][sample_id] = parts["behavior_consistency"]
            encoder_features_by_id["behavior_evidence"][sample_id] = compose_variant_features(
                parts["behavior"],
                parts["behavior_consistency"],
            )
            encoder_features_by_id["behavior_evidence_rf"][sample_id] = encoder_features_by_id["behavior_evidence"][
                sample_id
            ]
            encoder_features_by_id["behavior_evidence_xgb"][sample_id] = encoder_features_by_id["behavior_evidence"][
                sample_id
            ]
            if len(encoder_features_by_id["static"]) % 2000 == 0:
                print(f"[fusion] built {len(encoder_features_by_id['static'])}/{len(wanted_ids)} rows", flush=True)
    missing = wanted_ids - set(encoder_features_by_id["static"])
    if missing:
        raise ValueError(f"Missing encoder fusion features for {len(missing)} samples.")
    return encoder_features_by_id


def run_encoder_fusion_variant(
    seed: int,
    encoder_features_by_id: dict[str, dict[str, dict[str, float]]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    out_dir: Path,
    args: argparse.Namespace,
    variant_id: str,
    display_name: str,
    method_name: str,
    branch_score_cache: dict[tuple[str, str, int], tuple[list[float], list[float], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoders = FUSION_VARIANT_ENCODERS[variant_id]
    branch_train_scores: dict[str, list[float]] = {}
    branch_test_scores: dict[str, list[float]] = {}
    branch_metadata: dict[str, Any] = {}
    fit_seconds = 0.0
    for encoder_name in encoders:
        classifier_name = {
            "static": args.fusion_static_classifier,
            "behavior": args.fusion_behavior_classifier,
            "consistency": args.fusion_consistency_classifier,
            "behavior_consistency": args.fusion_consistency_classifier,
            "behavior_evidence": fusion_behavior_evidence_classifier(args),
            "behavior_evidence_rf": args.fusion_behavior_classifier,
            "behavior_evidence_xgb": fusion_behavior_evidence_classifier(args),
        }[encoder_name]
        select_k_best = {
            "static": args.fusion_static_select_k,
            "behavior": args.fusion_behavior_select_k,
            "consistency": args.fusion_consistency_select_k,
            "behavior_consistency": args.fusion_consistency_select_k,
            "behavior_evidence": fusion_behavior_evidence_select_k(args),
            "behavior_evidence_rf": args.fusion_behavior_select_k,
            "behavior_evidence_xgb": fusion_behavior_evidence_select_k(args),
        }[encoder_name]
        cache_key = (encoder_name, classifier_name, int(select_k_best))
        start = time.perf_counter()
        if branch_score_cache is not None and cache_key in branch_score_cache:
            print(f"[seed {seed}][fusion {variant_id}] reuse {encoder_name} branch", flush=True)
            train_scores, test_scores, metadata = branch_score_cache[cache_key]
        else:
            print(
                f"[seed {seed}][fusion {variant_id}] train {encoder_name} branch "
                f"({classifier_name}, select_k={select_k_best})",
                flush=True,
            )
            train_scores, test_scores, metadata = encoder_branch_scores(
                seed=seed + FUSION_ENCODER_OFFSETS[encoder_name],
                encoder_name=encoder_name,
                features_by_id=encoder_features_by_id[encoder_name],
                train_ids=train_ids,
                test_ids=test_ids,
                y_train=y_train,
                args=args,
            )
            metadata["branch_fit_seconds"] = float(time.perf_counter() - start)
            if branch_score_cache is not None:
                branch_score_cache[cache_key] = (train_scores, test_scores, metadata)
            print(
                f"[seed {seed}][fusion {variant_id}] done {encoder_name} branch "
                f"in {metadata['branch_fit_seconds']:.1f}s",
                flush=True,
            )
            fit_seconds += float(metadata["branch_fit_seconds"])
        branch_train_scores[encoder_name] = train_scores
        branch_test_scores[encoder_name] = test_scores
        branch_metadata[encoder_name] = metadata

    predict_start = time.perf_counter()
    fusion_weights: dict[str, float] | None = None
    if args.fusion_score_mode == "weighted_mean":
        fusion_weights = normalized_fusion_weights(args, encoders)
        print(f"[seed {seed}][fusion {variant_id}] weighted score fusion {fusion_weights}", flush=True)
        train_scores = weighted_fusion_scores(branch_train_scores, encoders, len(train_ids), fusion_weights)
        scores = weighted_fusion_scores(branch_test_scores, encoders, len(test_ids), fusion_weights)
    else:
        fusion_train_rows = build_fusion_score_rows(branch_train_scores, encoders, len(train_ids))
        fusion_test_rows = build_fusion_score_rows(branch_test_scores, encoders, len(test_ids))
        fusion_model = make_classifier_pipeline(args.fusion_head_classifier, random_state=seed, select_k_best=0)
        fusion_start = time.perf_counter()
        print(f"[seed {seed}][fusion {variant_id}] train fusion head", flush=True)
        fusion_model.fit(fusion_train_rows, y_train)
        fit_seconds += time.perf_counter() - fusion_start
        train_scores = _normalize_scores(decision_scores(fusion_model, fusion_train_rows))
        scores = _normalize_scores(decision_scores(fusion_model, fusion_test_rows))
    threshold_info = _select_threshold(
        y_train,
        train_scores,
        mode=args.fusion_threshold_mode,
        fixed_threshold=float(args.fusion_fixed_threshold),
        beta=float(args.fusion_threshold_beta),
        min_precision=float(args.fusion_min_train_precision),
        min_recall=float(args.fusion_min_train_recall),
    )
    best_threshold, train_best_f1 = _best_threshold(y_train, train_scores)
    threshold = float(threshold_info["threshold"])
    predictions = [1 if score >= threshold else 0 for score in scores]
    predict_seconds = time.perf_counter() - predict_start
    metrics = evaluate_metric_row(
        y_test=y_test,
        predictions=predictions,
        scores=scores,
        test_ids=test_ids,
        save_predictions=args.save_predictions,
        extra={
            "method": method_name,
            "name": display_name,
            "display_name": display_name,
            "status": "ok",
            "classifier": "encoder_fusion",
            "fusion_head": args.fusion_head_classifier,
            "fusion_score_mode": args.fusion_score_mode,
            "fusion_weights": fusion_weights,
            "fusion_encoders": list(encoders),
            "fusion_threshold": float(threshold),
            "fusion_train_best_threshold": float(best_threshold),
            "fusion_threshold_mode": args.fusion_threshold_mode,
            "fusion_train_best_f1": float(train_best_f1),
            "fusion_threshold_selection": threshold_info,
            "branch_metadata": branch_metadata,
            "feature_count": int(sum(row.get("feature_count", 0) for row in branch_metadata.values())),
            "selected_feature_count": int(sum(row.get("selected_feature_count", 0) for row in branch_metadata.values())),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        },
    )
    write_json(out_dir / "encoder_fusion_metrics.json", metrics)
    return metrics

def encoder_branch_scores(
    seed: int,
    encoder_name: str,
    features_by_id: dict[str, dict[str, float]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], dict[str, Any]]:
    classifier_name = {
        "static": args.fusion_static_classifier,
        "behavior": args.fusion_behavior_classifier,
        "consistency": args.fusion_consistency_classifier,
        "behavior_consistency": args.fusion_consistency_classifier,
        "behavior_evidence": fusion_behavior_evidence_classifier(args),
        "behavior_evidence_rf": args.fusion_behavior_classifier,
        "behavior_evidence_xgb": fusion_behavior_evidence_classifier(args),
    }[encoder_name]
    select_k_best = {
        "static": args.fusion_static_select_k,
        "behavior": args.fusion_behavior_select_k,
        "consistency": args.fusion_consistency_select_k,
        "behavior_consistency": args.fusion_consistency_select_k,
        "behavior_evidence": fusion_behavior_evidence_select_k(args),
        "behavior_evidence_rf": args.fusion_behavior_select_k,
        "behavior_evidence_xgb": fusion_behavior_evidence_select_k(args),
    }[encoder_name]
    x_train = [features_by_id[sample_id] for sample_id in train_ids]
    x_test = [features_by_id[sample_id] for sample_id in test_ids]
    n_splits = max(2, min(int(args.fusion_folds), min(Counter(y_train).values())))
    oof_scores = [0.5 for _ in train_ids]
    y_array = np.asarray(y_train)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_feature_counts = []
    fold_selected_counts = []
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_array), start=1):
        print(f"[fusion {encoder_name}] fold {fold_idx}/{n_splits}", flush=True)
        model = make_classifier_pipeline(classifier_name, random_state=seed + fold_idx, select_k_best=select_k_best)
        model.fit([x_train[idx] for idx in fit_idx], [y_train[idx] for idx in fit_idx])
        scores = _normalize_scores(decision_scores(model, [x_train[idx] for idx in val_idx]))
        for idx, score in zip(val_idx, scores):
            oof_scores[int(idx)] = float(score)
        meta = feature_selection_metadata(model)
        fold_feature_counts.append(int(meta["feature_count"]))
        fold_selected_counts.append(int(meta["selected_feature_count"]))
        del model
        gc.collect()

    final_model = make_classifier_pipeline(classifier_name, random_state=seed, select_k_best=select_k_best)
    final_model.fit(x_train, y_train)
    test_scores = _normalize_scores(decision_scores(final_model, x_test))
    final_meta = feature_selection_metadata(final_model)
    oof_predictions = [1 if score >= 0.5 else 0 for score in oof_scores]
    metadata = {
        "encoder": encoder_name,
        "classifier": classifier_name,
        "select_k_best": select_k_best,
        "folds": n_splits,
        "feature_count": int(final_meta["feature_count"]),
        "selected_feature_count": int(final_meta["selected_feature_count"]),
        "fold_feature_count_mean": float(statistics.mean(fold_feature_counts)) if fold_feature_counts else 0.0,
        "fold_selected_feature_count_mean": float(statistics.mean(fold_selected_counts)) if fold_selected_counts else 0.0,
        "train_oof_accuracy": float(accuracy_score(y_train, oof_predictions)),
        "train_oof_precision": float(precision_score(y_train, oof_predictions, zero_division=0)),
        "train_oof_recall": float(recall_score(y_train, oof_predictions, zero_division=0)),
        "train_oof_f1": float(f1_score(y_train, oof_predictions, zero_division=0)),
    }
    try:
        metadata["train_oof_roc_auc"] = float(roc_auc_score(y_train, oof_scores))
    except ValueError:
        metadata["train_oof_roc_auc"] = None
    del final_model
    gc.collect()
    return oof_scores, test_scores, metadata


def build_fusion_score_rows(
    scores_by_encoder: dict[str, list[float]],
    encoders: tuple[str, ...],
    row_count: int,
) -> list[dict[str, float]]:
    rows = []
    for idx in range(row_count):
        row: dict[str, float] = {}
        for encoder in encoders:
            row[f"fusion::{encoder}"] = float(scores_by_encoder[encoder][idx])
            row[f"fusion::{encoder}_confidence"] = abs(float(scores_by_encoder[encoder][idx]) - 0.5) * 2.0
        encoder_scores = [row[f"fusion::{encoder}"] for encoder in encoders]
        if encoder_scores:
            row["fusion::score_mean"] = float(statistics.mean(encoder_scores))
            row["fusion::score_max"] = float(max(encoder_scores))
            row["fusion::score_min"] = float(min(encoder_scores))
            row["fusion::score_range"] = float(max(encoder_scores) - min(encoder_scores))
        for left_idx, left in enumerate(encoders):
            for right in encoders[left_idx + 1 :]:
                pair_key = f"{left}__{right}"
                left_score = row[f"fusion::{left}"]
                right_score = row[f"fusion::{right}"]
                row[f"fusion::pair::{pair_key}::product"] = left_score * right_score
                row[f"fusion::pair::{pair_key}::diff"] = left_score - right_score
                row[f"fusion::pair::{pair_key}::abs_diff"] = abs(left_score - right_score)
        if "static" in encoders and "behavior" in encoders:
            row["fusion::static_x_behavior"] = row["fusion::static"] * row["fusion::behavior"]
            row["fusion::behavior_minus_static"] = row["fusion::behavior"] - row["fusion::static"]
            row["fusion::abs_behavior_static"] = abs(row["fusion::behavior"] - row["fusion::static"])
        if "static" in encoders and "consistency" in encoders:
            row["fusion::static_x_consistency"] = row["fusion::static"] * row["fusion::consistency"]
            row["fusion::consistency_minus_static"] = row["fusion::consistency"] - row["fusion::static"]
            row["fusion::abs_consistency_static"] = abs(row["fusion::consistency"] - row["fusion::static"])
        if "behavior" in encoders and "consistency" in encoders:
            row["fusion::behavior_x_consistency"] = row["fusion::behavior"] * row["fusion::consistency"]
            row["fusion::behavior_minus_consistency"] = row["fusion::behavior"] - row["fusion::consistency"]
            row["fusion::abs_behavior_consistency"] = abs(row["fusion::behavior"] - row["fusion::consistency"])
        rows.append(row)
    return rows


def normalized_fusion_weights(args: argparse.Namespace, encoders: tuple[str, ...]) -> dict[str, float]:
    raw_weights = {
        "static": max(0.0, float(args.fusion_static_weight)),
        "behavior": max(0.0, float(args.fusion_behavior_weight)),
        "consistency": max(0.0, float(args.fusion_consistency_weight)),
        "behavior_consistency": max(0.0, float(args.fusion_consistency_weight)),
        "behavior_evidence": max(0.0, float(args.fusion_behavior_weight)),
        "behavior_evidence_rf": max(0.0, float(args.fusion_behavior_weight)),
        "behavior_evidence_xgb": max(0.0, float(args.fusion_behavior_weight)),
    }
    total = sum(raw_weights[encoder] for encoder in encoders)
    if total <= 0:
        return {encoder: 1.0 / len(encoders) for encoder in encoders}
    return {encoder: raw_weights[encoder] / total for encoder in encoders}


def weighted_fusion_scores(
    scores_by_encoder: dict[str, list[float]],
    encoders: tuple[str, ...],
    row_count: int,
    weights: dict[str, float],
) -> list[float]:
    scores = []
    for idx in range(row_count):
        score = 0.0
        for encoder in encoders:
            score += float(weights[encoder]) * float(scores_by_encoder[encoder][idx])
        scores.append(float(score))
    return scores


def add_a3_stacked_features(
    seed: int,
    static_by_id: dict[str, dict[str, float]],
    extras_by_variant: dict[str, dict[str, dict[str, float]]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    source: str,
    folds: int,
    classifier_name: str,
    select_k_best: int,
) -> dict[str, Any]:
    print(f"[seed {seed}][ablation] fitting A3 stacked behavior risk ({source}, {classifier_name})", flush=True)
    if source not in {"extras", "full"}:
        raise ValueError(f"Unknown A3 stack source: {source}")
    a3_extras = extras_by_variant.get("A3", {})

    def row_for(sample_id: str) -> dict[str, float]:
        extra = a3_extras.get(sample_id, {})
        if source == "extras":
            return dict(extra)
        return compose_variant_features(static_by_id[sample_id], extra)

    x_train_stack = [row_for(sample_id) for sample_id in train_ids]
    x_test_stack = [row_for(sample_id) for sample_id in test_ids]
    n_splits = max(2, min(int(folds), min(Counter(y_train).values())))
    oof_scores = [0.5 for _ in train_ids]
    y_array = np.asarray(y_train)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_train_stack, y_array), start=1):
        model = make_classifier_pipeline(
            classifier_name,
            random_state=seed + fold_idx,
            select_k_best=select_k_best,
        )
        model.fit([x_train_stack[idx] for idx in fit_idx], [y_train[idx] for idx in fit_idx])
        scores = _normalize_scores(decision_scores(model, [x_train_stack[idx] for idx in val_idx]))
        for idx, score in zip(val_idx, scores):
            oof_scores[int(idx)] = float(score)
        del model
        gc.collect()

    final_model = make_classifier_pipeline(classifier_name, random_state=seed, select_k_best=select_k_best)
    final_model.fit(x_train_stack, y_train)
    test_scores = _normalize_scores(decision_scores(final_model, x_test_stack))
    del final_model

    for sample_id, score in zip(train_ids, oof_scores):
        _add_stack_score_features(a3_extras[sample_id], float(score))
    for sample_id, score in zip(test_ids, test_scores):
        _add_stack_score_features(a3_extras[sample_id], float(score))

    oof_predictions = [1 if score >= 0.5 else 0 for score in oof_scores]
    best_threshold, best_f1 = _best_threshold(y_train, oof_scores)
    best_predictions = [1 if score >= best_threshold else 0 for score in oof_scores]
    metadata: dict[str, Any] = {
        "source": source,
        "folds": n_splits,
        "classifier": classifier_name,
        "select_k_best": select_k_best,
        "best_oof_threshold": float(best_threshold),
        "best_oof_f1": float(best_f1),
        "train_oof_accuracy": float(accuracy_score(y_train, oof_predictions)),
        "train_oof_precision": float(precision_score(y_train, oof_predictions, zero_division=0)),
        "train_oof_recall": float(recall_score(y_train, oof_predictions, zero_division=0)),
        "train_oof_f1": float(f1_score(y_train, oof_predictions, zero_division=0)),
        "best_oof_accuracy": float(accuracy_score(y_train, best_predictions)),
        "best_oof_precision": float(precision_score(y_train, best_predictions, zero_division=0)),
        "best_oof_recall": float(recall_score(y_train, best_predictions, zero_division=0)),
    }
    try:
        metadata["train_oof_roc_auc"] = float(roc_auc_score(y_train, oof_scores))
    except ValueError:
        metadata["train_oof_roc_auc"] = None
    return {
        "metadata": metadata,
        "train_scores_by_id": {sample_id: float(score) for sample_id, score in zip(train_ids, oof_scores)},
        "test_scores_by_id": {sample_id: float(score) for sample_id, score in zip(test_ids, test_scores)},
    }


def _add_stack_score_features(features: dict[str, float], score: float) -> None:
    score = min(1.0, max(0.0, float(score)))
    centered = score - 0.5
    features["behavior_v2::stack::bc_lr_prob"] = score
    features[f"behavior_v2::stack::bc_lr_prob_bucket::{_prob_bucket(score)}"] = 1.0
    features["consistency_v2::stack::bc_lr_prob"] = score
    features[f"consistency_v2::stack::bc_lr_prob_bucket::{_prob_bucket(score)}"] = 1.0
    features["consistency_v2::stack::bc_lr_confidence"] = abs(centered) * 2.0


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if all(0.0 <= float(score) <= 1.0 for score in scores):
        return [float(score) for score in scores]
    return [_stable_sigmoid(float(score)) for score in scores]


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _prob_bucket(score: float) -> str:
    if score >= 0.9:
        return "0.90_1.00"
    if score >= 0.8:
        return "0.80_0.90"
    if score >= 0.7:
        return "0.70_0.80"
    if score >= 0.6:
        return "0.60_0.70"
    if score >= 0.5:
        return "0.50_0.60"
    if score >= 0.4:
        return "0.40_0.50"
    if score >= 0.3:
        return "0.30_0.40"
    if score >= 0.2:
        return "0.20_0.30"
    if score >= 0.1:
        return "0.10_0.20"
    return "0.00_0.10"


def _best_threshold(y_true: list[int], scores: list[float]) -> tuple[float, float]:
    selected = _select_threshold(y_true, scores, mode="train_f1")
    return float(selected["threshold"]), float(selected["train_f1"])


def _select_threshold(
    y_true: list[int],
    scores: list[float],
    mode: str = "train_f1",
    fixed_threshold: float = 0.5,
    beta: float = 1.5,
    min_precision: float = 0.88,
    min_recall: float = 0.0,
) -> dict[str, Any]:
    if not scores:
        return {
            "mode": mode,
            "threshold": 0.5,
            "objective": 0.0,
            "constraint_satisfied": False,
            "train_accuracy": 0.0,
            "train_precision": 0.0,
            "train_recall": 0.0,
            "train_f1": 0.0,
            "train_fbeta": 0.0,
        }
    if mode == "fixed":
        return {
            **_threshold_metrics(y_true, scores, float(fixed_threshold), beta=beta),
            "mode": mode,
            "threshold": float(fixed_threshold),
            "objective": 0.0,
            "constraint_satisfied": True,
        }
    candidates = _threshold_candidates(scores)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        row = _threshold_metrics(y_true, scores, threshold, beta=beta)
        if mode == "train_f1":
            objective = row["train_f1"]
            key = (objective, row["train_recall"], row["train_precision"], row["train_accuracy"], -float(threshold))
            constraint_satisfied = True
        elif mode == "train_fbeta":
            objective = row["train_fbeta"]
            key = (objective, row["train_recall"], row["train_f1"], row["train_precision"], -float(threshold))
            constraint_satisfied = True
        elif mode == "train_recall_at_precision":
            constraint_satisfied = row["train_precision"] >= min_precision
            if not constraint_satisfied:
                continue
            objective = row["train_recall"]
            key = (objective, row["train_f1"], row["train_precision"], row["train_accuracy"], -float(threshold))
        elif mode == "train_f1_recall_floor":
            constraint_satisfied = row["train_recall"] >= min_recall
            if not constraint_satisfied:
                continue
            objective = row["train_f1"]
            key = (objective, row["train_recall"], row["train_precision"], row["train_accuracy"], -float(threshold))
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")
        candidate = {
            **row,
            "mode": mode,
            "threshold": float(threshold),
            "objective": float(objective),
            "constraint_satisfied": bool(constraint_satisfied),
            "_key": key,
        }
        if best is None or candidate["_key"] > best["_key"]:
            best = candidate
    if best is None:
        best = _select_threshold(y_true, scores, mode="train_f1", beta=beta)
        best["requested_mode"] = mode
        best["constraint_satisfied"] = False
        best["min_precision"] = float(min_precision)
        best["min_recall"] = float(min_recall)
    best.pop("_key", None)
    if mode == "train_recall_at_precision":
        best["min_precision"] = float(min_precision)
    if mode == "train_f1_recall_floor":
        best["min_recall"] = float(min_recall)
    if mode == "train_fbeta":
        best["beta"] = float(beta)
    return best


def _threshold_candidates(scores: list[float]) -> list[float]:
    candidates = sorted({round(float(score), 6) for score in scores})
    if len(candidates) > 200:
        quantiles = np.linspace(0.01, 0.99, 199)
        candidates = sorted({float(np.quantile(scores, q)) for q in quantiles})
    return sorted(set(candidates) | {0.0, 0.5, 1.0})


def _threshold_metrics(y_true: list[int], scores: list[float], threshold: float, beta: float = 1.5) -> dict[str, float]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    return {
        "train_accuracy": float(accuracy_score(y_true, predictions)),
        "train_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "train_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "train_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "train_fbeta": float(fbeta_score(y_true, predictions, beta=max(float(beta), 1e-9), zero_division=0)),
    }


def evaluate_metric_row(
    y_test: list[int],
    predictions: Any,
    scores: list[float],
    test_ids: list[str],
    extra: dict[str, Any],
    save_predictions: bool = False,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "test_sample_ids": test_ids,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["benign", "malware"],
            zero_division=0,
            output_dict=True,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    except ValueError:
        metrics["roc_auc"] = None
    if save_predictions:
        metrics["prediction_rows"] = [
            {
                "sample_id": str(sample_id),
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "score": float(score),
                "true_label": "malware" if int(y_true) == 1 else "benign",
                "predicted_label": "malware" if int(y_pred) == 1 else "benign",
            }
            for sample_id, y_true, y_pred, score in zip(test_ids, y_test, predictions, scores)
        ]
    metrics.update(extra)
    return metrics


def count_labels(labels: list[int]) -> dict[str, int]:
    counter = Counter(labels)
    return {"benign": int(counter.get(0, 0)), "malware": int(counter.get(1, 0))}


def summarize_main(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for row in run.get("metrics", []):
            grouped.setdefault(str(row.get("display_name") or row.get("name")), []).append(row)
    return {name: summarize_rows(rows, METRIC_KEYS) for name, rows in grouped.items()}


def summarize_ablation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for row in run.get("metrics", []):
            grouped.setdefault(str(row.get("variant_id")), []).append(row)
    return {name: summarize_rows(rows, METRIC_KEYS) for name, rows in grouped.items()}


def summarize_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
        if not values:
            summary[key] = {"mean": None, "std": None, "values": []}
            continue
        summary[key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
            "values": values,
        }
    return summary


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# EviDroid Multi-Seed Experiment Summary",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Evidence: `{summary['evidence_path']}`",
        f"- Behaviors: `{summary['behavior_path']}`",
        f"- Ablation features: static=`{summary.get('ablation_static_profile')}`, "
        f"version=`{summary.get('ablation_feature_version')}`, "
        f"grouped_select=`{summary.get('ablation_grouped_select_k_best')}`, "
        f"A3_consistency=`{summary.get('ablation_a3_consistency_scope')}`, "
        f"A3_stacking=`{summary.get('ablation_a3_stacking')}` "
        f"({summary.get('ablation_a3_stack_source')}, "
        f"{summary.get('ablation_a3_stack_classifier')}, "
        f"mode={summary.get('ablation_a3_prediction_mode')})",
        "",
    ]
    if summary.get("encoder_fusion"):
        threshold_label = fusion_threshold_label(summary)
        lines.extend(
            [
                "- Fusion: `encoder-level late fusion` "
                f"with folds=`{summary.get('fusion_folds')}`, "
                f"mode=`{summary.get('fusion_score_mode')}`, "
                f"head=`{summary.get('fusion_head_classifier')}`, "
                f"threshold={threshold_label}, "
                f"branch_classifiers=`{summary.get('fusion_branch_classifiers')}`, "
                f"branch_select_k=`{summary.get('fusion_branch_select_k')}`, "
                f"score_weights=`{summary.get('fusion_score_weights')}`",
                "",
            ]
        )
    if summary.get("main_summary"):
        lines.extend(["## Main Comparison", ""])
        lines.extend(summary_table(summary["main_summary"], "Method"))
        lines.append("")
    if summary.get("ablation_summary"):
        lines.extend(["## Ablation", ""])
        lines.extend(summary_table(summary["ablation_summary"], "Variant"))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def fusion_threshold_label(summary: dict[str, Any]) -> str:
    mode = str(summary.get("fusion_threshold_mode"))
    if mode == "fixed":
        return f"`fixed` ({float(summary.get('fusion_fixed_threshold', 0.5)):.4f})"
    if mode == "train_fbeta":
        return f"`train_fbeta` (beta={float(summary.get('fusion_threshold_beta', 1.5)):.4f})"
    if mode == "train_recall_at_precision":
        return (
            "`train_recall_at_precision` "
            f"(min_precision={float(summary.get('fusion_min_train_precision', 0.0)):.4f})"
        )
    if mode == "train_f1_recall_floor":
        return (
            "`train_f1_recall_floor` "
            f"(min_recall={float(summary.get('fusion_min_train_recall', 0.0)):.4f})"
        )
    return f"`{mode}`"


def summary_table(grouped: dict[str, Any], name_col: str) -> list[str]:
    lines = [
        f"| {name_col} | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in grouped.items():
        lines.append(
            f"| {name} | {fmt_mean_std(row['accuracy'])} | {fmt_mean_std(row['precision'])} | "
            f"{fmt_mean_std(row['recall'])} | {fmt_mean_std(row['f1'])} | {fmt_mean_std(row['roc_auc'])} |"
        )
    return lines


def fmt_mean_std(metric: dict[str, Any]) -> str:
    if metric.get("mean") is None:
        return "-"
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


if __name__ == "__main__":
    main()
