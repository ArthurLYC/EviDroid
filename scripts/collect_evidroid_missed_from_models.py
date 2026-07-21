from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from evidroid.baselines import (  # noqa: E402
    build_deep_inputs,
    convert_mamadroid_package_cache_to_family,
    decision_scores,
    load_mamadroid_cache,
    load_sample_index,
    require_torch,
)
from evidroid.baselines.deep import (  # noqa: E402
    ApiTransformerClassifier,
    AppPoetLikeDNN,
    BagDataset,
    CLS_ID,
    SequenceDataset,
    UNK_ID,
    collate_bag,
    predict_bag,
    predict_sequence,
    torch,
    DataLoader,
)

from scripts.find_evidroid_missed_baseline_correct import (  # noqa: E402
    ALL_METHODS,
    DEEP_METHODS,
    SKLEARN_METHODS,
    build_streamed_static_features,
    collect_overlap_rows,
    load_evidroid_prediction_rows,
    load_evidroid_threshold,
    read_json,
    recompute_evidroid_predictions,
    summarize_correct_baseline_counts,
    summarize_seed,
    write_csv,
    write_json,
    write_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect EviDroid-wrong/baseline-correct cases from saved baseline models."
    )
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--behaviors", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--mamadroid-cache", default="data/processed/mamadroid_features_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--mamadroid-abstraction", choices=["package", "family_from_package"], default="family_from_package")
    parser.add_argument("--multiseed-dir", default="artifacts/optimized/full_llm_final_20000_balanced_multiseed_20260706")
    parser.add_argument(
        "--precision90-dir",
        default="artifacts/optimized/full_llm_final_20000_balanced_multiseed_20260706",
        help="Legacy argument name; defaults to the current paper-threshold multiseed directory.",
    )
    parser.add_argument(
        "--model-roots",
        default="",
        help="Comma-separated roots containing seed_<seed>/models/<method> model files.",
    )
    parser.add_argument("--out-dir", default="artifacts/analysis/final_20000_missed_baseline_correct")
    parser.add_argument("--seeds", default="42,2026,2027")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--max-api-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    behavior_path = Path(args.behaviors)
    mamadroid_cache_path = Path(args.mamadroid_cache)
    multiseed_dir = Path(args.multiseed_dir)
    precision90_dir = Path(args.precision90_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    model_roots = [Path(item.strip()) for item in args.model_roots.split(",") if item.strip()]

    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if set(methods) & DEEP_METHODS:
        require_torch()
        if args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)

    sample_index = load_sample_index(evidence_path)
    label_by_id = {row["sample_id"]: int(row["label_int"]) for row in sample_index}
    label_name_by_id = {row["sample_id"]: str(row["label"]) for row in sample_index}
    print(f"[setup] loaded labels for {len(label_by_id)} samples", flush=True)

    mamadroid_features: dict[str, dict[str, float]] | None = None
    if "mamadroid" in methods:
        print("[setup] loading MaMaDroid cache", flush=True)
        mamadroid_features = load_mamadroid_cache(mamadroid_cache_path)
        if args.mamadroid_abstraction == "family_from_package":
            mamadroid_features = convert_mamadroid_package_cache_to_family(mamadroid_features)
        print(f"[setup] loaded MaMaDroid features for {len(mamadroid_features)} samples", flush=True)

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input": {
            "evidence": str(evidence_path),
            "behaviors": str(behavior_path),
            "multiseed_dir": str(multiseed_dir),
            "precision90_dir": str(precision90_dir),
            "model_roots": [str(path) for path in model_roots],
            "methods": methods,
            "seeds": seeds,
        },
        "seeds": {},
    }

    for seed in seeds:
        started = time.perf_counter()
        print(f"[seed {seed}] collecting predictions from saved models", flush=True)
        split = read_json(multiseed_dir / f"seed_{seed}" / "split.json")
        train_ids = list(split["train_sample_ids"])
        test_ids = list(split["test_sample_ids"])
        y_train = [label_by_id[sample_id] for sample_id in train_ids]

        evidroid_rows = load_evidroid_prediction_rows(multiseed_dir / f"seed_{seed}" / "main" / "main_metrics.json")
        threshold = load_evidroid_threshold(precision90_dir / f"seed_{seed}" / "main" / "main_metrics.json")
        evidroid_by_id = recompute_evidroid_predictions(evidroid_rows, threshold)
        evidroid_wrong_ids = [
            sample_id
            for sample_id in test_ids
            if sample_id in evidroid_by_id and int(evidroid_by_id[sample_id]["y_pred"]) != label_by_id[sample_id]
        ]

        baseline_predictions: dict[str, dict[str, dict[str, Any]]] = {}
        static_methods = [
            method
            for method in methods
            if method in {"drebin", "droidapiminer"} and find_model_path(model_roots, seed, method)
        ]
        if static_methods:
            static_features = build_streamed_static_features(evidence_path, set(train_ids) | set(test_ids), static_methods)
            for method in static_methods:
                model_path = find_model_path(model_roots, seed, method)
                if model_path is None:
                    continue
                print(f"[seed {seed}] loading {method}: {model_path}", flush=True)
                if method == "drebin":
                    x_test = [static_features["drebin"][sample_id] for sample_id in test_ids]
                elif method == "droidapiminer":
                    x_test = [static_features["droidapiminer"][sample_id] for sample_id in test_ids]
                baseline_predictions[method] = predict_sklearn_model(model_path, x_test, test_ids)

        if "mamadroid" in methods:
            model_path = find_model_path(model_roots, seed, "mamadroid")
            if model_path is not None and mamadroid_features is not None:
                print(f"[seed {seed}] loading mamadroid: {model_path}", flush=True)
                x_test = [mamadroid_features.get(sample_id, {}) for sample_id in test_ids]
                baseline_predictions["mamadroid"] = predict_sklearn_model(model_path, x_test, test_ids)

        deep_methods = [method for method in methods if method in DEEP_METHODS and find_model_path(model_roots, seed, method)]
        if deep_methods:
            deep_cache = build_deep_inputs(
                evidence_path=evidence_path,
                behavior_by_id={},
                wanted_ids=set(train_ids) | set(test_ids),
                max_api_len=args.max_api_len,
                include_behavior_in_apppoet=False,
            )
            for method in deep_methods:
                model_path = find_model_path(model_roots, seed, method)
                if model_path is None:
                    continue
                print(f"[seed {seed}] loading {method}: {model_path}", flush=True)
                if method == "apppoet":
                    baseline_predictions[method] = predict_apppoet_model(
                        model_path=model_path,
                        deep_cache=deep_cache,
                        test_ids=test_ids,
                        batch_size=args.batch_size,
                    )
                elif method == "api_transformer":
                    baseline_predictions[method] = predict_api_transformer_model(
                        model_path=model_path,
                        deep_cache=deep_cache,
                        test_ids=test_ids,
                        batch_size=args.batch_size,
                        fallback_max_len=args.max_api_len,
                    )

        missing_methods = [method for method in methods if method not in baseline_predictions]
        if missing_methods:
            print(f"[seed {seed}] missing saved predictions for: {','.join(missing_methods)}", flush=True)

        seed_rows = collect_overlap_rows(
            seed=seed,
            test_ids=test_ids,
            label_by_id=label_by_id,
            label_name_by_id=label_name_by_id,
            evidroid_by_id=evidroid_by_id,
            evidroid_wrong_ids=evidroid_wrong_ids,
            threshold=threshold,
            baseline_predictions=baseline_predictions,
            methods=methods,
        )
        all_rows.extend(seed_rows)
        seed_summary = summarize_seed(
            seed_rows=seed_rows,
            evidroid_wrong_ids=evidroid_wrong_ids,
            baseline_predictions=baseline_predictions,
            label_by_id=label_by_id,
            methods=methods,
        )
        seed_summary["missing_methods"] = missing_methods
        seed_summary["seconds"] = round(time.perf_counter() - started, 3)
        summary["seeds"][str(seed)] = seed_summary
        print(f"[seed {seed}] collected {len(seed_rows)} rows", flush=True)

    all_rows.sort(
        key=lambda row: (
            -int(row["num_correct_baselines"]),
            str(row["evi_error_type"]),
            int(row["seed"]),
            str(row["sample_id"]),
        )
    )
    summary["total_rows"] = len(all_rows)
    summary["total_unique_samples"] = len({row["sample_id"] for row in all_rows})
    summary["correct_baseline_counts"] = summarize_correct_baseline_counts(all_rows, methods)
    summary["error_type_counts"] = {}
    for row in all_rows:
        summary["error_type_counts"][row["evi_error_type"]] = summary["error_type_counts"].get(row["evi_error_type"], 0) + 1

    write_csv(out_dir / "evidroid_wrong_baseline_correct.csv", all_rows, methods)
    write_json(out_dir / "evidroid_wrong_baseline_correct.json", all_rows)
    write_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "top_cases.md", all_rows, summary, methods)
    print(f"[done] wrote outputs to {out_dir}", flush=True)


def find_model_path(model_roots: list[Path], seed: int, method: str) -> Path | None:
    suffix = ".joblib" if method in SKLEARN_METHODS else ".pt"
    candidates = [
        root / f"seed_{seed}" / "models" / f"{method}_model{suffix}"
        for root in model_roots
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def predict_sklearn_model(model_path: Path, x_test: list[dict[str, float]], test_ids: list[str]) -> dict[str, dict[str, Any]]:
    model = joblib.load(model_path)
    predictions = [int(item) for item in model.predict(x_test)]
    scores = [float(item) for item in decision_scores(model, x_test)]
    return {
        sample_id: {
            "y_pred": int(y_pred),
            "score": float(score),
            "predicted_label": "malware" if int(y_pred) == 1 else "benign",
        }
        for sample_id, y_pred, score in zip(test_ids, predictions, scores)
    }


def predict_apppoet_model(
    model_path: Path,
    deep_cache: dict[str, dict[str, Any]],
    test_ids: list[str],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    test_rows = [[vocab.get(token, UNK_ID) for token in deep_cache[sample_id]["apppoet_tokens"]] for sample_id in test_ids]
    model = AppPoetLikeDNN(vocab_size=max(vocab.values(), default=UNK_ID) + 1)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(BagDataset(test_rows, [0] * len(test_rows)), batch_size=batch_size, shuffle=False, collate_fn=collate_bag)
    scores = predict_bag(model, loader)
    return {
        sample_id: {
            "y_pred": 1 if score >= 0.5 else 0,
            "score": float(score),
            "predicted_label": "malware" if score >= 0.5 else "benign",
        }
        for sample_id, score in zip(test_ids, scores)
    }


def predict_api_transformer_model(
    model_path: Path,
    deep_cache: dict[str, dict[str, Any]],
    test_ids: list[str],
    batch_size: int,
    fallback_max_len: int,
) -> dict[str, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    max_len = int(checkpoint.get("max_len") or fallback_max_len)
    test_rows = [
        encode_sequence_local(deep_cache[sample_id]["api_sequence"], vocab, max_len=max_len)
        for sample_id in test_ids
    ]
    model = ApiTransformerClassifier(vocab_size=max(vocab.values(), default=CLS_ID) + 1, max_len=max_len)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(SequenceDataset(test_rows, [0] * len(test_rows)), batch_size=batch_size, shuffle=False)
    scores = predict_sequence(model, loader)
    return {
        sample_id: {
            "y_pred": 1 if score >= 0.5 else 0,
            "score": float(score),
            "predicted_label": "malware" if score >= 0.5 else "benign",
        }
        for sample_id, score in zip(test_ids, scores)
    }


def encode_sequence_local(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [CLS_ID] + [vocab.get(token, UNK_ID) for token in tokens[: max_len - 1]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids[:max_len]


if __name__ == "__main__":
    main()
