from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
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
    build_drebin_features,
    build_droidapiminer_features,
    convert_mamadroid_package_cache_to_family,
    decision_scores,
    iter_jsonl,
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
    build_vocab,
    collate_bag,
    encode_sequence,
    predict_bag,
    predict_sequence,
    set_torch_seed,
    torch,
    nn,
    DataLoader,
)
from evidroid.classifier_selection import make_classifier_pipeline  # noqa: E402


SKLEARN_METHODS = {"drebin", "droidapiminer", "mamadroid"}
DEEP_METHODS = {"apppoet", "api_transformer"}
ALL_METHODS = ("drebin", "droidapiminer", "mamadroid", "apppoet", "api_transformer")
DISPLAY_NAMES = {
    "drebin": "Drebin",
    "droidapiminer": "DroidAPIMiner",
    "mamadroid": "MaMaDroid",
    "apppoet": "AppPoet-like",
    "api_transformer": "API-Transformer",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export samples where EviDroid is wrong under the paper threshold "
            "but one or more baseline models are correct."
        )
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
    parser.add_argument("--out-dir", default="artifacts/analysis/final_20000_missed_baseline_correct")
    parser.add_argument("--seeds", default="42,2026,2027")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--select-k-best", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--max-api-len", type=int, default=256)
    parser.add_argument("--max-api-vocab", type=int, default=8000)
    parser.add_argument("--max-appoet-vocab", type=int, default=12000)
    parser.add_argument("--apppoet-include-behavior", action="store_true")
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Optional limit for CSV/JSON rows after sorting; 0 keeps all rows.",
    )
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    behavior_path = Path(args.behaviors)
    mamadroid_cache_path = Path(args.mamadroid_cache)
    multiseed_dir = Path(args.multiseed_dir)
    precision90_dir = Path(args.precision90_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

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
            "mamadroid_cache": str(mamadroid_cache_path),
            "multiseed_dir": str(multiseed_dir),
            "precision90_dir": str(precision90_dir),
            "methods": methods,
            "seeds": seeds,
        },
        "seeds": {},
    }

    for seed in seeds:
        started = time.perf_counter()
        print(f"[seed {seed}] loading split and EviDroid predictions", flush=True)
        split = read_json(multiseed_dir / f"seed_{seed}" / "split.json")
        train_ids = list(split["train_sample_ids"])
        test_ids = list(split["test_sample_ids"])
        y_train = [label_by_id[sample_id] for sample_id in train_ids]
        y_test = [label_by_id[sample_id] for sample_id in test_ids]

        evidroid_rows = load_evidroid_prediction_rows(multiseed_dir / f"seed_{seed}" / "main" / "main_metrics.json")
        threshold = load_evidroid_threshold(precision90_dir / f"seed_{seed}" / "main" / "main_metrics.json")
        evidroid_by_id = recompute_evidroid_predictions(evidroid_rows, threshold)
        evidroid_wrong_ids = [
            sample_id
            for sample_id in test_ids
            if sample_id in evidroid_by_id and int(evidroid_by_id[sample_id]["y_pred"]) != label_by_id[sample_id]
        ]
        print(
            f"[seed {seed}] EviDroid wrong under paper threshold: {len(evidroid_wrong_ids)} / {len(test_ids)}",
            flush=True,
        )

        baseline_predictions: dict[str, dict[str, dict[str, Any]]] = {}
        if set(methods) & {"drebin", "droidapiminer"}:
            streamed_features = build_streamed_static_features(evidence_path, set(train_ids) | set(test_ids), methods)
            if "drebin" in methods:
                baseline_predictions["drebin"] = train_sklearn_predictions(
                    x_train=[streamed_features["drebin"][sample_id] for sample_id in train_ids],
                    x_test=[streamed_features["drebin"][sample_id] for sample_id in test_ids],
                    y_train=y_train,
                    test_ids=test_ids,
                    classifier="linear_svm_sgd",
                    select_k_best=0,
                    seed=seed,
                    method="drebin",
                    out_dir=out_dir / f"seed_{seed}" / "models",
                )
            if "droidapiminer" in methods:
                baseline_predictions["droidapiminer"] = train_sklearn_predictions(
                    x_train=[streamed_features["droidapiminer"][sample_id] for sample_id in train_ids],
                    x_test=[streamed_features["droidapiminer"][sample_id] for sample_id in test_ids],
                    y_train=y_train,
                    test_ids=test_ids,
                    classifier="linear_svm_sgd",
                    select_k_best=args.select_k_best,
                    seed=seed,
                    method="droidapiminer",
                    out_dir=out_dir / f"seed_{seed}" / "models",
                )

        if "mamadroid" in methods:
            if mamadroid_features is None:
                raise ValueError("MaMaDroid features were not loaded.")
            baseline_predictions["mamadroid"] = train_sklearn_predictions(
                x_train=[mamadroid_features.get(sample_id, {}) for sample_id in train_ids],
                x_test=[mamadroid_features.get(sample_id, {}) for sample_id in test_ids],
                y_train=y_train,
                test_ids=test_ids,
                classifier="random_forest",
                select_k_best=0,
                seed=seed,
                method="mamadroid",
                out_dir=out_dir / f"seed_{seed}" / "models",
            )

        if set(methods) & DEEP_METHODS:
            print(f"[seed {seed}] building deep baseline inputs", flush=True)
            behavior_by_id = {}
            if args.apppoet_include_behavior:
                behavior_by_id = {
                    row["sample_id"]: row
                    for row in iter_jsonl(behavior_path)
                    if row.get("sample_id") in set(train_ids) | set(test_ids)
                }
            deep_cache = build_deep_inputs(
                evidence_path=evidence_path,
                behavior_by_id=behavior_by_id,
                wanted_ids=set(train_ids) | set(test_ids),
                max_api_len=args.max_api_len,
                include_behavior_in_apppoet=args.apppoet_include_behavior,
            )
            if "apppoet" in methods:
                baseline_predictions["apppoet"] = train_apppoet_predictions(
                    deep_cache=deep_cache,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    max_vocab=args.max_appoet_vocab,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=seed,
                    out_dir=out_dir / f"seed_{seed}" / "models",
                )
            if "api_transformer" in methods:
                baseline_predictions["api_transformer"] = train_api_transformer_predictions(
                    deep_cache=deep_cache,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    y_train=y_train,
                    max_vocab=args.max_api_vocab,
                    max_len=args.max_api_len,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=seed,
                    out_dir=out_dir / f"seed_{seed}" / "models",
                )

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
        seed_summary["seconds"] = round(time.perf_counter() - started, 3)
        summary["seeds"][str(seed)] = seed_summary
        print(
            f"[seed {seed}] overlap rows={len(seed_rows)} in {seed_summary['seconds']:.1f}s",
            flush=True,
        )

    all_rows.sort(
        key=lambda row: (
            -int(row["num_correct_baselines"]),
            str(row["evi_error_type"]),
            int(row["seed"]),
            str(row["sample_id"]),
        )
    )
    if args.limit_cases > 0:
        all_rows = all_rows[: args.limit_cases]

    summary["total_rows"] = len(all_rows)
    summary["total_unique_samples"] = len({row["sample_id"] for row in all_rows})
    summary["error_type_counts"] = dict(Counter(row["evi_error_type"] for row in all_rows))
    summary["correct_baseline_counts"] = summarize_correct_baseline_counts(all_rows, methods)

    write_csv(out_dir / "evidroid_wrong_baseline_correct.csv", all_rows, methods)
    write_json(out_dir / "evidroid_wrong_baseline_correct.json", all_rows)
    write_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "top_cases.md", all_rows, summary, methods)
    print(f"[done] wrote outputs to {out_dir}", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_evidroid_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    metrics = payload.get("metrics", []) if isinstance(payload, dict) else []
    for row in metrics:
        if str(row.get("name", "")).lower() == "evidroid" or str(row.get("display_name", "")).lower() == "evidroid":
            prediction_rows = row.get("prediction_rows")
            if isinstance(prediction_rows, list) and prediction_rows:
                return prediction_rows
    raise ValueError(f"No EviDroid prediction_rows found in {path}")


def load_evidroid_threshold(path: Path) -> float:
    payload = read_json(path)
    metrics = payload.get("metrics", []) if isinstance(payload, dict) else []
    for row in metrics:
        if str(row.get("name", "")).lower() == "evidroid" or str(row.get("display_name", "")).lower() == "evidroid":
            for key in ("fusion_threshold", "threshold"):
                if row.get(key) is not None:
                    return float(row[key])
    raise ValueError(f"No EviDroid threshold found in {path}")


def recompute_evidroid_predictions(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        score = float(row["score"])
        y_true = int(row["y_true"])
        y_pred = 1 if score >= threshold else 0
        result[sample_id] = {
            "sample_id": sample_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "score": score,
            "predicted_label": "malware" if y_pred == 1 else "benign",
            "true_label": "malware" if y_true == 1 else "benign",
        }
    return result


def build_streamed_static_features(
    evidence_path: Path,
    wanted_ids: set[str],
    methods: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    if "drebin" in methods:
        result["drebin"] = {}
    if "droidapiminer" in methods:
        result["droidapiminer"] = {}
    start = time.perf_counter()
    for evidence_doc in iter_jsonl(evidence_path):
        sample_id = evidence_doc.get("sample_id")
        if sample_id not in wanted_ids:
            continue
        if "drebin" in result:
            result["drebin"][sample_id] = build_drebin_features(evidence_doc)
        if "droidapiminer" in result:
            result["droidapiminer"][sample_id] = build_droidapiminer_features(evidence_doc)
        if sum(len(rows) for rows in result.values()) and len(next(iter(result.values()))) % 5000 == 0:
            print(f"[static] built {len(next(iter(result.values())))} rows", flush=True)
    print(f"[static] built feature rows in {time.perf_counter() - start:.2f}s", flush=True)
    return result


def train_sklearn_predictions(
    x_train: list[dict[str, float]],
    x_test: list[dict[str, float]],
    y_train: list[int],
    test_ids: list[str],
    classifier: str,
    select_k_best: int,
    seed: int,
    method: str,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    print(f"[seed {seed}] training {method}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_classifier_pipeline(classifier, random_state=seed, select_k_best=select_k_best)
    model.fit(x_train, y_train)
    predictions = [int(item) for item in model.predict(x_test)]
    scores = [float(item) for item in decision_scores(model, x_test)]
    model_path = out_dir / f"{method}_model.joblib"
    joblib.dump(model, model_path)
    return {
        sample_id: {
            "y_pred": int(y_pred),
            "score": float(score),
            "predicted_label": "malware" if int(y_pred) == 1 else "benign",
        }
        for sample_id, y_pred, score in zip(test_ids, predictions, scores)
    }


def train_apppoet_predictions(
    deep_cache: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    max_vocab: int,
    epochs: int,
    batch_size: int,
    seed: int,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    print(f"[seed {seed}] training apppoet", flush=True)
    require_torch()
    set_torch_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab([deep_cache[sample_id]["apppoet_tokens"] for sample_id in train_ids], max_vocab=max_vocab, reserved=2)
    train_rows = [[vocab.get(token, UNK_ID) for token in deep_cache[sample_id]["apppoet_tokens"]] for sample_id in train_ids]
    test_rows = [[vocab.get(token, UNK_ID) for token in deep_cache[sample_id]["apppoet_tokens"]] for sample_id in test_ids]
    model = AppPoetLikeDNN(vocab_size=max(vocab.values(), default=UNK_ID) + 1)
    train_loader = DataLoader(BagDataset(train_rows, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate_bag)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, offsets, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, offsets)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"[apppoet] seed={seed} epoch={epoch} loss={sum(losses) / len(losses):.4f}", flush=True)
    test_loader = DataLoader(BagDataset(test_rows, [0] * len(test_rows)), batch_size=batch_size, shuffle=False, collate_fn=collate_bag)
    scores = predict_bag(model, test_loader)
    torch.save({"model_state": model.state_dict(), "vocab": vocab}, out_dir / "apppoet_model.pt")
    return {
        sample_id: {
            "y_pred": 1 if score >= 0.5 else 0,
            "score": float(score),
            "predicted_label": "malware" if score >= 0.5 else "benign",
        }
        for sample_id, score in zip(test_ids, scores)
    }


def train_api_transformer_predictions(
    deep_cache: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    max_vocab: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    seed: int,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    print(f"[seed {seed}] training api_transformer", flush=True)
    require_torch()
    set_torch_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab([deep_cache[sample_id]["api_sequence"] for sample_id in train_ids], max_vocab=max_vocab, reserved=3)
    train_rows = [encode_sequence(deep_cache[sample_id]["api_sequence"], vocab, max_len=max_len) for sample_id in train_ids]
    test_rows = [encode_sequence(deep_cache[sample_id]["api_sequence"], vocab, max_len=max_len) for sample_id in test_ids]
    model = ApiTransformerClassifier(vocab_size=max(vocab.values(), default=CLS_ID) + 1, max_len=max_len)
    train_loader = DataLoader(SequenceDataset(train_rows, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for input_ids, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"[api_transformer] seed={seed} epoch={epoch} loss={sum(losses) / len(losses):.4f}", flush=True)
    test_loader = DataLoader(SequenceDataset(test_rows, [0] * len(test_rows)), batch_size=batch_size, shuffle=False)
    scores = predict_sequence(model, test_loader)
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "max_len": max_len}, out_dir / "api_transformer_model.pt")
    return {
        sample_id: {
            "y_pred": 1 if score >= 0.5 else 0,
            "score": float(score),
            "predicted_label": "malware" if score >= 0.5 else "benign",
        }
        for sample_id, score in zip(test_ids, scores)
    }


def collect_overlap_rows(
    seed: int,
    test_ids: list[str],
    label_by_id: dict[str, int],
    label_name_by_id: dict[str, str],
    evidroid_by_id: dict[str, dict[str, Any]],
    evidroid_wrong_ids: list[str],
    threshold: float,
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
    methods: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_id_set = set(test_ids)
    for sample_id in evidroid_wrong_ids:
        if sample_id not in test_id_set:
            continue
        y_true = label_by_id[sample_id]
        correct_methods = [
            method
            for method in methods
            if method in baseline_predictions
            and sample_id in baseline_predictions[method]
            and int(baseline_predictions[method][sample_id]["y_pred"]) == y_true
        ]
        if not correct_methods:
            continue
        evi = evidroid_by_id[sample_id]
        row: dict[str, Any] = {
            "seed": seed,
            "sample_id": sample_id,
            "true_y": y_true,
            "true_label": label_name_by_id.get(sample_id, "malware" if y_true == 1 else "benign"),
            "evi_pred": int(evi["y_pred"]),
            "evi_predicted_label": evi["predicted_label"],
            "evi_score": float(evi["score"]),
            "evi_threshold": float(threshold),
            "evi_error_type": "FN" if y_true == 1 and int(evi["y_pred"]) == 0 else "FP",
            "correct_baselines": ";".join(correct_methods),
            "num_correct_baselines": len(correct_methods),
        }
        for method in methods:
            pred = baseline_predictions.get(method, {}).get(sample_id)
            if pred is None:
                row[f"{method}_pred"] = ""
                row[f"{method}_predicted_label"] = ""
                row[f"{method}_score"] = ""
                row[f"{method}_correct"] = ""
            else:
                method_pred = int(pred["y_pred"])
                row[f"{method}_pred"] = method_pred
                row[f"{method}_predicted_label"] = pred["predicted_label"]
                row[f"{method}_score"] = float(pred["score"])
                row[f"{method}_correct"] = method_pred == y_true
        rows.append(row)
    return rows


def summarize_seed(
    seed_rows: list[dict[str, Any]],
    evidroid_wrong_ids: list[str],
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
    label_by_id: dict[str, int],
    methods: list[str],
) -> dict[str, Any]:
    baseline_correct_on_evi_wrong = Counter()
    for sample_id in evidroid_wrong_ids:
        y_true = label_by_id[sample_id]
        for method in methods:
            pred = baseline_predictions.get(method, {}).get(sample_id)
            if pred is not None and int(pred["y_pred"]) == y_true:
                baseline_correct_on_evi_wrong[method] += 1
    return {
        "evidroid_wrong": len(evidroid_wrong_ids),
        "overlap_rows": len(seed_rows),
        "unique_overlap_samples": len({row["sample_id"] for row in seed_rows}),
        "error_type_counts": dict(Counter(row["evi_error_type"] for row in seed_rows)),
        "baseline_correct_on_evidroid_wrong": dict(baseline_correct_on_evi_wrong),
    }


def summarize_correct_baseline_counts(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for method in methods:
            if row.get(f"{method}_correct") is True:
                counts[method] += 1
    return dict(counts)


def write_csv(path: Path, rows: list[dict[str, Any]], methods: list[str]) -> None:
    base_fields = [
        "seed",
        "sample_id",
        "true_y",
        "true_label",
        "evi_pred",
        "evi_predicted_label",
        "evi_score",
        "evi_threshold",
        "evi_error_type",
        "correct_baselines",
        "num_correct_baselines",
    ]
    method_fields: list[str] = []
    for method in methods:
        method_fields.extend(
            [
                f"{method}_pred",
                f"{method}_predicted_label",
                f"{method}_score",
                f"{method}_correct",
            ]
        )
    fields = base_fields + method_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], methods: list[str]) -> None:
    lines = [
        "# EviDroid-Wrong / Baseline-Correct Cases",
        "",
        "This file lists representative cases where EviDroid is wrong under the paper threshold, but at least one baseline is correct.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary.get('total_rows', 0)}",
        f"- Unique samples: {summary.get('total_unique_samples', 0)}",
        f"- Error types: {summary.get('error_type_counts', {})}",
        f"- Baseline coverage: {summary.get('correct_baseline_counts', {})}",
        "",
        "## Top Cases",
        "",
    ]
    header = ["seed", "sample_id", "true", "EviDroid", "error", "correct_baselines"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[:50]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["sample_id"]),
                    str(row["true_label"]),
                    f"{row['evi_predicted_label']} ({float(row['evi_score']):.4f})",
                    str(row["evi_error_type"]),
                    "; ".join(DISPLAY_NAMES.get(method, method) for method in str(row["correct_baselines"]).split(";") if method),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Included Baselines", ""])
    for method in methods:
        lines.append(f"- {DISPLAY_NAMES.get(method, method)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
