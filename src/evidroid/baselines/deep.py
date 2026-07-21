from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evidroid.baselines.runner import evaluate_predictions, iter_jsonl
from evidroid.baselines.static import abstract_api
from evidroid.features import normalize_feature_value
from evidroid.schemas import iter_evidence

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
_TORCH_INSTALL_HINT = (
    "PyTorch is required only for AppPoet-like/API-Transformer baselines. "
    "Install a PyTorch build compatible with the server CUDA driver before "
    "running these methods."
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    torch = None
    DataLoader = None
    _TORCH_IMPORT_ERROR = exc

    class _MissingTorchNN:
        Module = object

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(_TORCH_INSTALL_HINT) from _TORCH_IMPORT_ERROR

    class Dataset:  # type: ignore[no-redef]
        pass

    nn = _MissingTorchNN()  # type: ignore[assignment]


def require_torch() -> Any:
    if _TORCH_IMPORT_ERROR is not None or torch is None:
        raise RuntimeError(_TORCH_INSTALL_HINT) from _TORCH_IMPORT_ERROR
    return torch


def build_deep_inputs(
    evidence_path: Path,
    behavior_by_id: dict[str, dict[str, Any]],
    wanted_ids: set[str],
    max_api_len: int,
    include_behavior_in_apppoet: bool = False,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    start = time.perf_counter()
    for row in iter_jsonl(evidence_path):
        sample_id = row.get("sample_id")
        if sample_id not in wanted_ids:
            continue
        behavior_doc = behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []})
        result[sample_id] = {
            "api_sequence": api_sequence_tokens(row, max_len=max_api_len),
            "apppoet_tokens": apppoet_tokens(
                row,
                behavior_doc,
                include_behavior=include_behavior_in_apppoet,
            ),
        }
        if len(result) % 5000 == 0:
            print(f"[deep-input] built {len(result)} compact rows", flush=True)
    print(f"[deep-input] built {len(result)} rows in {time.perf_counter() - start:.2f}s", flush=True)
    return result


def api_sequence_tokens(evidence_doc: dict[str, Any], max_len: int) -> list[str]:
    tokens: list[str] = []
    for item in iter_evidence(evidence_doc):
        if item["view"] != "api":
            continue
        tokens.append(abstract_api(item["value"], abstraction="package"))
        if len(tokens) >= max_len:
            break
    return tokens or ["no_api"]


def apppoet_tokens(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    include_behavior: bool = False,
) -> list[str]:
    tokens: list[str] = []
    api_packages: set[str] = set()
    for item in iter_evidence(evidence_doc):
        view = item["view"]
        value = str(item.get("value", ""))
        if view == "permission":
            tokens.append(f"permission:{normalize_feature_value('permission', value)}")
        elif view == "api":
            api_packages.add(abstract_api(value, abstraction="package"))
        elif view == "component":
            component_type = item.get("detail", {}).get("component_type", "component")
            tokens.append(f"component:{component_type}")
        elif view == "string":
            for category in string_categories(value):
                tokens.append(f"string:{category}")
    for package in sorted(api_packages)[:256]:
        tokens.append(f"api:{package}")
    if include_behavior:
        for behavior in behavior_doc.get("behaviors", []):
            label = behavior.get("label")
            if label:
                tokens.append(f"behavior:{label}")
            for view in behavior.get("support_by_view", {}):
                tokens.append(f"behavior_view:{label}:{view}")
    return tokens or ["empty_app"]


def string_categories(value: str) -> list[str]:
    lower = value.lower()
    categories: list[str] = []
    if "http://" in lower or "https://" in lower:
        categories.append("url")
    if "content://" in lower:
        categories.append("content_uri")
    if lower.startswith("/"):
        categories.append("path")
    if ".apk" in lower:
        categories.append("apk")
    if ".dex" in lower:
        categories.append("dex")
    if ".so" in lower:
        categories.append("native_lib")
    if "sms" in lower:
        categories.append("sms")
    if "imei" in lower:
        categories.append("imei")
    if "imsi" in lower:
        categories.append("imsi")
    if "android_id" in lower:
        categories.append("android_id")
    if "su" in lower or "chmod" in lower or "sh " in lower:
        categories.append("command")
    return categories


def build_vocab(token_rows: list[list[str]], max_vocab: int, reserved: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in token_rows:
        counter.update(tokens)
    return {token: idx + reserved for idx, (token, _count) in enumerate(counter.most_common(max_vocab))}


def run_apppoet_like(
    deep_cache: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    max_vocab: int,
    epochs: int,
    batch_size: int,
    random_state: int,
    out_dir: Path,
    include_behavior: bool = False,
) -> dict[str, Any]:
    require_torch()
    set_torch_seed(random_state)
    vocab = build_vocab([deep_cache[sample_id]["apppoet_tokens"] for sample_id in train_ids], max_vocab=max_vocab, reserved=2)
    train_rows = [[vocab.get(token, UNK_ID) for token in deep_cache[sample_id]["apppoet_tokens"]] for sample_id in train_ids]
    test_rows = [[vocab.get(token, UNK_ID) for token in deep_cache[sample_id]["apppoet_tokens"]] for sample_id in test_ids]
    model = AppPoetLikeDNN(vocab_size=max(vocab.values(), default=UNK_ID) + 1)
    metrics, history = train_bag_model(
        model=model,
        train_rows=train_rows,
        test_rows=test_rows,
        y_train=y_train,
        y_test=y_test,
        test_ids=test_ids,
        epochs=epochs,
        batch_size=batch_size,
    )
    model_path = out_dir / "apppoet_model.pt"
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "history": history}, model_path)
    metrics.update(
        {
            "name": "apppoet",
            "display_name": "AppPoet-like",
            "feature_type": "Static multi-view semantic tokens"
            if not include_behavior
            else "Static + behavior semantic tokens",
            "classifier": "embedding_bag_dnn",
            "status": "ok",
            "feature_count": len(vocab),
            "selected_feature_count": len(vocab),
            "model_path": str(model_path),
            "model_size_bytes": int(model_path.stat().st_size),
            "history": history,
        }
    )
    return metrics


def run_api_transformer(
    deep_cache: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    max_vocab: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    random_state: int,
    out_dir: Path,
) -> dict[str, Any]:
    require_torch()
    set_torch_seed(random_state)
    vocab = build_vocab([deep_cache[sample_id]["api_sequence"] for sample_id in train_ids], max_vocab=max_vocab, reserved=3)
    train_rows = [encode_sequence(deep_cache[sample_id]["api_sequence"], vocab, max_len=max_len) for sample_id in train_ids]
    test_rows = [encode_sequence(deep_cache[sample_id]["api_sequence"], vocab, max_len=max_len) for sample_id in test_ids]
    model = ApiTransformerClassifier(vocab_size=max(vocab.values(), default=CLS_ID) + 1, max_len=max_len)
    metrics, history = train_sequence_model(
        model=model,
        train_rows=train_rows,
        test_rows=test_rows,
        y_train=y_train,
        y_test=y_test,
        test_ids=test_ids,
        epochs=epochs,
        batch_size=batch_size,
    )
    model_path = out_dir / "api_transformer_model.pt"
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "history": history, "max_len": max_len}, model_path)
    metrics.update(
        {
            "name": "api_transformer",
            "display_name": "API-Transformer",
            "feature_type": "API sequence semantics",
            "classifier": "transformer_encoder",
            "status": "ok",
            "feature_count": len(vocab),
            "selected_feature_count": len(vocab),
            "model_path": str(model_path),
            "model_size_bytes": int(model_path.stat().st_size),
            "history": history,
        }
    )
    return metrics


def encode_sequence(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [CLS_ID] + [vocab.get(token, UNK_ID) for token in tokens[: max_len - 1]]
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    return ids[:max_len]


class BagDataset(Dataset):
    def __init__(self, rows: list[list[int]], labels: list[int]) -> None:
        self.rows = [row or [UNK_ID] for row in rows]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.rows[idx], self.labels[idx]


def collate_bag(batch: list[tuple[list[int], int]]) -> tuple[Any, Any, Any]:
    values: list[int] = []
    offsets: list[int] = []
    labels: list[int] = []
    offset = 0
    for row, label in batch:
        offsets.append(offset)
        values.extend(row)
        offset += len(row)
        labels.append(label)
    return (
        torch.tensor(values, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


class SequenceDataset(Dataset):
    def __init__(self, rows: list[list[int]], labels: list[int]) -> None:
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return torch.tensor(self.rows[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


class AppPoetLikeDNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean", padding_idx=PAD_ID)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, values: Any, offsets: Any) -> Any:
        pooled = self.embedding(values, offsets)
        return self.classifier(pooled).squeeze(-1)


class ApiTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 128,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, input_ids: Any) -> Any:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids.eq(PAD_ID)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.classifier(encoded[:, 0, :]).squeeze(-1)


def train_bag_model(
    model: Any,
    train_rows: list[list[int]],
    test_rows: list[list[int]],
    y_train: list[int],
    y_test: list[int],
    test_ids: list[str],
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    require_torch()
    train_loader = DataLoader(BagDataset(train_rows, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate_bag)
    test_loader = DataLoader(BagDataset(test_rows, y_test), batch_size=batch_size, shuffle=False, collate_fn=collate_bag)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    fit_start = time.perf_counter()
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
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else math.nan})
        print(f"[apppoet] epoch={epoch} loss={history[-1]['train_loss']:.4f}", flush=True)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    scores = predict_bag(model, test_loader)
    predict_seconds = time.perf_counter() - predict_start
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    metrics = evaluate_predictions(
        name="apppoet",
        display_name="AppPoet-like",
        feature_type="LLM-style multi-view semantic tokens",
        classifier="embedding_bag_dnn",
        y_test=y_test,
        predictions=predictions,
        scores=scores,
        test_ids=test_ids,
        extra={"fit_seconds": float(fit_seconds), "predict_seconds": float(predict_seconds)},
    )
    return metrics, history


def train_sequence_model(
    model: Any,
    train_rows: list[list[int]],
    test_rows: list[list[int]],
    y_train: list[int],
    y_test: list[int],
    test_ids: list[str],
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    require_torch()
    train_loader = DataLoader(SequenceDataset(train_rows, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SequenceDataset(test_rows, y_test), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    fit_start = time.perf_counter()
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
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else math.nan})
        print(f"[api_transformer] epoch={epoch} loss={history[-1]['train_loss']:.4f}", flush=True)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    scores = predict_sequence(model, test_loader)
    predict_seconds = time.perf_counter() - predict_start
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    metrics = evaluate_predictions(
        name="api_transformer",
        display_name="API-Transformer",
        feature_type="API sequence semantics",
        classifier="transformer_encoder",
        y_test=y_test,
        predictions=predictions,
        scores=scores,
        test_ids=test_ids,
        extra={"fit_seconds": float(fit_seconds), "predict_seconds": float(predict_seconds)},
    )
    return metrics, history


def predict_bag(model: Any, loader: Any) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for values, offsets, _labels in loader:
            logits = model(values, offsets)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return [float(score) for score in scores]


def predict_sequence(model: Any, loader: Any) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for input_ids, _labels in loader:
            logits = model(input_ids)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return [float(score) for score in scores]


def set_torch_seed(seed: int) -> None:
    require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
