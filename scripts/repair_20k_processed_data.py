from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from evidroid.baselines.static import build_mamadroid_features_from_evidence
from evidroid.extractors.androguard_extractor import AndroguardEvidenceExtractor


SAMPLE_ID_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair processed APK inputs by filling missing rows. Defaults target the final 20,000-sample full-LLM corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-missing-evidence")
    extract.add_argument("--raw-dir", default="data/raw")
    extract.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    extract.add_argument("--out", default="artifacts/analysis/final_20000_repair/missing_evidence.jsonl")
    extract.add_argument("--summary", default="artifacts/analysis/final_20000_repair/missing_evidence_summary.json")

    append = subparsers.add_parser("append-missing")
    append.add_argument("--source", required=True)
    append.add_argument("--target", required=True)

    mamadroid = subparsers.add_parser("build-mamadroid")
    mamadroid.add_argument("--evidence", required=True)
    mamadroid.add_argument("--out", required=True)
    mamadroid.add_argument("--abstraction", choices=["package", "family"], default="package")
    mamadroid.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    verify.add_argument("--behaviors", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    verify.add_argument("--mamadroid", default="data/processed/mamadroid_features_final_20000_balanced_20260706.jsonl")
    verify.add_argument("--raw-dir", default="data/raw")
    verify.add_argument("--out", default="artifacts/analysis/final_20000_repair/processed_alignment_summary.json")

    args = parser.parse_args()
    if args.command == "extract-missing-evidence":
        extract_missing_evidence(args)
    elif args.command == "append-missing":
        append_missing(args)
    elif args.command == "build-mamadroid":
        build_mamadroid(args)
    elif args.command == "verify":
        verify_alignment(args)


def extract_missing_evidence(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    evidence_path = Path(args.evidence)
    out_path = Path(args.out)
    summary_path = Path(args.summary)

    seen = read_sample_ids(evidence_path)
    missing = []
    for label in ("benign", "malware"):
        label_dir = raw_dir / label
        for apk_path in sorted(label_dir.rglob("*.apk")):
            if apk_path.stem.upper() not in seen:
                missing.append((apk_path, label))

    completed = read_sample_ids(out_path) if out_path.exists() else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = AndroguardEvidenceExtractor()
    written = 0
    error_docs = 0
    empty_docs = 0
    started = time.perf_counter()

    print(
        f"[repair-evidence] existing={len(seen)} missing={len(missing)} resume={len(completed)}",
        flush=True,
    )
    with out_path.open("a", encoding="utf-8", newline="\n") as handle:
        for idx, (apk_path, label) in enumerate(missing, start=1):
            if apk_path.stem.upper() in completed:
                continue
            item_start = time.perf_counter()
            doc = extractor.extract(apk_path, label=label)
            elapsed = time.perf_counter() - item_start
            doc["timing"] = {**doc.get("timing", {}), "extract_seconds": elapsed}
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            error_docs += int(bool(doc.get("errors")))
            empty_docs += int(not bool(doc.get("evidence")))
            print(
                "[repair-evidence] "
                f"{idx}/{len(missing)} {label} {apk_path.stem} "
                f"evidence={len(doc.get('evidence', []))} "
                f"errors={len(doc.get('errors', []))} seconds={elapsed:.2f}",
                flush=True,
            )

    output_ids = read_sample_ids(out_path)
    summary = {
        "existing_evidence_rows": len(seen),
        "missing_raw_apks": len(missing),
        "resume_rows_before": len(completed),
        "written_this_run": written,
        "output_rows": len(output_ids),
        "error_docs_this_run": error_docs,
        "empty_docs_this_run": empty_docs,
        "seconds": round(time.perf_counter() - started, 3),
        "out": str(out_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[repair-evidence] summary " + json.dumps(summary), flush=True)


def append_missing(args: argparse.Namespace) -> None:
    source = Path(args.source)
    target = Path(args.target)
    target_ids = read_sample_ids(target)
    appended = 0
    duplicate = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("a", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            sample_id = sample_id_from_line(line)
            if not sample_id:
                continue
            key = sample_id.upper()
            if key in target_ids:
                duplicate += 1
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            target_ids.add(key)
            appended += 1
    print(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "appended": appended,
                "duplicates_skipped": duplicate,
                "target_unique_ids": len(target_ids),
            },
            indent=2,
        )
    )


def build_mamadroid(args: argparse.Namespace) -> None:
    evidence_path = Path(args.evidence)
    out_path = Path(args.out)
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    elif out_path.exists():
        raise SystemExit(f"{out_path} exists. Pass --overwrite to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with evidence_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            row = {
                "sample_id": doc["sample_id"],
                "features": build_mamadroid_features_from_evidence(doc, abstraction=args.abstraction),
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count == 1 or count % 100 == 0:
                print(f"[mamadroid-cache] {count}", flush=True)
    print(json.dumps({"out": str(out_path), "rows": count}, indent=2))


def verify_alignment(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    raw_ids = set()
    raw_label_counts: dict[str, int] = {}
    for label in ("benign", "malware"):
        label_ids = {path.stem.upper() for path in (raw_dir / label).rglob("*.apk")}
        raw_ids.update(label_ids)
        raw_label_counts[label] = len(label_ids)

    evidence_ids, evidence_label_counts = read_ids_and_labels(Path(args.evidence))
    behavior_ids, _behavior_label_counts = read_ids_and_labels(Path(args.behaviors))
    mamadroid_ids, _mamadroid_label_counts = read_ids_and_labels(Path(args.mamadroid))
    common_ids = evidence_ids & behavior_ids & mamadroid_ids
    summary = {
        "raw_count": len(raw_ids),
        "raw_label_counts": raw_label_counts,
        "evidence_count": len(evidence_ids),
        "evidence_label_counts": evidence_label_counts,
        "behavior_count": len(behavior_ids),
        "mamadroid_count": len(mamadroid_ids),
        "common_processed_count": len(common_ids),
        "missing_from_evidence": sorted(raw_ids - evidence_ids),
        "missing_from_behaviors": sorted(raw_ids - behavior_ids),
        "missing_from_mamadroid": sorted(raw_ids - mamadroid_ids),
        "extra_evidence_not_raw": sorted(evidence_ids - raw_ids),
        "extra_behaviors_not_raw": sorted(behavior_ids - raw_ids),
        "extra_mamadroid_not_raw": sorted(mamadroid_ids - raw_ids),
        "evidence_behavior_symmetric_difference": len(evidence_ids ^ behavior_ids),
        "evidence_mamadroid_symmetric_difference": len(evidence_ids ^ mamadroid_ids),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
    print(f"[verify] wrote {out_path}")


def read_sample_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_id = sample_id_from_line(line)
            if sample_id:
                ids.add(sample_id.upper())
    return ids


def read_ids_and_labels(path: Path) -> tuple[set[str], dict[str, int]]:
    ids = set()
    label_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_id = sample_id_from_line(line)
            if sample_id:
                ids.add(sample_id.upper())
            label_match = LABEL_RE.search(line)
            if label_match:
                label = label_match.group(1)
                if label in {"benign", "malware"}:
                    label_counts[label] = label_counts.get(label, 0) + 1
    return ids, label_counts


def sample_id_from_line(line: str) -> str | None:
    match = SAMPLE_ID_RE.search(line)
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
