from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multiseed_experiments import (  # noqa: E402
    summarize_ablation,
    summarize_main,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge compatible run_multiseed_experiments JSON outputs."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input multiseed_summary.json or multiseed_partial.json. Repeatable.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for merged outputs.")
    parser.add_argument(
        "--note",
        default="",
        help="Optional traceability note written into the merged JSON.",
    )
    parser.add_argument(
        "--copy",
        action="append",
        default=[],
        help="Optional file copy in SRC:DST_NAME form, relative paths are allowed.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(row: dict[str, Any], metric: str, key: str) -> str:
    value = row[metric].get(key)
    return "" if value is None else f"{float(value):.6f}"


def write_summary_csv(path: Path, grouped: dict[str, Any], name_col: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                name_col,
                "n",
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
            ]
        )
        for name, row in grouped.items():
            writer.writerow(
                [
                    name,
                    row.get("n", ""),
                    metric_value(row, "accuracy", "mean"),
                    metric_value(row, "accuracy", "std"),
                    metric_value(row, "precision", "mean"),
                    metric_value(row, "precision", "std"),
                    metric_value(row, "recall", "mean"),
                    metric_value(row, "recall", "std"),
                    metric_value(row, "f1", "mean"),
                    metric_value(row, "f1", "std"),
                    metric_value(row, "roc_auc", "mean"),
                    metric_value(row, "roc_auc", "std"),
                ]
            )


def write_runs_csv(path: Path, runs: list[dict[str, Any]], name_col: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", name_col, "accuracy", "precision", "recall", "f1", "roc_auc"])
        for run in runs:
            for row in run.get("metrics", []):
                name = row.get("display_name") or row.get("variant_id") or row.get("name")
                writer.writerow(
                    [
                        run.get("seed"),
                        name,
                        row.get("accuracy"),
                        row.get("precision"),
                        row.get("recall"),
                        row.get("f1"),
                        row.get("roc_auc"),
                    ]
                )


def copy_trace_files(out_dir: Path, copy_specs: list[str]) -> None:
    for spec in copy_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --copy spec: {spec!r}; expected SRC:DST_NAME")
        src_text, dst_name = spec.split(":", 1)
        src = Path(src_text)
        if not src.exists():
            continue
        shutil.copy2(src, out_dir / dst_name)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    loaded = [load_json(path) for path in input_paths]
    if not loaded:
        raise SystemExit("No inputs provided.")

    summary = dict(loaded[0])
    main_runs: list[dict[str, Any]] = []
    ablation_runs: list[dict[str, Any]] = []
    for item in loaded:
        main_runs.extend(item.get("main_runs", []))
        ablation_runs.extend(item.get("ablation_runs", []))

    seeds = sorted({int(run["seed"]) for run in main_runs + ablation_runs})
    summary["seeds"] = seeds
    summary["main_runs"] = main_runs
    summary["ablation_runs"] = ablation_runs
    summary["main_summary"] = summarize_main(main_runs)
    summary["ablation_summary"] = summarize_ablation(ablation_runs)
    summary["combined_from"] = [str(path) for path in input_paths]
    if args.note:
        summary["combination_note"] = args.note

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    write_markdown_report(out_dir / "multiseed_report.md", summary)
    write_summary_csv(out_dir / "main_comparison_summary.csv", summary["main_summary"], "method")
    write_summary_csv(out_dir / "ablation_summary.csv", summary["ablation_summary"], "variant")
    write_runs_csv(out_dir / "main_comparison_by_seed.csv", main_runs, "method")
    write_runs_csv(out_dir / "ablation_by_seed.csv", ablation_runs, "variant")
    copy_trace_files(out_dir, args.copy)

    print(out_dir)
    print(f"main_runs={len(main_runs)} seeds={sorted({run['seed'] for run in main_runs})}")
    print(f"ablation_runs={len(ablation_runs)} seeds={sorted({run['seed'] for run in ablation_runs})}")


if __name__ == "__main__":
    main()
