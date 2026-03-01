#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from statistics import mean
from typing import Dict, List, Optional, Tuple

import wandb


ENV_ORDER = [
    "box-close-v2",
    "button-press-topdown-v2",
    "button-press-topdown-wall-v2",
    "sweep-into-v2",
    "drawer-open-v2",
    "peg-insert-side-v2",
]

METHOD_SPECS = [
    {
        "label": "Heap+BT (K=5)",
        "feedback_type": "heap",
        "model_type": "BT",
        "q_budget": 5,
        "method_tag": "heap",
    },
    {
        "label": "Heap+linear_BT (K=5)",
        "feedback_type": "heap",
        "model_type": "linear_BT",
        "q_budget": 5,
        "method_tag": "heap",
    },
    {
        "label": "Heap+PL (K=5)",
        "feedback_type": "heap",
        "model_type": "PL",
        "q_budget": 5,
        "method_tag": "heap",
    },
    {
        "label": "Heap+linear_PL (K=5)",
        "feedback_type": "heap",
        "model_type": "linear_PL",
        "q_budget": 5,
        "method_tag": "heap",
    },
    {
        "label": "LiRE+linear_BT (K=100)",
        "feedback_type": "RLT",
        "model_type": "linear_BT",
        "q_budget": 100,
        "method_tag": "RLT",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reward accuracy comparison report from W&B."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default="", help="W&B entity/team (optional)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=10, help="Seed used for runs")
    parser.add_argument(
        "--reward-epochs", type=int, default=220, help="Reward training epochs"
    )
    parser.add_argument("--iql-steps", type=int, default=65000, help="IQL max steps")
    parser.add_argument(
        "--iql-eval-freq", type=int, default=10000, help="IQL eval frequency"
    )
    parser.add_argument(
        "--iql-eval-episodes", type=int, default=10, help="IQL eval episodes"
    )
    parser.add_argument(
        "--report-run-name",
        default="Comparison Report",
        help="W&B run name for report upload",
    )
    return parser.parse_args()


def parse_int(value, default: int = -1) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace("_", "")
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_created_at(value: Optional[str]) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def normalize_env(env_name: str) -> str:
    name = env_name.strip()
    if name.startswith("metaworld_"):
        return name[len("metaworld_") :]
    if name.startswith("metaworld-"):
        return name[len("metaworld-") :]
    return name


def resolve_method_label(config: Dict) -> Optional[str]:
    feedback_type = str(config.get("feedback_type", "")).strip()
    model_type = str(config.get("model_type", "")).strip()
    q_budget = parse_int(config.get("q_budget"))

    if feedback_type == "heap" and q_budget == 5:
        if model_type in {"BT", "linear_BT", "PL", "linear_PL"}:
            return f"Heap+{model_type} (K=5)"
    if feedback_type == "RLT" and model_type == "linear_BT" and q_budget == 100:
        return "LiRE+linear_BT (K=100)"
    return None


def extract_final_test_acc(run) -> Optional[float]:
    summary = dict(run.summary) if run.summary is not None else {}
    direct = summary.get("test_eval/acc")
    if isinstance(direct, (int, float)):
        return float(direct)

    last_value = None
    try:
        for row in run.scan_history(keys=["test_eval/acc"]):
            value = row.get("test_eval/acc")
            if isinstance(value, (int, float)):
                last_value = float(value)
    except Exception:
        return None
    return last_value


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_markdown_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def resolve_runs_path(api: wandb.Api, project: str, entity: str) -> str:
    candidates = []
    if entity:
        candidates.append(f"{entity}/{project}")
    else:
        env_entity = os.getenv("WANDB_ENTITY", "")
        if env_entity:
            candidates.append(f"{env_entity}/{project}")
        default_entity = getattr(api, "default_entity", None)
        if default_entity:
            candidates.append(f"{default_entity}/{project}")
        candidates.append(project)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            runs = api.runs(candidate, per_page=1)
            _ = len(runs)
            return candidate
        except Exception:
            continue
    raise RuntimeError(
        "Unable to resolve W&B project path. Provide --entity explicitly."
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    method_order = [spec["label"] for spec in METHOD_SPECS]
    expected_configs = {
        spec["label"]: spec
        for spec in METHOD_SPECS
    }

    api = wandb.Api()
    runs_path = resolve_runs_path(api, args.project, args.entity)
    print(f"Reading runs from: {runs_path}")

    runs = api.runs(runs_path)
    latest_by_key: Dict[Tuple[str, str, int], Dict] = {}

    inspected = 0
    matched = 0
    for run in runs:
        inspected += 1
        group = run.group or ""
        if group.startswith("IQL_"):
            continue

        config = dict(run.config) if run.config is not None else {}
        method_label = resolve_method_label(config)
        if method_label is None:
            continue
        env_name = normalize_env(str(config.get("env", "")))
        if env_name not in ENV_ORDER:
            continue
        seed = parse_int(config.get("seed"), default=args.seed)
        if seed != args.seed:
            continue

        final_acc = extract_final_test_acc(run)
        if final_acc is None:
            continue

        matched += 1
        key = (method_label, env_name, seed)
        created_at = parse_created_at(getattr(run, "created_at", None))
        prev = latest_by_key.get(key)
        if prev is None or created_at > prev["created_at"]:
            latest_by_key[key] = {
                "acc": final_acc,
                "run_id": run.id,
                "run_name": run.name,
                "group": group,
                "created_at": created_at,
            }

    print(f"Inspected runs: {inspected}")
    print(f"Matched reward runs: {matched}")
    print(f"Unique method-env-seed points: {len(latest_by_key)}")

    accuracy_headers = ["Method"] + ENV_ORDER + ["AVG"]
    accuracy_rows_for_csv: List[List[str]] = []
    accuracy_rows_for_wandb: List[List] = []

    for method in method_order:
        values = []
        row_csv = [method]
        row_wandb = [method]
        for env_name in ENV_ORDER:
            rec = latest_by_key.get((method, env_name, args.seed))
            val = None if rec is None else rec["acc"]
            if val is not None:
                values.append(val)
            row_csv.append(format_float(val))
            row_wandb.append("N/A" if val is None else round(val, 6))
        avg_val = mean(values) if values else None
        row_csv.append(format_float(avg_val))
        row_wandb.append("N/A" if avg_val is None else round(avg_val, 6))
        accuracy_rows_for_csv.append(row_csv)
        accuracy_rows_for_wandb.append(row_wandb)

    report_csv_path = os.path.join(args.output_dir, "report_accuracy.csv")
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(accuracy_headers)
        writer.writerows(accuracy_rows_for_csv)
    print(f"Wrote: {report_csv_path}")

    common_cfg = {
        "feedback_num": 500,
        "seed": args.seed,
        "reward_epochs": args.reward_epochs,
        "iql_steps": args.iql_steps,
        "iql_eval_freq": args.iql_eval_freq,
        "iql_eval_episodes": args.iql_eval_episodes,
    }

    config_headers = [
        "Method",
        "feedback_type",
        "model_type",
        "q_budget",
        "method_tag",
        "feedback_num",
        "seed",
        "reward_epochs",
        "iql_steps",
        "iql_eval_freq",
        "iql_eval_episodes",
    ]
    config_rows_for_wandb: List[List] = []
    config_rows_for_markdown: List[List[str]] = []
    for method in method_order:
        spec = expected_configs[method]
        row = [
            method,
            spec["feedback_type"],
            spec["model_type"],
            spec["q_budget"],
            spec["method_tag"],
            common_cfg["feedback_num"],
            common_cfg["seed"],
            common_cfg["reward_epochs"],
            common_cfg["iql_steps"],
            common_cfg["iql_eval_freq"],
            common_cfg["iql_eval_episodes"],
        ]
        config_rows_for_wandb.append(row)
        config_rows_for_markdown.append([str(x) for x in row])

    coverage_headers = ["Method", "Available Envs", "Coverage"]
    coverage_rows: List[List[str]] = []
    for method in method_order:
        available = 0
        for env_name in ENV_ORDER:
            if (method, env_name, args.seed) in latest_by_key:
                available += 1
        coverage_rows.append([method, str(available), f"{available}/{len(ENV_ORDER)}"])

    md_accuracy_rows = []
    for row in accuracy_rows_for_csv:
        md_row = [row[0]]
        for cell in row[1:]:
            if cell == "":
                md_row.append("N/A")
            else:
                md_row.append(format_markdown_float(float(cell)))
        md_accuracy_rows.append(md_row)

    report_md_path = os.path.join(args.output_dir, "report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Reward Accuracy Report\n\n")
        f.write(f"- W&B project: `{runs_path}`\n")
        f.write(f"- Seed: `{args.seed}`\n")
        f.write(
            f"- Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}`\n"
        )
        f.write(f"- Matched reward runs: `{matched}`\n")
        f.write(f"- Unique points used: `{len(latest_by_key)}`\n\n")
        f.write("## Accuracy Comparison (test_eval/acc)\n\n")
        f.write(markdown_table(accuracy_headers, md_accuracy_rows))
        f.write("\n\n## Config Matrix\n\n")
        f.write(markdown_table(config_headers, config_rows_for_markdown))
        f.write("\n\n## Coverage\n\n")
        f.write(markdown_table(coverage_headers, coverage_rows))
        f.write("\n")
    print(f"Wrote: {report_md_path}")

    run = wandb.init(
        project=args.project,
        entity=args.entity if args.entity else None,
        name=args.report_run_name,
        job_type="report",
        config={
            "seed": args.seed,
            "reward_epochs": args.reward_epochs,
            "iql_steps": args.iql_steps,
            "iql_eval_freq": args.iql_eval_freq,
            "iql_eval_episodes": args.iql_eval_episodes,
            "source_project_path": runs_path,
        },
    )

    accuracy_table = wandb.Table(columns=accuracy_headers)
    for row in accuracy_rows_for_wandb:
        accuracy_table.add_data(*row)

    config_table = wandb.Table(columns=config_headers)
    for row in config_rows_for_wandb:
        config_table.add_data(*row)

    coverage_table = wandb.Table(columns=coverage_headers)
    for row in coverage_rows:
        coverage_table.add_data(*row)

    wandb.log(
        {
            "accuracy_comparison": accuracy_table,
            "config_matrix": config_table,
            "coverage": coverage_table,
            "report/matched_reward_runs": matched,
            "report/unique_points": len(latest_by_key),
        }
    )
    wandb.finish()

    print("Uploaded accuracy_comparison and config_matrix tables to W&B.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
