#!/usr/bin/env python3
"""绘制 DAgger 学习曲线"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DAgger learning curves")
    parser.add_argument("--env", required=True, help="Environment name (ant or hopper)")
    parser.add_argument("--data-root", default="data", help="Root directory of logs")
    parser.add_argument("--output", default=None, help="Output figure path")
    return parser.parse_args()


def extract_eval_returns(log_dir):
    """从 TensorBoard event 文件中提取所有迭代的 Eval_AverageReturn 和 Eval_StdReturn"""
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        if "Eval_AverageReturn" in ea.Tags()["scalars"]:
            mean_events = ea.Scalars("Eval_AverageReturn")
            std_events = ea.Scalars("Eval_StdReturn")
            iterations = [e.step for e in mean_events]
            means = [e.value for e in mean_events]
            stds = [e.value for e in std_events]
            return iterations, means, stds
    except Exception as exc:
        print(f"[WARN] failed to read {log_dir}: {exc}", file=sys.stderr)
    return None, None, None


def main():
    args = parse_args()
    data_root = pathlib.Path(args.data_root)

    if args.env == "ant":
        bc_prefix = "q1_bc_ant_Ant-v4"
        dagger_prefix = "q2_dagger_ant_Ant-v4"
        expert_perf = 4681.89
        title = "Ant-v4"
    elif args.env == "hopper":
        bc_prefix = "q1_bc_hopper_Hopper-v4"
        dagger_prefix = "q2_dagger_hopper_Hopper-v4"
        expert_perf = 3717.51
        title = "Hopper-v4"
    else:
        print(f"[ERROR] unknown environment: {args.env}", file=sys.stderr)
        return 1

    # 提取 BC 性能
    bc_dirs = list(data_root.glob(f"{bc_prefix}_*"))
    if not bc_dirs:
        print(f"[ERROR] BC directory not found for {bc_prefix}", file=sys.stderr)
        return 1
    _, bc_means, bc_stds = extract_eval_returns(bc_dirs[0])
    bc_perf = bc_means[0] if bc_means else None

    # 提取 DAgger 性能
    dagger_dirs = list(data_root.glob(f"{dagger_prefix}_*"))
    if not dagger_dirs:
        print(f"[ERROR] DAgger directory not found for {dagger_prefix}", file=sys.stderr)
        return 1
    iterations, dagger_means, dagger_stds = extract_eval_returns(dagger_dirs[0])

    if not iterations:
        print("[ERROR] no DAgger data found", file=sys.stderr)
        return 1

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))

    # DAgger 学习曲线（带误差条）
    ax.errorbar(iterations, dagger_means, yerr=dagger_stds, marker="o",
                linewidth=2, markersize=6, capsize=4, label="DAgger")

    # Expert 性能（水平线）
    ax.axhline(y=expert_perf, color="r", linestyle="--", linewidth=2, label="Expert")

    # BC 性能（水平线）
    if bc_perf:
        ax.axhline(y=bc_perf, color="g", linestyle="--", linewidth=2, label="BC")

    ax.set_xlabel("DAgger Iteration", fontsize=12)
    ax.set_ylabel("Evaluation Average Return", fontsize=12)
    ax.set_title(f"DAgger Learning Curve on {title}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    output_path = args.output or f"figures/dagger_{args.env}.pdf"
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Figure saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
