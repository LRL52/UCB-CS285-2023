#!/usr/bin/env python3
"""绘制 BC 性能随训练步数变化的曲线"""
import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot BC performance vs training steps")
    parser.add_argument(
        "--data-root", default="data",
        help="Root directory of experiment logs (default: data)")
    parser.add_argument(
        "--output", default="figures/bc_training_steps.pdf",
        help="Output figure path")
    return parser.parse_args()


def extract_eval_return(log_dir):
    """从 TensorBoard event 文件中提取 Eval_AverageReturn"""
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        if "Eval_AverageReturn" in ea.Tags()["scalars"]:
            events = ea.Scalars("Eval_AverageReturn")
            return events[0].value if events else None
    except Exception as exc:
        print(f"[WARN] failed to read {log_dir}: {exc}", file=sys.stderr)
    return None


def main():
    args = parse_args()
    data_root = pathlib.Path(args.data_root)

    # 定义实验名称和对应的训练步数
    experiments = [
        ("q1_bc_ant_steps100_Ant-v4", 100),
        ("q1_bc_ant_steps250_Ant-v4", 250),
        ("q1_bc_ant_steps500_Ant-v4", 500),
        ("q1_bc_ant_steps1000_Ant-v4", 1000),
        ("q1_bc_ant_steps2000_Ant-v4", 2000),
        ("q1_bc_ant_steps4000_Ant-v4", 4000),
    ]

    data = []
    for exp_prefix, steps in experiments:
        # 查找匹配的目录（包含时间戳）
        matching_dirs = list(data_root.glob(f"{exp_prefix}_*"))
        if not matching_dirs:
            print(f"[WARN] no directory found for {exp_prefix}", file=sys.stderr)
            continue

        log_dir = matching_dirs[0]
        eval_return = extract_eval_return(log_dir)
        if eval_return is not None:
            data.append((steps, eval_return))

    if not data:
        print("[ERROR] no data to plot", file=sys.stderr)
        return 1

    data.sort(key=lambda x: x[0])
    steps_vals, return_vals = zip(*data)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_vals, return_vals, marker="o", linewidth=2, markersize=8)
    ax.axhline(y=4681.89, color="r", linestyle="--", label="Expert Performance")

    ax.set_xlabel("Number of Training Steps", fontsize=12)
    ax.set_ylabel("Evaluation Average Return", fontsize=12)
    ax.set_title("BC Performance vs Training Steps on Ant-v4", fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Figure saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
