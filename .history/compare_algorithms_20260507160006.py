# -*- coding: utf-8 -*-
"""三算法对比可视化（ALNS/ACO/GA），基于 core30 实例。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from result_analyzer_batch import BEST_KNOWN_SOLUTIONS, CORE30_INSTANCES


def load_results(results_dir: Path, algorithm_label: str) -> list[dict]:
    rows = []
    if not results_dir.exists():
        print(f"[Skip] 结果目录不存在: {results_dir}")
        return rows

    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            instance = data.get("instance") or ""
            best_cost = data.get("best_cost")
            if instance not in CORE30_INSTANCES or best_cost is None:
                continue
            bks = BEST_KNOWN_SOLUTIONS.get(instance)
            gap = None
            if bks is not None and bks > 0:
                gap = (best_cost - bks) / bks * 100
            rows.append({
                "instance": instance,
                "algorithm": algorithm_label,
                "best_cost": float(best_cost),
                "gap_pct": gap,
            })
        except Exception:
            continue

    return rows


def build_dataframe(alns_dir: Path, aco_dir: Path, ga_dir: Path) -> pd.DataFrame:
    rows = []
    rows += load_results(alns_dir, "ALNS")
    rows += load_results(aco_dir, "ACO")
    rows += load_results(ga_dir, "GA")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby(["instance", "algorithm"], as_index=False).agg({
        "best_cost": "min",
        "gap_pct": "min",
    })
    return df


def plot_gap_by_instance(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="instance", y="gap_pct", hue="algorithm")
    plt.xticks(rotation=90)
    plt.ylabel("Gap to BKS (%)")
    plt.title("Algorithm Comparison - Gap to BKS by Instance")
    plt.tight_layout()
    plt.savefig(out_dir / "gap_by_instance.png", dpi=300)
    plt.close()


def plot_overall_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby("algorithm", as_index=False).agg({
        "gap_pct": "mean",
        "best_cost": "mean",
    })

    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary, x="algorithm", y="gap_pct")
    plt.ylabel("Mean Gap to BKS (%)")
    plt.title("Overall Mean Gap to BKS")
    plt.tight_layout()
    plt.savefig(out_dir / "gap_mean.png", dpi=300)
    plt.close()


def save_cost_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """保存成本对比表（按实例与总体）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 表1：按实例对比（每实例每算法的最优成本）
    pivot = df.pivot(index="instance", columns="algorithm", values="best_cost")
    pivot.to_csv(out_dir / "cost_by_instance.csv", encoding="utf-8")

    # 表2：总体均值对比
    summary = df.groupby("algorithm", as_index=False).agg({
        "best_cost": "mean",
        "gap_pct": "mean",
    })
    summary.to_csv(out_dir / "cost_summary.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="三算法对比可视化")
    parser.add_argument("--alns", default="experiments_alns_baseline/results", help="ALNS 结果目录")
    parser.add_argument("--aco", default="experiments_aco_baseline/results", help="ACO 结果目录")
    parser.add_argument("--ga", default="experiments_ga_baseline/results", help="GA 结果目录")
    parser.add_argument("--out", default="experiments_compare/figures", help="输出目录")
    args = parser.parse_args()

    df = build_dataframe(Path(args.alns), Path(args.aco), Path(args.ga))
    if df.empty:
        print("[Error] 未加载到有效结果")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "compare_table.csv", index=False, encoding="utf-8")
    save_cost_tables(df, out_dir)
    plot_gap_by_instance(df, out_dir)
    plot_overall_summary(df, out_dir)

    print(f"[Done] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
