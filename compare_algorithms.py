# -*- coding: utf-8 -*-
"""三算法对比可视化（LLM-ALNS/ACO/GA），基于 core30 实例。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from result_analyzer_batch import BEST_KNOWN_SOLUTIONS, CORE30_INSTANCES

# ── Academic paper style ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.linewidth':     1.0,
    'axes.edgecolor':     'black',
    'axes.facecolor':     'white',
    'axes.grid':          True,
    'axes.grid.axis':     'y',
    'grid.linestyle':     '--',
    'grid.linewidth':     0.5,
    'grid.color':         '#CCCCCC',
    'grid.alpha':         0.8,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.labelsize':    9,
    'ytick.labelsize':    10,
    'legend.frameon':     True,
    'legend.framealpha':  1.0,
    'legend.edgecolor':   'black',
    'legend.fontsize':    11,
    'figure.facecolor':   'white',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

# Color palette consistent with cost plots
COLORS = {'ACO': '#4878CF', 'LLM-ALNS': '#6ACC65', 'GA': '#D65F5F'}


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
    rows += load_results(alns_dir, "LLM-ALNS")
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
    
    # Custom academic bar plotting (without seaborn dependency if possible, or using precise matplotlib)
    instances = sorted(df['instance'].unique())
    x = np.arange(len(instances))
    w = 0.26
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Pivot dataframe for easier plotting
    pivot_df = df.pivot(index='instance', columns='algorithm', values='gap_pct')
    
    for k, algo in enumerate(['ACO', 'LLM-ALNS', 'GA']):
        if algo in pivot_df.columns:
            ax.bar(x + (k - 1) * w, pivot_df[algo], w,
                   label=algo,
                   color=COLORS[algo],
                   edgecolor='black',
                   linewidth=0.7,
                   alpha=0.90)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Gap to BKS (%)', fontsize=12)
    ax.set_xlabel('Instance', fontsize=12)
    ax.legend(loc='upper right', ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add vertical dividers for classes
    c_count = len([i for i in instances if i.startswith('c')])
    r_count = len([i for i in instances if i.startswith('r') and not i.startswith('rc')])
    ax.axvline(c_count - 0.5, color='gray', linestyle=':', linewidth=1.0)
    ax.axvline(c_count + r_count - 0.5, color='gray', linestyle=':', linewidth=1.0)
    
    # Label regions
    def add_region_label(ax, xstart, xend, label, y_frac=0.97):
        ax.text((xstart + xend) / 2, ax.get_ylim()[1] * y_frac,
                label, ha='center', va='top', fontsize=10,
                color='gray', style='italic')
        
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.06)
    add_region_label(ax, 0, c_count - 0.5, 'C Instances', 0.99)
    add_region_label(ax, c_count, c_count + r_count - 0.5, 'R Instances', 0.99)
    add_region_label(ax, c_count + r_count, len(instances) - 1, 'RC Instances', 0.99)
    
    ax.set_title("Algorithm Comparison - Gap to BKS by Instance (ACO vs. LLM-ALNS vs. GA)", 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(out_dir / "gap_by_instance.png", dpi=300)
    plt.close()


def plot_overall_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby("algorithm", as_index=False).agg({
        "gap_pct": "mean",
        "best_cost": "mean",
    })
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Draw nice clean academic bar plot
    algos = ['ACO', 'LLM-ALNS', 'GA']
    # Filter only available ones
    summary_filtered = summary[summary['algorithm'].isin(algos)].set_index('algorithm').reindex(algos)
    
    bars = ax.bar(summary_filtered.index, summary_filtered['gap_pct'], width=0.4,
                  color=[COLORS[a] for a in summary_filtered.index],
                  edgecolor='black', linewidth=0.7, alpha=0.9)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Mean Gap to BKS (%)", fontsize=11)
    ax.set_title("Overall Mean Gap to BKS", fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make space for labels above bars
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(0, yhi * 1.1)
    
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
