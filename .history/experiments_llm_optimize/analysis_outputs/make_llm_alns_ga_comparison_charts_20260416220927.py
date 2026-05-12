"""
LLM优化算子 vs 传统ALNS vs GA 对比可视化
覆盖算例: c1_c101, c1_c105, c2_c201, c2_c205
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INSTANCES = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_llm_best(root: Path, inst: str) -> float:
    summary_path = root / "experiments_llm_optimize" / inst / "summary.json"
    if summary_path.exists():
        d = read_json(summary_path)
        return float(d["best_cost"])

    # 回退: 从轮次结果中取最小值
    inst_dir = root / "experiments_llm_optimize" / inst
    costs = []
    for p in sorted(inst_dir.glob("round_*_*/result.json")):
        d = read_json(p)
        val = d.get("cost", d.get("best_cost", d.get("final_cost")))
        if val is not None:
            costs.append(float(val))
    if not costs:
        raise FileNotFoundError(f"无法找到 LLM 结果: {inst}")
    return min(costs)


def load_baseline_runs(root: Path, folder: str, inst: str) -> list[float]:
    results_dir = root / folder / "results"
    vals: list[float] = []
    for p in sorted(results_dir.glob(f"{inst}_run*.json")):
        d = read_json(p)
        if d.get("success") and d.get("best_cost") is not None:
            vals.append(float(d["best_cost"]))
    if not vals:
        raise FileNotFoundError(f"无法找到 {folder} 结果: {inst}")
    return vals


def build_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    for inst in INSTANCES:
        llm_best = load_llm_best(root, inst)
        alns_runs = load_baseline_runs(root, "experiments_alns_baseline", inst)
        ga_runs = load_baseline_runs(root, "experiments_ga_baseline", inst)

        alns_best = min(alns_runs)
        ga_best = min(ga_runs)

        rows.append(
            {
                "instance": inst,
                "instance_label": inst.replace("c1_", "C1/").replace("c2_", "C2/").upper(),
                "llm_best": llm_best,
                "alns_best": alns_best,
                "alns_mean": mean(alns_runs),
                "ga_best": ga_best,
                "ga_mean": mean(ga_runs),
                "llm_vs_alns_best_pct": (alns_best - llm_best) / alns_best * 100,
                "llm_vs_ga_best_pct": (ga_best - llm_best) / ga_best * 100,
            }
        )

    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300


def plot_cost_comparison(df: pd.DataFrame, out_png: Path) -> None:
    x = np.arange(len(df))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w, df["llm_best"], width=w, label="LLM优化算子(最优)", color="#0B3C6D")
    ax.bar(x, df["alns_best"], width=w, label="传统ALNS(最优)", color="#6C8EBF")
    ax.bar(x + w, df["ga_best"], width=w, label="GA(最优)", color="#C26B5A")

    ax.set_xticks(x)
    ax.set_xticklabels(df["instance_label"])
    ax.set_ylabel("总成本")
    ax.set_title("LLM优化算子 vs 传统ALNS vs GA（最优成本）", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_gain_comparison(df: pd.DataFrame, out_png: Path) -> None:
    x = np.arange(len(df))
    w = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w / 2, df["llm_vs_alns_best_pct"], width=w, label="相对ALNS提升(%)", color="#4C78A8")
    b2 = ax.bar(x + w / 2, df["llm_vs_ga_best_pct"], width=w, label="相对GA提升(%)", color="#F58518")

    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["instance_label"])
    ax.set_ylabel("提升率(%)")
    ax.set_title("LLM优化算子相对基线提升率（最优成本口径）", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.1 if h >= 0 else -0.1),
                f"{h:+.2f}%",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8.5,
            )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent / "figures"

    setup_plot_style()
    df = build_dataframe(root)

    # 导出数据表
    table_path = Path(__file__).resolve().parent / "llm_alns_ga_comparison.csv"
    df.to_csv(table_path, index=False, encoding="utf-8-sig")

    # 生成图表
    chart1 = out_dir / "llm_alns_ga_best_cost_comparison.png"
    chart2 = out_dir / "llm_alns_ga_gain_comparison.png"
    plot_cost_comparison(df, chart1)
    plot_gain_comparison(df, chart2)

    print(f"Saved table: {table_path}")
    print(f"Saved chart: {chart1}")
    print(f"Saved chart: {chart2}")


if __name__ == "__main__":
    main()
