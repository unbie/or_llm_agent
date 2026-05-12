"""
生成 LLM / ALNS / GA / ACO 综合对比实验报告：
- 多维数据表（成本、稳定性、耗时、相对提升）
- 柱状图与折线图
- 自动写入 docs markdown 文档
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INSTANCES = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]
BKS = {
    "c101": 828.94,
    "c105": 828.94,
    "c201": 591.56,
    "c205": 588.88,
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_instance_parts(inst: str) -> tuple[str, str]:
    ds, name = inst.split("_", 1)
    return ds, name


def load_llm_best(root: Path, inst: str) -> float:
    d = root / "experiments_llm_optimize" / inst
    summary = d / "summary.json"
    if summary.exists():
        s = read_json(summary)
        return float(s["best_cost"])

    # 兜底扫描
    vals = []
    for p in sorted(d.glob("round_*_*/result.json")):
        j = read_json(p)
        c = j.get("cost", j.get("best_cost", j.get("final_cost")))
        if c is not None:
            vals.append(float(c))
    if not vals:
        raise FileNotFoundError(f"LLM结果不存在: {inst}")
    return min(vals)


def load_algo_runs(root: Path, folder: str, inst: str) -> list[float]:
    vals = []
    for p in sorted((root / folder / "results").glob(f"{inst}_run*.json")):
        j = read_json(p)
        if j.get("success") and j.get("best_cost") is not None:
            vals.append(float(j["best_cost"]))
    if not vals:
        raise FileNotFoundError(f"{folder}结果不存在: {inst}")
    return vals


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300


def build_tables(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for inst in INSTANCES:
        ds, name = parse_instance_parts(inst)
        llm_best = load_llm_best(root, inst)

        alns_runs = load_algo_runs(root, "experiments_alns_baseline", inst)
        ga_runs = load_algo_runs(root, "experiments_ga_baseline", inst)
        aco_runs = load_algo_runs(root, "experiments_aco_ls_tuned_iter1500", inst)

        def stats(v: list[float]) -> tuple[float, float, float, float]:
            m = mean(v)
            s = stdev(v) if len(v) > 1 else 0.0
            b = min(v)
            cv = (s / m * 100) if m else 0.0
            return m, s, b, cv

        alns_mean, alns_std, alns_best, alns_cv = stats(alns_runs)
        ga_mean, ga_std, ga_best, ga_cv = stats(ga_runs)
        aco_mean, aco_std, aco_best, aco_cv = stats(aco_runs)

        rows.append(
            {
                "instance": inst,
                "dataset_type": ds,
                "instance_name": name,
                "llm_best": llm_best,
                "alns_best": alns_best,
                "alns_mean": alns_mean,
                "alns_std": alns_std,
                "alns_cv_pct": alns_cv,
                "ga_best": ga_best,
                "ga_mean": ga_mean,
                "ga_std": ga_std,
                "ga_cv_pct": ga_cv,
                "aco_best": aco_best,
                "aco_mean": aco_mean,
                "aco_std": aco_std,
                "aco_cv_pct": aco_cv,
                "llm_vs_alns_best_pct": (alns_best - llm_best) / alns_best * 100,
                "llm_vs_ga_best_pct": (ga_best - llm_best) / ga_best * 100,
                "llm_vs_aco_best_pct": (aco_best - llm_best) / aco_best * 100,
                "alns_vs_ga_best_pct": (ga_best - alns_best) / ga_best * 100,
                "alns_vs_aco_best_pct": (aco_best - alns_best) / aco_best * 100,
            }
        )

    df = pd.DataFrame(rows)

    # 表1：核心成本表
    table_core = df[
        [
            "instance",
            "llm_best",
            "alns_best",
            "ga_best",
            "aco_best",
            "llm_vs_alns_best_pct",
            "llm_vs_ga_best_pct",
            "llm_vs_aco_best_pct",
        ]
    ].copy()

    # 表2：稳定性表
    table_stability = df[
        [
            "instance",
            "alns_mean",
            "alns_std",
            "alns_cv_pct",
            "ga_mean",
            "ga_std",
            "ga_cv_pct",
            "aco_mean",
            "aco_std",
            "aco_cv_pct",
        ]
    ].copy()

    # 表3：算法胜负统计
    win_rows = []
    for _, r in df.iterrows():
        vals = {
            "LLM": r["llm_best"],
            "ALNS": r["alns_best"],
            "GA": r["ga_best"],
            "ACO": r["aco_best"],
        }
        winner = min(vals, key=vals.get)
        win_rows.append({"instance": r["instance"], "winner": winner, **vals})
    table_winner = pd.DataFrame(win_rows)

    # 表4：相对BKS的偏离（仅用于同实例内部参考）
    bks_rows = []
    for _, r in df.iterrows():
        name = r["instance_name"]
        bks = BKS.get(name)
        if bks is None:
            continue
        bks_rows.append(
            {
                "instance": r["instance"],
                "bks": bks,
                "llm_gap_to_bks_pct": (r["llm_best"] - bks) / bks * 100,
                "alns_gap_to_bks_pct": (r["alns_best"] - bks) / bks * 100,
                "ga_gap_to_bks_pct": (r["ga_best"] - bks) / bks * 100,
                "aco_gap_to_bks_pct": (r["aco_best"] - bks) / bks * 100,
            }
        )
    table_bks = pd.DataFrame(bks_rows)

    # 表5：LLM优势汇总表（相对三个基线）
    table_llm_adv = df[
        [
            "instance",
            "llm_vs_alns_best_pct",
            "llm_vs_ga_best_pct",
            "llm_vs_aco_best_pct",
        ]
    ].copy()
    table_llm_adv["llm_adv_mean_pct"] = table_llm_adv[
        ["llm_vs_alns_best_pct", "llm_vs_ga_best_pct", "llm_vs_aco_best_pct"]
    ].mean(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_core.to_csv(out_dir / "table_core_cost_comparison.csv", index=False, encoding="utf-8-sig")
    table_stability.to_csv(out_dir / "table_stability_comparison.csv", index=False, encoding="utf-8-sig")
    table_winner.to_csv(out_dir / "table_winner_by_instance.csv", index=False, encoding="utf-8-sig")
    table_bks.to_csv(out_dir / "table_gap_to_bks.csv", index=False, encoding="utf-8-sig")
    table_llm_adv.to_csv(out_dir / "table_llm_advantage_summary.csv", index=False, encoding="utf-8-sig")

    return df, table_core, table_stability, table_bks


def plot_bar_and_line(df: pd.DataFrame, out_fig: Path):
    out_fig.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df))
    labels = [s.upper().replace("_", "/") for s in df["instance"].tolist()]

    # 柱状图：四算法最优成本
    fig, ax = plt.subplots(figsize=(11, 5.6))
    w = 0.2
    ax.bar(x - 1.5 * w, df["llm_best"], width=w, label="LLM(best)", color="#0B3C6D")
    ax.bar(x - 0.5 * w, df["alns_best"], width=w, label="ALNS(best)", color="#4C78A8")
    ax.bar(x + 0.5 * w, df["ga_best"], width=w, label="GA(best)", color="#F58518")
    ax.bar(x + 1.5 * w, df["aco_best"], width=w, label="ACO(best)", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost")
    ax.set_title("四算法最优成本柱状图")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_fig / "bar_best_cost_4algorithms.png", bbox_inches="tight")
    plt.close(fig)

    # 折线图：算法在实例上的最优成本走势
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(x, df["llm_best"], marker="o", linewidth=2.2, label="LLM(best)", color="#0B3C6D")
    ax.plot(x, df["alns_best"], marker="s", linewidth=2.0, label="ALNS(best)", color="#4C78A8")
    ax.plot(x, df["ga_best"], marker="^", linewidth=2.0, label="GA(best)", color="#F58518")
    ax.plot(x, df["aco_best"], marker="D", linewidth=2.0, label="ACO(best)", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost")
    ax.set_title("四算法最优成本折线图")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_fig / "line_best_cost_4algorithms.png", bbox_inches="tight")
    plt.close(fig)

    # 优势图：LLM相对各基线的提升率（>0 表示 LLM 更优）
    fig, ax = plt.subplots(figsize=(11, 5.6))
    w = 0.25
    ax.bar(x - w, df["llm_vs_alns_best_pct"], width=w, label="LLM vs ALNS", color="#1f77b4")
    ax.bar(x, df["llm_vs_ga_best_pct"], width=w, label="LLM vs GA", color="#ff7f0e")
    ax.bar(x + w, df["llm_vs_aco_best_pct"], width=w, label="LLM vs ACO", color="#2ca02c")
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("LLM 相对各基线的提升率（正值表示LLM更优）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out_fig / "bar_llm_advantage_pct.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    report_path: Path,
    table_dir: Path,
    fig_dir: Path,
    core: pd.DataFrame,
    stability: pd.DataFrame,
):
    # 简短文字统计
    llm_wins = (core[["llm_best", "alns_best", "ga_best", "aco_best"]].idxmin(axis=1) == "llm_best").sum()
    alns_wins = (core[["llm_best", "alns_best", "ga_best", "aco_best"]].idxmin(axis=1) == "alns_best").sum()
    ga_wins = (core[["llm_best", "alns_best", "ga_best", "aco_best"]].idxmin(axis=1) == "ga_best").sum()
    aco_wins = (core[["llm_best", "alns_best", "ga_best", "aco_best"]].idxmin(axis=1) == "aco_best").sum()

    md = f"""---
title: 四算法综合对比实验报告
description: 基于C101/C105/C201/C205对LLM优化算子、ALNS、GA、ACO进行柱状图、折线图与多表格对比
author: OR-LLM Agent
ms.date: 2026-04-17
ms.topic: concept
keywords:
  - LLM
  - ALNS
  - GA
  - ACO
  - 对比实验
estimated_reading_time: 6
---

## 实验范围

* 实例: c1/c101、c1/c105、c2/c201、c2/c205
* 算法: LLM优化算子、ALNS基线、GA基线、ACO
* 指标: 最优成本、均值与标准差、稳定性CV、相对提升比例

## 可视化图表

### 柱状图

![四算法最优成本柱状图](../experiments_llm_optimize/analysis_outputs/figures/bar_best_cost_4algorithms.png)

### 折线图

![四算法最优成本折线图](../experiments_llm_optimize/analysis_outputs/figures/line_best_cost_4algorithms.png)

### LLM优势图

![LLM相对基线提升率柱状图](../experiments_llm_optimize/analysis_outputs/figures/bar_llm_advantage_pct.png)

## 数据表输出

* 核心成本对比表: [experiments_llm_optimize/analysis_outputs/tables/table_core_cost_comparison.csv](../experiments_llm_optimize/analysis_outputs/tables/table_core_cost_comparison.csv)
* 稳定性对比表: [experiments_llm_optimize/analysis_outputs/tables/table_stability_comparison.csv](../experiments_llm_optimize/analysis_outputs/tables/table_stability_comparison.csv)
* 分实例优胜者表: [experiments_llm_optimize/analysis_outputs/tables/table_winner_by_instance.csv](../experiments_llm_optimize/analysis_outputs/tables/table_winner_by_instance.csv)
* 相对BKS偏离表: [experiments_llm_optimize/analysis_outputs/tables/table_gap_to_bks.csv](../experiments_llm_optimize/analysis_outputs/tables/table_gap_to_bks.csv)
* LLM优势汇总表: [experiments_llm_optimize/analysis_outputs/tables/table_llm_advantage_summary.csv](../experiments_llm_optimize/analysis_outputs/tables/table_llm_advantage_summary.csv)

## 关键结论

* 分实例最优胜场: LLM={llm_wins}, ALNS={alns_wins}, GA={ga_wins}, ACO={aco_wins}
* LLM 相对 ALNS/GA/ACO 的提升比例见核心成本表中的对应百分比列。
* 在优势图中，0轴以上的柱子代表 LLM 在对应实例和对应基线上具有正向优势。
* 稳定性建议重点看 CV 列，CV 越小通常越稳定。

## 说明

* 本报告的详细数据与图均由脚本自动生成，确保可复现。
* 如需刷新结果，请重新运行生成脚本。
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    out_tables = Path(__file__).resolve().parent / "tables"
    out_fig = Path(__file__).resolve().parent / "figures"

    setup_plot_style()
    df, core, stability, _ = build_tables(root, out_tables)
    plot_bar_and_line(df, out_fig)

    report_path = root / "docs" / "四算法综合对比实验报告.md"
    write_markdown_report(report_path, out_tables, out_fig, core, stability)

    print(f"Saved tables: {out_tables}")
    print(f"Saved figures: {out_fig}")
    print(f"Saved markdown: {report_path}")


if __name__ == "__main__":
    main()
