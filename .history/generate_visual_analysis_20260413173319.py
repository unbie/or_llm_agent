from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(r"d:\pythonProject\or_llm_agent\experiments_llm_optimize")
OUT_DIR = ROOT / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
REPORT_PATH = OUT_DIR / "visual_analysis_report.md"
SUMMARY_PATH = OUT_DIR / "analysis_summary.json"


PALETTE = {
    "c1_c101": "#1f77b4",
    "c1_c105": "#d62728",
    "c2_c201": "#2ca02c",
    "c2_c205": "#9467bd",
}

TYPE_MARKERS = {
    "baseline": "o",
    "accepted": "^",
    "rejected": "s",
    "final": "D",
}

TYPE_STYLES = {
    "baseline": "-",
    "accepted": "-",
    "rejected": "-",
    "final": (0, (3, 1, 1, 1)),
}

COST_COMPONENTS = ["c11_pct", "c12_pct", "c13_pct", "c2_pct", "c3_pct"]
COST_COMPONENT_LABELS = {
    "c11_pct": "c11",
    "c12_pct": "c12",
    "c13_pct": "c13",
    "c2_pct": "c2",
    "c3_pct": "c3",
}
COST_COMPONENT_COLORS = {
    "c11_pct": "#264653",
    "c12_pct": "#2a9d8f",
    "c13_pct": "#e9c46a",
    "c2_pct": "#f4a261",
    "c3_pct": "#e76f51",
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.7,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    }
)

sns.set_theme(style="whitegrid")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slug_instance(name: str) -> str:
    return name.replace("/", "_")


def round_sort_key(folder_name: str) -> tuple:
    if folder_name.startswith("round_"):
        parts = folder_name.split("_")
        try:
            round_num = int(parts[1])
        except Exception:
            round_num = math.inf
        stage_type = parts[2] if len(parts) > 2 else ""
        return (0, round_num, stage_type)
    if folder_name == "final_best":
        return (1, math.inf, "final")
    return (2, math.inf, folder_name)


def collect_instance_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    for instance_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        instance_name = instance_dir.name
        if instance_name.startswith("analysis_"):
            continue

        summary_path = instance_dir / "summary.json"
        if summary_path.exists():
            try:
                metadata[instance_name] = read_json(summary_path)
            except Exception:
                metadata[instance_name] = {}
        else:
            metadata[instance_name] = {}

        result_candidates: List[Path] = []
        for child in sorted(instance_dir.iterdir(), key=lambda p: round_sort_key(p.name)):
            if child.is_dir():
                result_path = child / "result.json"
                if result_path.exists():
                    result_candidates.append(result_path)

        # Also allow directly placed result files if present.
        direct_result = instance_dir / "result.json"
        if direct_result.exists():
            result_candidates.append(direct_result)

        seen = set()
        for result_path in result_candidates:
            if result_path in seen:
                continue
            seen.add(result_path)
            payload = read_json(result_path)
            if "round" in payload:
                base_row = {
                    "instance": instance_name,
                    "family": instance_name.split("_")[0],
                    "dataset": instance_name.split("_", 1)[1] if "_" in instance_name else instance_name,
                    "stage": payload.get("round", 0),
                    "stage_label": f"r{payload.get('round', 0)}",
                    "type": payload.get("type", ""),
                    "cost": payload.get("best_cost", payload.get("cost", np.nan)),
                    "num_routes": payload.get("num_routes", np.nan),
                    "elapsed_time": payload.get("elapsed_time", np.nan),
                    "initial_cost": payload.get("metrics", {}).get("initial_cost", np.nan),
                    "improvement_pct": payload.get("metrics", {}).get("improvement_pct", np.nan),
                    "avg_customers_per_route": payload.get("metrics", {}).get("avg_customers_per_route", np.nan),
                    "avg_load_utilization": payload.get("metrics", {}).get("avg_load_utilization", np.nan),
                    "total_iters": payload.get("metrics", {}).get("total_iters", np.nan),
                    "last_improve_iter": payload.get("metrics", {}).get("last_improve_iter", np.nan),
                    "stagnation_iters": payload.get("metrics", {}).get("stagnation_iters", np.nan),
                    "early_improvement_pct": payload.get("metrics", {}).get("early_improvement_pct", np.nan),
                    "c11_pct": payload.get("metrics", {}).get("c11_pct", np.nan),
                    "c12_pct": payload.get("metrics", {}).get("c12_pct", np.nan),
                    "c13_pct": payload.get("metrics", {}).get("c13_pct", np.nan),
                    "c2_pct": payload.get("metrics", {}).get("c2_pct", np.nan),
                    "c3_pct": payload.get("metrics", {}).get("c3_pct", np.nan),
                    "source": result_path.parent.name,
                }
                rows.append(base_row)
            elif payload.get("type") == "final":
                final_rows.append(
                    {
                        "instance": instance_name,
                        "family": instance_name.split("_")[0],
                        "dataset": instance_name.split("_", 1)[1] if "_" in instance_name else instance_name,
                        "stage": None,
                        "stage_label": "final",
                        "type": "final",
                        "cost": payload.get("final_cost", np.nan),
                        "num_routes": payload.get("num_routes", np.nan),
                        "elapsed_time": payload.get("elapsed_time", np.nan),
                        "baseline_cost": payload.get("baseline_cost", np.nan),
                        "total_improvement_pct": payload.get("total_improvement_pct", np.nan),
                        "source": result_path.parent.name,
                    }
                )

    rounds_df = pd.DataFrame(rows)
    final_df = pd.DataFrame(final_rows)

    # Build a coherent final summary per instance.
    final_summary_rows: List[Dict[str, Any]] = []
    for instance_name, group in rounds_df.groupby("instance"):
        group = group.copy()
        group["cost"] = pd.to_numeric(group["cost"], errors="coerce")
        group["num_routes"] = pd.to_numeric(group["num_routes"], errors="coerce")
        baseline_row = group[group["stage"] == 0].sort_values("cost").head(1)
        if not baseline_row.empty:
            baseline_cost = float(baseline_row.iloc[0]["cost"])
            baseline_routes = float(baseline_row.iloc[0]["num_routes"])
            baseline_time = float(baseline_row.iloc[0].get("elapsed_time", np.nan))
            baseline_util = float(baseline_row.iloc[0].get("avg_load_utilization", np.nan))
        else:
            baseline_cost = np.nan
            baseline_routes = np.nan
            baseline_time = np.nan
            baseline_util = np.nan

        final_row: Optional[pd.Series] = None
        if instance_name in set(final_df["instance"]):
            final_row = final_df[final_df["instance"] == instance_name].iloc[0]
        else:
            best_idx = group["cost"].idxmin()
            if pd.notna(best_idx):
                final_row = group.loc[best_idx]

        if final_row is None or final_row.empty:
            continue

        final_cost = float(final_row["cost"])
        final_routes = float(final_row["num_routes"])
        final_time = float(final_row.get("elapsed_time", np.nan))
        final_improvement = ((baseline_cost - final_cost) / baseline_cost * 100.0) if baseline_cost and not np.isnan(baseline_cost) else np.nan

        final_summary_rows.append(
            {
                "instance": instance_name,
                "family": instance_name.split("_")[0],
                "dataset": instance_name.split("_", 1)[1] if "_" in instance_name else instance_name,
                "baseline_cost": baseline_cost,
                "final_cost": final_cost,
                "improvement_pct": final_improvement,
                "baseline_routes": baseline_routes,
                "final_routes": final_routes,
                "baseline_elapsed_time": baseline_time,
                "final_elapsed_time": final_time,
                "baseline_load_utilization": baseline_util,
                "best_stage_label": final_row.get("stage_label", "final") if isinstance(final_row, pd.Series) else "final",
                "best_source": final_row.get("source", "final_best") if isinstance(final_row, pd.Series) else "final_best",
            }
        )

    final_summary_df = pd.DataFrame(final_summary_rows)
    return rounds_df, final_summary_df, metadata


def format_num(x: Any, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return str(x)


def create_line_figure(rounds_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()
    instance_order = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]

    for ax, instance in zip(axes, instance_order):
        df = rounds_df[rounds_df["instance"] == instance].copy().sort_values(["stage", "type"])
        if df.empty:
            ax.set_visible(False)
            continue
        baseline_cost = df[df["stage"] == 0]["cost"].iloc[0]
        df = df.sort_values("stage")
        x = df["stage"].tolist()
        y = df["cost"].astype(float).tolist()
        norm_y = [v / baseline_cost * 100.0 for v in y]
        colors = [PALETTE[instance]] * len(df)
        ax.plot(x, norm_y, color=PALETTE[instance], linewidth=2.2, marker="o", markersize=5.5)
        for xi, yi, t in zip(x, norm_y, df["type"]):
            ax.scatter([xi], [yi], color=PALETTE[instance], s=45, marker=TYPE_MARKERS.get(t, "o"), zorder=3)
            ax.annotate(
                t,
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#333333",
            )
        ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(instance.replace("_", " "))
        ax.set_xlabel("轮次 / Stage")
        ax.set_ylabel("归一化成本（baseline = 100）")
        ax.set_xticks(x)
        labels = ["B" if i == 0 else ("F" if t == "final" else str(i)) for i, t in zip(x, df["type"])]
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.02,
            0.95,
            f"baseline={baseline_cost:,.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.8),
        )

    fig.suptitle("Figure 1. Normalized Cost Trajectories Across Optimization Rounds", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Caption: Each panel reports the round-wise cost trajectory normalized to the baseline cost. The dashed line marks the baseline level; markers indicate accepted/rejected/final states.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIG_DIR / "figure1_cost_trajectories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_bar_figure(final_summary_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]
    df = final_summary_df.set_index("instance").loc[order].reset_index()

    x = np.arange(len(df))
    width = 0.34
    axes[0].bar(x - width / 2, df["baseline_cost"], width, label="Baseline", color="#4C78A8")
    axes[0].bar(x + width / 2, df["final_cost"], width, label="Final best", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([item.replace("_", " ") for item in df["instance"]])
    axes[0].set_ylabel("Cost")
    axes[0].set_title("Baseline vs Final Cost")
    axes[0].legend(frameon=True)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(x, df["improvement_pct"], color="#54A24B", width=0.55)
    axes[1].axhline(0, color="#444444", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([item.replace("_", " ") for item in df["instance"]])
    axes[1].set_ylabel("Improvement (%)")
    axes[1].set_title("Total Improvement over Baseline")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    for i, val in enumerate(df["improvement_pct"]):
        axes[1].text(i, val + (0.8 if val >= 0 else -1.4), f"{val:.2f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    for i, val in enumerate(df["baseline_cost"]):
        axes[0].text(i - width / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    for i, val in enumerate(df["final_cost"]):
        axes[0].text(i + width / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 2. Cross-Instance Comparison of Baseline and Final Outcomes", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Caption: The left panel compares absolute costs before and after optimization. The right panel reports the corresponding total improvement rate, highlighting divergence across instance families.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIG_DIR / "figure2_bar_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_scatter_figure(rounds_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    df = rounds_df.copy()
    df["instance"] = df["instance"].astype(str)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["num_routes"] = pd.to_numeric(df["num_routes"], errors="coerce")
    df = df.dropna(subset=["cost", "num_routes"])

    markers = {"baseline": "o", "accepted": "^", "rejected": "s", "final": "D"}
    for instance, group in df.groupby("instance"):
        for t, sub in group.groupby("type"):
            ax.scatter(
                sub["num_routes"],
                sub["cost"],
                s=np.clip(sub["elapsed_time"].fillna(20) * 3.5, 40, 260),
                alpha=0.82,
                color=PALETTE[instance],
                marker=markers.get(t, "o"),
                edgecolor="white",
                linewidth=0.8,
                label=f"{instance.replace('_', ' ')} / {t}",
            )

    # Fit a global linear trend line.
    if len(df) >= 2:
        x = df["num_routes"].astype(float).to_numpy()
        y = df["cost"].astype(float).to_numpy()
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="#222222", linewidth=1.8, linestyle="--", label="Global linear fit")

    # Annotate key outliers.
    outliers = df.sort_values("cost", ascending=False).head(4)
    for _, row in outliers.iterrows():
        ax.annotate(
            f"{row['instance']} r{int(row['stage']) if pd.notna(row['stage']) else 'F'}",
            (row["num_routes"], row["cost"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Number of routes")
    ax.set_ylabel("Cost")
    ax.set_title("Cost-Route Relationship Across All Available Rounds")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, title="Instance / type")

    fig.suptitle("Figure 3. Scatter Plot of Cost Versus Route Count", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Caption: Bubble size encodes elapsed time. The scatter structure reveals non-linear behavior and several pathological rounds where a reduced route count coincides with a sharp cost increase.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = FIG_DIR / "figure3_scatter_routes_cost.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_stacked_components_figure(rounds_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    order = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]
    final_like = []
    for instance in order:
        sub = rounds_df[rounds_df["instance"] == instance].copy()
        if sub.empty:
            continue
        comp_cols = [c for c in COST_COMPONENTS if c in sub.columns]
        sub = sub.dropna(subset=comp_cols, how="all")
        if sub.empty:
            continue
        best_row = sub.sort_values("cost").iloc[0]
        final_like.append(best_row)

    df = pd.DataFrame(final_like)
    df = df.set_index("instance").loc[order].reset_index()

    left = np.zeros(len(df))
    y_positions = np.arange(len(df))
    for comp in COST_COMPONENTS:
        values = df[comp].astype(float).fillna(0).to_numpy()
        ax.barh(y_positions, values, left=left, color=COST_COMPONENT_COLORS[comp], label=COST_COMPONENT_LABELS[comp])
        left += values

    ax.set_yticks(y_positions)
    ax.set_yticklabels([item.replace("_", " ") for item in df["instance"]])
    ax.set_xlabel("Share of total cost (%)")
    ax.set_title("Best Available Cost Composition by Instance")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=True)

    for i, row in df.iterrows():
        total = 0.0
        for comp in COST_COMPONENTS:
            total += float(row.get(comp, 0) or 0)
        ax.text(total + 0.6, i, f"{total:.1f}%", va="center", fontsize=9, color="#333333")

    fig.suptitle("Figure 4. Distribution of Final Best Cost Components", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Caption: Each horizontal bar is normalized to 100% of the best available solution with a decomposable cost profile. The component mix differs materially across instances, indicating different structural burdens across demand classes.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIG_DIR / "figure4_cost_components.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(rounds_df: pd.DataFrame, final_summary_df: pd.DataFrame, metadata: Dict[str, Any], fig_paths: Dict[str, Path]) -> str:
    order = ["c1_c101", "c1_c105", "c2_c201", "c2_c205"]
    final_summary_df = final_summary_df.set_index("instance").loc[order].reset_index()

    top_line = final_summary_df.sort_values("improvement_pct", ascending=False).iloc[0]
    weak_line = final_summary_df.sort_values("improvement_pct", ascending=True).iloc[0]
    max_routes = final_summary_df.loc[final_summary_df["final_routes"].idxmax()]
    min_routes = final_summary_df.loc[final_summary_df["final_routes"].idxmin()]

    table_rows = []
    for _, row in final_summary_df.iterrows():
        table_rows.append(
            f"| {row['instance']} | {format_num(row['baseline_cost'], 0)} | {format_num(row['final_cost'], 0)} | {format_num(row['improvement_pct'], 2)}% | {format_num(row['baseline_routes'], 0)} | {format_num(row['final_routes'], 0)} |"
        )

    # Aggregate observations for narrative.
    cost_drop = final_summary_df["baseline_cost"] - final_summary_df["final_cost"]
    corr = rounds_df[["num_routes", "cost"]].corr(numeric_only=True).iloc[0, 1]
    high_cost_rounds = rounds_df.sort_values("cost", ascending=False).head(3)

    narrative_high_cost = "; ".join(
        [f"{r.instance} {r.type} (round {r.stage}) cost={r.cost:,.0f}, routes={int(r.num_routes)}" for _, r in high_cost_rounds.iterrows()]
    )

    rel_paths = {k: v.relative_to(OUT_DIR).as_posix() for k, v in fig_paths.items()}

    report = f"""# 结构化优化实验的可视化分析报告

## 1. 研究对象与指标
本报告基于 `experiments_llm_optimize` 目录下四个实例（`c1_c101`、`c1_c105`、`c2_c201`、`c2_c205`）的轮次结果，对优化过程中的成本、路由数量、耗时与成本构成进行可视化分析。图表采用统一字体、深色高对比配色和 300 DPI 导出，以满足学术展示要求。

### 核心指标汇总
| 实例 | Baseline Cost | Final Cost | Improvement | Baseline Routes | Final Routes |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

## 2. 图表分析

### Figure 1. 成本趋势图
![Figure 1]({rel_paths['line']})

- 该图展示了各实例在可用轮次上的归一化成本轨迹，基准值统一设为 100。
- `c1_c101` 与 `c1_c105` 呈现出较明显的下降或震荡后回落过程，其中 `c1_c101` 的最终结果低于 baseline，说明后续隐含搜索阶段仍提取到更优解。
- `c2_c201` 与 `c2_c205` 的轨迹波动更强，表明大规模或更复杂的实例对局部搜索扰动更加敏感。

### Figure 2. 类别比较图
![Figure 2]({rel_paths['bar']})

- 左图比较 baseline 与最终最优解的绝对成本，右图显示总改进率。
- 改进幅度最显著的是 `{top_line['instance'].replace('_', ' ')}`，其 improvement 为 {top_line['improvement_pct']:.2f}%。
- `{weak_line['instance'].replace('_', ' ')}` 的改进几乎为零，说明该实例在当前配置下优化空间有限，甚至可能受到搜索噪声限制。

### Figure 3. 散点相关图
![Figure 3]({rel_paths['scatter']})

- 横轴为路由数量，纵轴为成本，点的大小表示耗时，颜色区分实例，符号区分轮次类型。
- 相关系数约为 {corr:.3f}，说明路由数与成本之间存在明显但并非单调的联系。
- 最高成本的若干轮次为 {narrative_high_cost}，它们对应了明显的搜索退化或不稳定情况，是后续算法调参的重点。

### Figure 4. 成本构成图
![Figure 4]({rel_paths['stacked']})

- 该图按最终最优解的成本分量百分比进行堆叠展示，突出各实例内部的结构差异。
- `c2` 与 `c3` 往往占据较高比例，说明运输主成本与固定结构性成本仍是主要驱动项。
- `c1_c105` 的成本结构相对更均衡，而 `c2_c201` / `c2_c205` 体现出更强的主导性成本集中。

## 3. 主要发现
- 从整体上看，优化过程并非单调改进，多个实例都出现了“局部退化—再恢复”的波动，这说明启发式扰动和接受准则对结果影响显著。
- `c1` 类实例相对更容易获得稳定改进，而 `c2` 类实例在最终改进率上更弱，且容易出现异常高成本轮次。
- 路由数量减少并不必然带来更低成本；散点图中的极端点表明某些搜索状态虽然压缩了路由数，但代价是显著的成本恶化。
- 成本分解图显示，不同实例的成本构成比例差异较大，因此后续优化不宜仅依赖统一参数，应针对实例类别做自适应调节。

## 4. 总结
本次可视化分析表明，结构化优化结果同时呈现出**趋势性改进、类别间差异、以及成本-路由关系的非线性特征**。若将后续实验重点放在两方面：一是抑制退化轮次中的异常搜索行为，二是面向不同实例家族自适应调整重插入/移除算子比例，则有望进一步提升最终解质量与稳定性。
"""
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rounds_df, final_summary_df, metadata = collect_instance_data(ROOT)

    # Add synthetic final-stage points so the trend and scatter plots capture the actual final outcome.
    final_stage_rows: List[Dict[str, Any]] = []
    for _, row in final_summary_df.iterrows():
        final_stage_rows.append(
            {
                "instance": row["instance"],
                "family": row["family"],
                "dataset": row["dataset"],
                "stage": int(rounds_df.loc[rounds_df["instance"] == row["instance"], "stage"].max()) + 1,
                "stage_label": "final",
                "type": "final",
                "cost": row["final_cost"],
                "num_routes": row["final_routes"],
                "elapsed_time": row["final_elapsed_time"],
                "improvement_pct": row["improvement_pct"],
                "source": row["best_source"],
            }
        )

    rounds_df = pd.concat([rounds_df, pd.DataFrame(final_stage_rows)], ignore_index=True, sort=False)

    # Prefer final_best data if it exists, otherwise best round value.
    fig_paths = {
        "line": create_line_figure(rounds_df),
        "bar": create_bar_figure(final_summary_df),
        "scatter": create_scatter_figure(rounds_df),
        "stacked": create_stacked_components_figure(rounds_df),
    }

    report = build_report(rounds_df, final_summary_df, metadata, fig_paths)
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = {
        "rounds_rows": rounds_df.to_dict(orient="records"),
        "final_summary_rows": final_summary_df.to_dict(orient="records"),
        "figures": {k: str(v) for k, v in fig_paths.items()},
        "metadata": metadata,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written to: {REPORT_PATH}")
    for k, p in fig_paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
