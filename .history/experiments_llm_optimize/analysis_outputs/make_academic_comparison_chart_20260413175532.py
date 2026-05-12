import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_final_rows(summary_path: Path):
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("final_summary_rows", [])
    if not rows:
        raise ValueError("analysis_summary.json 中未找到 final_summary_rows")
    return rows


def build_chart(rows, out_png: Path, out_pdf: Path):
    # 按改进率排序，保证积极结果优先展示
    rows_sorted = sorted(rows, key=lambda x: x["improvement_pct"], reverse=True)

    instances = [r["instance"] for r in rows_sorted]
    baseline = np.array([r["baseline_cost"] for r in rows_sorted], dtype=float)
    final = np.array([r["final_cost"] for r in rows_sorted], dtype=float)
    improvement_pct = np.array([r["improvement_pct"] for r in rows_sorted], dtype=float)

    y = np.arange(len(instances))
    positive = improvement_pct > 0

    # 字体策略：优先 Times New Roman，回退 Arial
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 基准点：统一弱化
    ax.scatter(
        baseline,
        y,
        s=42,
        color="#9AA0A6",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Baseline cost",
    )

    # 连接线：积极改进加粗、劣势结果弱化
    for i in range(len(y)):
        if positive[i]:
            line_color = "#1F4E79"
            lw = 2.2
            alpha = 0.9
        else:
            line_color = "#C3C7CC"
            lw = 1.0
            alpha = 0.8
        ax.plot([baseline[i], final[i]], [y[i], y[i]], color=line_color, lw=lw, alpha=alpha, zorder=2)

    # 最优/最终点：优势结果高对比，劣势结果低饱和
    ax.scatter(
        final[positive],
        y[positive],
        s=64,
        color="#0B3C6D",
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="Relative optimal cost (improved)",
    )
    if np.any(~positive):
        ax.scatter(
            final[~positive],
            y[~positive],
            s=48,
            color="#B8BEC6",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="Relative optimal cost (non-improved)",
        )

    # 改进标注：强化积极结果，弱化负向结果
    x_span = baseline.max() - final.min()
    x_pad = max(x_span * 0.01, 120)
    for i in range(len(y)):
        text = f"{improvement_pct[i]:+.2f}%"
        if positive[i]:
            color = "#0B3C6D"
            weight = "bold"
            alpha = 0.95
        else:
            color = "#8B9198"
            weight = "normal"
            alpha = 0.9
        ax.text(
            max(baseline[i], final[i]) + x_pad,
            y[i],
            text,
            va="center",
            ha="left",
            fontsize=9.2,
            color=color,
            fontweight=weight,
            alpha=alpha,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(instances, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Total cost", fontsize=11)
    ax.set_title(
        "Baseline vs Relative Optimal Cost (Academic Emphasis)",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )

    # 诚实尺度：不截断到局部窄区间，保留0起点并仅做适度留白
    right_margin = x_pad * 8
    ax.set_xlim(left=0, right=max(baseline.max(), final.max()) + right_margin)

    ax.grid(axis="x", color="#E3E6EA", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#D0D4D8")
    ax.spines["bottom"].set_color("#D0D4D8")

    legend = ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
        handlelength=1.8,
    )
    for lh in legend.legendHandles:
        try:
            lh.set_alpha(1)
        except Exception:
            pass

    note = (
        "Note: Positive improvements are emphasized via darker color and thicker connectors; "
        "non-improved points are intentionally desaturated without changing numerical values."
    )
    fig.text(0.012, 0.01, note, fontsize=8.5, color="#5E646B")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    summary_path = root / "analysis_summary.json"
    rows = load_final_rows(summary_path)

    out_png = root / "figures" / "figure5_baseline_vs_relative_optimal_academic.png"
    out_pdf = root / "figures" / "figure5_baseline_vs_relative_optimal_academic.pdf"

    build_chart(rows, out_png, out_pdf)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
