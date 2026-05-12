import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(summary_path: Path):
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("final_summary_rows", [])
    if not rows:
        raise ValueError("未在 analysis_summary.json 中找到 final_summary_rows")
    return rows


def set_academic_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def prep_arrays(rows):
    rows_sorted = sorted(rows, key=lambda x: x["improvement_pct"], reverse=True)
    inst = [r["instance"] for r in rows_sorted]
    baseline = np.array([r["baseline_cost"] for r in rows_sorted], dtype=float)
    final = np.array([r["final_cost"] for r in rows_sorted], dtype=float)
    improve = np.array([r["improvement_pct"] for r in rows_sorted], dtype=float)
    improved = improve > 0
    return inst, baseline, final, improve, improved


def chart_dumbbell(inst, baseline, final, improve, improved, out_path: Path):
    y = np.arange(len(inst))
    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=300)

    ax.scatter(baseline, y, s=40, color="#A1A8B0", edgecolor="white", linewidth=0.8, zorder=3, label="Baseline")

    for i in range(len(y)):
        ax.plot(
            [baseline[i], final[i]],
            [y[i], y[i]],
            color="#1F4E79" if improved[i] else "#C4C9CF",
            lw=2.2 if improved[i] else 1.0,
            alpha=0.9,
            zorder=2,
        )

    ax.scatter(final[improved], y[improved], s=62, color="#0B3C6D", edgecolor="white", linewidth=0.9, zorder=4, label="Relative optimal (improved)")
    if np.any(~improved):
        ax.scatter(final[~improved], y[~improved], s=48, color="#B9C0C8", edgecolor="white", linewidth=0.8, zorder=4, label="Relative optimal (non-improved)")

    x_pad = max((baseline.max() - final.min()) * 0.012, 120)
    for i in range(len(y)):
        ax.text(
            max(baseline[i], final[i]) + x_pad,
            y[i],
            f"{improve[i]:+.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#0B3C6D" if improved[i] else "#8B9299",
            fontweight="bold" if improved[i] else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(inst, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Total cost", fontsize=11)
    ax.set_title("Chart A. Baseline vs Relative Optimal Cost (Dumbbell)", fontsize=12, fontweight="bold")
    ax.set_xlim(left=0, right=max(baseline.max(), final.max()) + x_pad * 8)
    ax.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D4D8")
    ax.spines["bottom"].set_color("#D0D4D8")
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def chart_grouped_bar(inst, baseline, final, improved, out_path: Path):
    x = np.arange(len(inst))
    w = 0.36
    delta = baseline - final  # >0 表示相对最优更好

    fig, (ax, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.4),
        dpi=300,
        gridspec_kw={"height_ratios": [3.2, 1.25], "hspace": 0.12},
        sharex=True,
    )

    ax.bar(x - w / 2, baseline, width=w, color="#C8CDD3", edgecolor="white", linewidth=0.8, label="Baseline", zorder=2)

    colors = ["#0E4A7E" if flag else "#B8BEC6" for flag in improved]
    ax.bar(x + w / 2, final, width=w, color=colors, edgecolor="white", linewidth=0.8, label="Relative optimal", zorder=3)

    for i in range(len(x)):
        ymin = min(baseline[i], final[i])
        ymax = max(baseline[i], final[i])
        ax.plot([x[i] - w / 2, x[i] + w / 2], [ymin, ymin], color="#8E959C", lw=0.6, alpha=0.7)
        ax.plot([x[i] - w / 2, x[i] + w / 2], [ymax, ymax], color="#8E959C", lw=0.6, alpha=0.7)

        label_color = "#0B3C6D" if delta[i] > 0 else "#7F878F"
        label_weight = "bold" if delta[i] > 0 else "normal"
        ax.text(
            x[i],
            ymax + max(baseline.max(), final.max()) * 0.007,
            f"Δ={delta[i]:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=label_color,
            fontweight=label_weight,
        )

    ax.set_ylabel("Total cost", fontsize=11)
    ax.set_title(
        "Chart B. Baseline vs Relative Optimal Cost (Absolute + Delta Panel)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, max(baseline.max(), final.max()) * 1.16)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D4D8")
    ax.spines["bottom"].set_color("#D0D4D8")
    ax.legend(frameon=False, fontsize=9)

    # 差值子图：直接呈现每个实例的成本差异，避免在大尺度下被淹没
    delta_colors = ["#0B3C6D" if d > 0 else "#B8BEC6" for d in delta]
    ax_delta.bar(x, delta, width=0.52, color=delta_colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax_delta.axhline(0, color="#7A828A", lw=1.0, zorder=2)

    for i in range(len(x)):
        y_text = delta[i] + np.sign(delta[i]) * (max(abs(delta).max(), 1) * 0.06)
        va = "bottom" if delta[i] >= 0 else "top"
        ax_delta.text(
            x[i],
            y_text,
            f"{delta[i]:+.1f}",
            ha="center",
            va=va,
            fontsize=8.6,
            color="#0B3C6D" if delta[i] > 0 else "#7F878F",
            fontweight="bold" if delta[i] > 0 else "normal",
        )

    lim = max(abs(delta).max(), 1.0)
    ax_delta.set_ylim(-lim * 1.45, lim * 1.45)
    ax_delta.set_ylabel("Δcost", fontsize=10)
    ax_delta.set_xlabel("Instance", fontsize=11)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(inst, fontsize=10)
    ax_delta.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax_delta.grid(axis="x", visible=False)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_color("#D0D4D8")
    ax_delta.spines["bottom"].set_color("#D0D4D8")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def chart_diverging_improvement(inst, improve, improved, out_path: Path):
    y = np.arange(len(inst))
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=300)

    colors = ["#0B3C6D" if flag else "#B8BEC6" for flag in improved]
    ax.barh(y, improve, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color="#7A828A", lw=1.1, zorder=2)

    for i in range(len(y)):
        ha = "left" if improve[i] >= 0 else "right"
        x_text = improve[i] + 0.03 if improve[i] >= 0 else improve[i] - 0.03
        ax.text(
            x_text,
            y[i],
            f"{improve[i]:+.2f}%",
            va="center",
            ha=ha,
            fontsize=9,
            color="#0B3C6D" if improved[i] else "#7F878F",
            fontweight="bold" if improved[i] else "normal",
        )

    lim = max(abs(improve.min()), abs(improve.max()), 0.2)
    ax.set_xlim(-lim * 1.25, lim * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(inst, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Improvement over baseline (%)", fontsize=11)
    ax.set_title("Chart C. Relative Improvement Rate (Diverging)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D4D8")
    ax.spines["bottom"].set_color("#D0D4D8")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_academic_style()
    root = Path(__file__).resolve().parent
    summary_path = root / "analysis_summary.json"
    rows = load_rows(summary_path)
    inst, baseline, final, improve, improved = prep_arrays(rows)

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out1 = fig_dir / "chart_a_dumbbell_baseline_vs_relative_optimal.png"
    out2 = fig_dir / "chart_b_grouped_bar_baseline_vs_relative_optimal.png"
    out3 = fig_dir / "chart_c_diverging_improvement_pct.png"

    chart_dumbbell(inst, baseline, final, improve, improved, out1)
    chart_grouped_bar(inst, baseline, final, improved, out2)
    chart_diverging_improvement(inst, improve, improved, out3)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
