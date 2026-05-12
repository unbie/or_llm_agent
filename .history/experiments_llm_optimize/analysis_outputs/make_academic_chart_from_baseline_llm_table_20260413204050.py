import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_table(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "instance": r["instance"],
                    "baseline_cost": float(r["baseline_cost"]),
                    "llm_best_cost": float(r["llm_best_cost"]),
                    "delta_cost": float(r["delta_cost"]),
                    "llm_improvement_pct": float(r["llm_improvement_pct"]),
                }
            )
    if not rows:
        raise ValueError("CSV 中没有可用数据")
    return rows


def set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def plot(rows, out_png: Path, out_pdf: Path):
    rows = sorted(rows, key=lambda x: x["llm_improvement_pct"], reverse=True)

    inst = [r["instance"] for r in rows]
    baseline = np.array([r["baseline_cost"] for r in rows], dtype=float)
    llm_best = np.array([r["llm_best_cost"] for r in rows], dtype=float)
    improve = np.array([r["llm_improvement_pct"] for r in rows], dtype=float)
    better = improve > 0

    y = np.arange(len(inst))
    fig, (ax_main, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.6),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        sharey=True,
    )

    # 主图：哑铃图（绝对成本）
    ax_main.scatter(baseline, y, s=44, color="#A4ABB2", edgecolor="white", linewidth=0.8, zorder=3, label="Baseline")
    for i in range(len(y)):
        ax_main.plot(
            [baseline[i], llm_best[i]],
            [y[i], y[i]],
            color="#1E4F7A" if better[i] else "#C4C9CF",
            lw=2.3 if better[i] else 1.0,
            alpha=0.92,
            zorder=2,
        )

    ax_main.scatter(
        llm_best[better],
        y[better],
        s=66,
        color="#0B3C6D",
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="LLM best (improved)",
    )
    if np.any(~better):
        ax_main.scatter(
            llm_best[~better],
            y[~better],
            s=50,
            color="#B8BEC6",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="LLM best (non-improved)",
        )

    x_pad = max((baseline.max() - llm_best.min()) * 0.012, 90)
    for i in range(len(y)):
        ax_main.text(
            max(baseline[i], llm_best[i]) + x_pad,
            y[i],
            f"{improve[i]:+.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#0B3C6D" if better[i] else "#7F878F",
            fontweight="bold" if better[i] else "normal",
        )

    ax_main.set_yticks(y)
    ax_main.set_yticklabels(inst, fontsize=10)
    ax_main.invert_yaxis()
    ax_main.set_xlim(0, max(baseline.max(), llm_best.max()) + x_pad * 10)
    ax_main.set_xlabel("Total cost", fontsize=11)
    ax_main.set_title("Baseline vs LLM Best Cost by Instance", fontsize=12.5, fontweight="bold")
    ax_main.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax_main.grid(axis="y", visible=False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_color("#D0D4D8")
    ax_main.spines["bottom"].set_color("#D0D4D8")
    ax_main.legend(frameon=False, fontsize=9, loc="lower right")

    # 子图：改进率条形图
    ax_delta.barh(
        y,
        improve,
        color=["#0B3C6D" if b else "#B8BEC6" for b in better],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax_delta.axvline(0, color="#7A828A", lw=1.0)
    lim = max(np.max(np.abs(improve)), 0.2)
    ax_delta.set_xlim(-lim * 1.35, lim * 1.35)
    ax_delta.set_xlabel("Improvement over baseline (%)", fontsize=11)
    ax_delta.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax_delta.grid(axis="y", visible=False)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_color("#D0D4D8")
    ax_delta.spines["bottom"].set_color("#D0D4D8")

    note = (
        "Note: Darker marks/lines indicate favorable (cost-reducing) results; "
        "desaturated marks indicate non-improving outcomes."
    )
    fig.text(0.012, 0.007, note, fontsize=8.6, color="#5D646B")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    set_style()
    root = Path(__file__).resolve().parent
    csv_path = root / "instance_baseline_vs_llm_best.csv"
    rows = read_table(csv_path)

    out_png = root / "figures" / "academic_baseline_vs_llm_best.png"
    out_pdf = root / "figures" / "academic_baseline_vs_llm_best.pdf"
    plot(rows, out_png, out_pdf)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
