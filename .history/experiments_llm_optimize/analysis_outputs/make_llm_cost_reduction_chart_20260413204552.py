import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "instance": r["instance"],
                    "baseline": float(r["baseline_cost"]),
                    "llm": float(r["llm_best_cost"]),
                    "delta": float(r["delta_cost"]),
                    "improve": float(r["llm_improvement_pct"]),
                }
            )
    return sorted(rows, key=lambda x: x["improve"], reverse=True)


def set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def plot(rows, out_png: Path, out_pdf: Path):
    inst = [r["instance"] for r in rows]
    delta = np.array([r["delta"] for r in rows], dtype=float)
    improve = np.array([r["improve"] for r in rows], dtype=float)
    better = delta > 0
    y = np.arange(len(inst))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.2),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.7, 1.6]},
        sharey=True,
    )

    # 上图：绝对降本金额
    ax1.barh(
        y,
        delta,
        color=["#0B3C6D" if b else "#B8BEC6" for b in better],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax1.axvline(0, color="#7A828A", lw=1.0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(inst, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Cost reduction amount (Baseline - LLM)")
    ax1.set_title("LLM vs Baseline: Cost Reduction by Instance", fontsize=12.5, fontweight="bold")
    lim_delta = max(np.max(np.abs(delta)), 1.0)
    pad_delta = lim_delta * 0.12
    ax1.set_xlim(np.min(delta) - pad_delta * 1.8, np.max(delta) + pad_delta * 1.2)
    ax1.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax1.grid(axis="y", visible=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i in range(len(y)):
        x_text = delta[i] + (max(abs(delta).max(), 1) * 0.03 if delta[i] >= 0 else -max(abs(delta).max(), 1) * 0.03)
        ax1.text(
            x_text,
            y[i],
            f"{delta[i]:+.1f}",
            va="center",
            ha="left" if delta[i] >= 0 else "right",
            fontsize=9,
            color="#0B3C6D" if delta[i] > 0 else "#7E868E",
            fontweight="bold" if delta[i] > 0 else "normal",
            clip_on=False,
        )

    # 下图：相对降本率
    ax2.barh(
        y,
        improve,
        color=["#174E80" if b else "#C1C7CE" for b in better],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax2.axvline(0, color="#7A828A", lw=1.0)
    lim = max(np.max(np.abs(improve)), 0.2)
    ax2.set_xlim(-lim * 1.5, lim * 1.35)
    ax2.set_xlabel("Cost reduction rate (%)")
    ax2.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax2.grid(axis="y", visible=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i in range(len(y)):
        x_text = improve[i] + (lim * 0.04 if improve[i] >= 0 else -lim * 0.04)
        ax2.text(
            x_text,
            y[i],
            f"{improve[i]:+.3f}%",
            va="center",
            ha="left" if improve[i] >= 0 else "right",
            fontsize=9,
            color="#0B3C6D" if improve[i] > 0 else "#7E868E",
            fontweight="bold" if improve[i] > 0 else "normal",
            clip_on=False,
        )

    note = "Dark bars indicate cost reduction by LLM; gray bars indicate non-reduction cases."
    fig.text(0.012, 0.008, note, fontsize=8.5, color="#5E656D")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    set_style()
    root = Path(__file__).resolve().parent
    rows = load_data(root / "instance_baseline_vs_llm_best.csv")

    out_png = root / "figures" / "llm_vs_baseline_cost_reduction.png"
    out_pdf = root / "figures" / "llm_vs_baseline_cost_reduction.pdf"
    plot(rows, out_png, out_pdf)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
