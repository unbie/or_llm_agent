import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "instance": r["instance"],
                    "baseline_cost": float(r["baseline_cost"]),
                    "llm_best_cost": float(r["llm_best_cost"]),
                    "improve_pct": float(r["llm_improvement_pct"]),
                    "delta_cost": float(r["delta_cost"]),
                }
            )
    return sorted(rows, key=lambda x: x["instance"])


def set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def chart_cost_lines(rows, out_path: Path):
    inst = [r["instance"] for r in rows]
    x = np.arange(len(inst))
    baseline = np.array([r["baseline_cost"] for r in rows], dtype=float)
    llm_best = np.array([r["llm_best_cost"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=300, constrained_layout=True)
    ax.plot(x, baseline, marker="o", ms=6, lw=2.0, color="#9AA3AC", label="Baseline")
    ax.plot(x, llm_best, marker="o", ms=6, lw=2.3, color="#0B3C6D", label="LLM best")

    for i in range(len(x)):
        ax.vlines(x[i], ymin=min(baseline[i], llm_best[i]), ymax=max(baseline[i], llm_best[i]), color="#D5DADF", lw=1.0, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(inst)
    ax.set_ylabel("Total cost")
    ax.set_title("Line Chart 1: Baseline vs LLM Best Cost", fontweight="bold")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def chart_improvement_lines(rows, out_path: Path):
    inst = [r["instance"] for r in rows]
    x = np.arange(len(inst))
    improve = np.array([r["improve_pct"] for r in rows], dtype=float)
    delta = np.array([r["delta_cost"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(9.8, 4.8), dpi=300, constrained_layout=True)
    ax1.plot(x, improve, marker="o", ms=6, lw=2.2, color="#0B3C6D", label="Improvement (%)")
    ax1.axhline(0, color="#7E868E", lw=1.0)
    ax1.set_ylabel("Improvement (%)", color="#0B3C6D")
    ax1.tick_params(axis="y", labelcolor="#0B3C6D")

    ax2 = ax1.twinx()
    ax2.plot(x, delta, marker="s", ms=5.5, lw=1.8, color="#8A929A", label="Delta cost")
    ax2.set_ylabel("Delta cost", color="#6F7780")
    ax2.tick_params(axis="y", labelcolor="#6F7780")

    ax1.set_xticks(x)
    ax1.set_xticklabels(inst)
    ax1.set_title("Line Chart 2: Improvement Trend and Delta Cost", fontweight="bold")
    ax1.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax1.grid(axis="x", visible=False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_style()
    root = Path(__file__).resolve().parent
    rows = load_rows(root / "instance_baseline_vs_llm_best.csv")

    out1 = root / "figures" / "line_1_baseline_vs_llm_best.png"
    out2 = root / "figures" / "line_2_improvement_and_delta.png"
    chart_cost_lines(rows, out1)
    chart_improvement_lines(rows, out2)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
