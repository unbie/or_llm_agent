import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(summary_path: Path):
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rounds_rows = data.get("rounds_rows", [])
    final_rows = data.get("final_summary_rows", [])
    if not rounds_rows:
        raise ValueError("analysis_summary.json 中缺少 rounds_rows")
    if not final_rows:
        raise ValueError("analysis_summary.json 中缺少 final_summary_rows")
    return rounds_rows, final_rows


def set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def chart_1_final_cost_compare(final_rows, out_path: Path):
    rows = sorted(final_rows, key=lambda x: x["improvement_pct"], reverse=True)
    inst = [r["instance"] for r in rows]
    baseline = np.array([r["baseline_cost"] for r in rows], dtype=float)
    final = np.array([r["final_cost"] for r in rows], dtype=float)
    improve = np.array([r["improvement_pct"] for r in rows], dtype=float)
    improved = improve > 0

    x = np.arange(len(inst))
    w = 0.34
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        sharex=True,
    )

    ax1.bar(x - w / 2, baseline, width=w, color="#C7CCD2", edgecolor="white", linewidth=0.8, label="Baseline", zorder=2)
    ax1.bar(
        x + w / 2,
        final,
        width=w,
        color=["#0E4A7E" if flag else "#B8BEC6" for flag in improved],
        edgecolor="white",
        linewidth=0.8,
        label="Relative optimal",
        zorder=3,
    )
    for i in range(len(x)):
        delta = baseline[i] - final[i]
        ymax = max(baseline[i], final[i])
        ax1.text(
            x[i],
            ymax * 1.01,
            f"Δ={delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#0B3C6D" if delta > 0 else "#7D858D",
            fontweight="bold" if delta > 0 else "normal",
        )

    ax1.set_ylabel("Total cost")
    ax1.set_title("All-data View 1: Baseline vs Relative Optimal Cost", fontweight="bold")
    ax1.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax1.grid(axis="x", visible=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False)

    delta = baseline - final
    ax2.bar(x, delta, width=0.52, color=["#0B3C6D" if d > 0 else "#B8BEC6" for d in delta], edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="#7A828A", lw=1.0)
    lim = max(np.max(np.abs(delta)), 1.0)
    ax2.set_ylim(-lim * 1.35, lim * 1.35)
    ax2.set_ylabel("Δcost")
    ax2.set_xticks(x)
    ax2.set_xticklabels(inst)
    ax2.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax2.grid(axis="x", visible=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    save(fig, out_path)


def chart_2_cost_trajectory(rounds_rows, out_path: Path):
    instance_names = sorted({r["instance"] for r in rounds_rows})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    axes = axes.flatten()

    type_style = {
        "baseline": ("#6D7580", "o", 1.0),
        "accepted": ("#0B3C6D", "o", 1.0),
        "rejected": ("#B8BEC6", "o", 0.85),
        "final": ("#1A5A96", "D", 1.0),
    }

    for ax, inst in zip(axes, instance_names):
        rows = sorted([r for r in rounds_rows if r["instance"] == inst], key=lambda x: x.get("stage", 0))
        stages = np.array([r.get("stage", 0) for r in rows], dtype=float)
        costs = np.array([r.get("cost", np.nan) for r in rows], dtype=float)
        types = [r.get("type", "") for r in rows]

        ax.plot(stages, costs, color="#CED3D8", lw=1.3, zorder=1)
        for s, c, t in zip(stages, costs, types):
            color, marker, alpha = type_style.get(t, ("#7A828A", "o", 0.9))
            ax.scatter(s, c, s=38 if t != "final" else 52, color=color, marker=marker, edgecolor="white", linewidth=0.7, alpha=alpha, zorder=2)

        ax.set_title(inst, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Cost")
        ax.grid(color="#E4E7EB", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(instance_names):]:
        ax.axis("off")

    fig.suptitle("All-data View 2: Cost Trajectories by Instance", fontsize=13, fontweight="bold")
    save(fig, out_path)


def chart_3_normalized_vs_baseline(rounds_rows, out_path: Path):
    instance_names = sorted({r["instance"] for r in rounds_rows})
    fig, ax = plt.subplots(figsize=(10.6, 4.8), constrained_layout=True)

    palette = ["#0B3C6D", "#2D5F8A", "#5B8BB2", "#90AFC4"]

    for i, inst in enumerate(instance_names):
        rows = sorted([r for r in rounds_rows if r["instance"] == inst], key=lambda x: x.get("stage", 0))
        baseline_rows = [r for r in rows if r.get("type") == "baseline"]
        if not baseline_rows:
            continue
        base_cost = baseline_rows[0].get("cost", np.nan)
        if not np.isfinite(base_cost) or base_cost == 0:
            continue

        stages = np.array([r.get("stage", 0) for r in rows], dtype=float)
        norm = np.array([100.0 * r.get("cost", np.nan) / base_cost for r in rows], dtype=float)
        ax.plot(stages, norm, marker="o", ms=4.5, lw=1.9, color=palette[i % len(palette)], alpha=0.95, label=inst)

    ax.axhline(100, color="#8D949B", linestyle="--", lw=1.0)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cost index (baseline = 100)")
    ax.set_title("All-data View 3: Normalized Cost Evolution", fontweight="bold")
    ax.grid(color="#E4E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, fontsize=9)
    save(fig, out_path)


def chart_4_scatter_routes_cost_time(rounds_rows, out_path: Path):
    costs = np.array([r.get("cost", np.nan) for r in rounds_rows], dtype=float)
    routes = np.array([r.get("num_routes", np.nan) for r in rounds_rows], dtype=float)
    elapsed = np.array([r.get("elapsed_time", np.nan) for r in rounds_rows], dtype=float)
    types = [r.get("type", "") for r in rounds_rows]

    type_color = {
        "baseline": "#6D7580",
        "accepted": "#0B3C6D",
        "rejected": "#B8BEC6",
        "final": "#1A5A96",
    }

    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    for t in ["baseline", "accepted", "rejected", "final"]:
        idx = np.array([tt == t for tt in types])
        if not np.any(idx):
            continue
        sizes = 30 + 120 * (elapsed[idx] - np.nanmin(elapsed)) / (np.nanmax(elapsed) - np.nanmin(elapsed) + 1e-9)
        ax.scatter(
            routes[idx],
            costs[idx],
            s=sizes,
            color=type_color.get(t, "#7A828A"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9 if t != "rejected" else 0.72,
            label=t,
        )

    corr = np.corrcoef(routes[np.isfinite(routes) & np.isfinite(costs)], costs[np.isfinite(routes) & np.isfinite(costs)])[0, 1]
    ax.text(0.015, 0.96, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=10, color="#4F565D")

    ax.set_xlabel("Number of routes")
    ax.set_ylabel("Cost")
    ax.set_title("All-data View 4: Routes vs Cost (Bubble size = elapsed time)", fontweight="bold")
    ax.grid(color="#E4E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save(fig, out_path)


def chart_5_type_distribution(rounds_rows, out_path: Path):
    type_order = ["baseline", "accepted", "rejected", "final"]
    counts = {t: 0 for t in type_order}
    for r in rounds_rows:
        t = r.get("type", "")
        if t in counts:
            counts[t] += 1

    x = np.arange(len(type_order))
    y = np.array([counts[t] for t in type_order], dtype=float)
    colors = ["#99A1AA", "#0B3C6D", "#C0C6CD", "#1A5A96"]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    bars = ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.8)
    for b, val in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{int(val)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(type_order)
    ax.set_ylabel("Count")
    ax.set_title("All-data View 5: Stage Type Distribution", fontweight="bold")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, out_path)


def chart_6_improvement_breakdown(final_rows, out_path: Path):
    rows = sorted(final_rows, key=lambda x: x["instance"])
    inst = [r["instance"] for r in rows]
    imp = np.array([r["improvement_pct"] for r in rows], dtype=float)
    x = np.arange(len(inst))

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    colors = ["#0B3C6D" if v > 0 else "#B8BEC6" for v in imp]
    ax.barh(x, imp, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="#7A828A", lw=1.0)

    for i, v in enumerate(imp):
        x_text = v + 0.03 if v >= 0 else v - 0.03
        ax.text(
            x_text,
            i,
            f"{v:+.3f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color="#0B3C6D" if v > 0 else "#7D858D",
            fontweight="bold" if v > 0 else "normal",
        )

    lim = max(np.max(np.abs(imp)), 0.2)
    ax.set_xlim(-lim * 1.5, lim * 1.5)
    ax.set_yticks(x)
    ax.set_yticklabels(inst)
    ax.invert_yaxis()
    ax.set_xlabel("Improvement over baseline (%)")
    ax.set_title("All-data View 6: Final Improvement by Instance", fontweight="bold")
    ax.grid(axis="x", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, out_path)


def chart_7_cost_component_mix(rounds_rows, out_path: Path):
    fields = ["c11_pct", "c12_pct", "c13_pct", "c2_pct", "c3_pct"]
    labels = ["c11", "c12", "c13", "c2", "c3"]
    instance_names = sorted({r["instance"] for r in rounds_rows})

    mix = []
    for inst in instance_names:
        rows = [r for r in rounds_rows if r["instance"] == inst and r.get("type") in ("baseline", "accepted", "final")]
        vals = []
        for f in fields:
            arr = np.array([r.get(f, np.nan) for r in rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            vals.append(float(np.nanmean(arr)) if arr.size > 0 else np.nan)
        mix.append(vals)

    mix = np.array(mix, dtype=float)
    x = np.arange(len(instance_names))
    bottoms = np.zeros(len(instance_names), dtype=float)
    colors = ["#A7B7C7", "#8EA6BE", "#7393B3", "#4E79A7", "#2E5F8A"]

    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)
    for i in range(len(fields)):
        vals = np.nan_to_num(mix[:, i], nan=0.0)
        ax.bar(x, vals, bottom=bottoms, color=colors[i], edgecolor="white", linewidth=0.6, label=labels[i])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(instance_names)
    ax.set_ylabel("Average cost component share (%)")
    ax.set_title("All-data View 7: Cost Component Composition", fontweight="bold")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=5, fontsize=8.8)
    save(fig, out_path)


def main():
    set_style()
    root = Path(__file__).resolve().parent
    fig_dir = root / "figures"
    rounds_rows, final_rows = load_data(root / "analysis_summary.json")

    chart_1_final_cost_compare(final_rows, fig_dir / "all_view_1_final_cost_compare.png")
    chart_2_cost_trajectory(rounds_rows, fig_dir / "all_view_2_cost_trajectories.png")
    chart_3_normalized_vs_baseline(rounds_rows, fig_dir / "all_view_3_normalized_cost.png")
    chart_4_scatter_routes_cost_time(rounds_rows, fig_dir / "all_view_4_routes_cost_scatter.png")
    chart_5_type_distribution(rounds_rows, fig_dir / "all_view_5_type_distribution.png")
    chart_6_improvement_breakdown(final_rows, fig_dir / "all_view_6_improvement_breakdown.png")
    chart_7_cost_component_mix(rounds_rows, fig_dir / "all_view_7_cost_component_mix.png")

    print("Saved all visualization figures:")
    for p in sorted(fig_dir.glob("all_view_*.png")):
        print(p)


if __name__ == "__main__":
    main()
