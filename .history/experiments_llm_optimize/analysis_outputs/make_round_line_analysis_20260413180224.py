import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Arial", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_round_result(result_path: Path):
    folder = result_path.parent.name
    m = re.match(r"round_(\d+)_([a-zA-Z]+)", folder)
    if not m:
        return None

    round_idx = int(m.group(1))
    folder_type = m.group(2).lower()
    data = read_json(result_path)

    cost = data.get("cost")
    if cost is None:
        cost = data.get("best_cost")
    if cost is None and isinstance(data.get("metrics"), dict):
        cost = data["metrics"].get("best_cost")
    if cost is None:
        return None

    round_type = str(data.get("type", folder_type)).lower()
    return {
        "x": round_idx,
        "round": round_idx,
        "type": round_type,
        "cost": float(cost),
        "source": folder,
    }


def parse_final_result(result_path: Path, next_x: int):
    data = read_json(result_path)
    final_cost = data.get("final_cost", data.get("cost"))
    baseline_cost = data.get("baseline_cost")
    if final_cost is None:
        return None, baseline_cost
    return {
        "x": next_x,
        "round": "final",
        "type": "final",
        "cost": float(final_cost),
        "source": "final_best",
    }, baseline_cost


def load_instance_series(instance_dir: Path):
    points = []
    for rp in sorted(instance_dir.glob("round_*_*/result.json")):
        pt = parse_round_result(rp)
        if pt is not None:
            points.append(pt)

    if not points:
        return None

    points = sorted(points, key=lambda d: d["x"])
    baseline_candidates = [p["cost"] for p in points if p["type"] == "baseline" or p["x"] == 0]
    baseline_cost = baseline_candidates[0] if baseline_candidates else points[0]["cost"]

    final_path = instance_dir / "final_best" / "result.json"
    if final_path.exists():
        final_point, baseline_from_final = parse_final_result(final_path, next_x=max(p["x"] for p in points) + 1)
        if baseline_from_final is not None:
            baseline_cost = float(baseline_from_final)
        if final_point is not None:
            points.append(final_point)

    # 保障 x 有序
    points = sorted(points, key=lambda d: d["x"])

    return {
        "instance": instance_dir.name,
        "baseline_cost": float(baseline_cost),
        "points": points,
    }


def draw_instance_chart(series, out_path: Path):
    instance = series["instance"]
    baseline = series["baseline_cost"]
    pts = series["points"]

    x = np.array([p["x"] for p in pts], dtype=float)
    y = np.array([p["cost"] for p in pts], dtype=float)
    labels = [f"r{p['round']}" if isinstance(p["round"], int) else "final" for p in pts]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.2),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.35], "hspace": 0.10},
        sharex=True,
    )

    # 主图：baseline 常量线 + 其他轮次轨迹线
    ax1.hlines(
        baseline,
        xmin=x.min(),
        xmax=x.max(),
        colors="#9EA5AD",
        linestyles="--",
        linewidth=1.6,
        label="Baseline line",
        zorder=1,
    )
    ax1.plot(
        x,
        y,
        color="#0B3C6D",
        linewidth=2.0,
        marker="o",
        markersize=5.5,
        label="Round cost line",
        zorder=3,
    )

    for p in pts:
        xi = p["x"]
        yi = p["cost"]
        ptype = p["type"]
        if ptype == "accepted" or ptype == "final":
            color = "#0B3C6D"
            marker = "o"
            size = 32
        elif ptype == "rejected":
            color = "#B8BEC6"
            marker = "x"
            size = 40
        else:
            color = "#8A929A"
            marker = "s"
            size = 24

        ax1.scatter([xi], [yi], c=color, marker=marker, s=size, linewidths=1.1, zorder=4)

    # 右侧标注改进率
    improve_final = (baseline - y[-1]) / baseline * 100.0
    ax1.text(
        x.max() + 0.08,
        y[-1],
        f"final Δ={improve_final:+.2f}%",
        va="center",
        ha="left",
        fontsize=9,
        color="#0B3C6D" if improve_final > 0 else "#7F878F",
        fontweight="bold" if improve_final > 0 else "normal",
    )

    ax1.set_ylabel("Total cost", fontsize=11)
    ax1.set_title(f"{instance}: Baseline Line vs Round Cost Line", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax1.grid(axis="x", visible=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#D0D4D8")
    ax1.spines["bottom"].set_color("#D0D4D8")
    ax1.legend(frameon=False, fontsize=9, loc="best")

    # 子图：相对 baseline 的偏差百分比
    pct_delta = (y - baseline) / baseline * 100.0
    bar_colors = ["#0B3C6D" if v <= 0 else "#B8BEC6" for v in pct_delta]
    ax2.bar(x, pct_delta, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="#7A828A", linewidth=1.0)

    for i, v in enumerate(pct_delta):
        va = "bottom" if v >= 0 else "top"
        y_text = v + (0.6 if v >= 0 else -0.6)
        ax2.text(x[i], y_text, f"{v:+.2f}%", ha="center", va=va, fontsize=8.2, color="#66707A")

    lim = max(2.5, np.max(np.abs(pct_delta)) * 1.25)
    ax2.set_ylim(-lim, lim)
    ax2.set_ylabel("Δ% vs base", fontsize=10)
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax2.grid(axis="x", visible=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#D0D4D8")
    ax2.spines["bottom"].set_color("#D0D4D8")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_overview_chart(series_list, out_path: Path):
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    ax.axhline(1.0, color="#9EA5AD", linestyle="--", linewidth=1.2, label="Baseline ratio = 1.0")
    for s in series_list:
        instance = s["instance"]
        base = s["baseline_cost"]
        pts = s["points"]
        x = np.array([p["x"] for p in pts], dtype=float)
        y = np.array([p["cost"] for p in pts], dtype=float) / base
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.5, label=instance)

    ax.set_title("All Instances: Cost Trajectory Ratio to Baseline", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cost / Baseline", fontsize=11)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", color="#F0F2F4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D4D8")
    ax.spines["bottom"].set_color("#D0D4D8")
    ax.legend(frameon=False, fontsize=8.8, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_style()

    exp_root = Path(__file__).resolve().parent.parent
    out_root = Path(__file__).resolve().parent / "figures" / "round_line_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    instance_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir() and re.match(r"c\d+_c\d+", d.name)])
    all_series = []

    for d in instance_dirs:
        series = load_instance_series(d)
        if series is None:
            continue
        all_series.append(series)
        out_file = out_root / f"{d.name}_baseline_vs_round_lines.png"
        draw_instance_chart(series, out_file)
        print(f"Saved: {out_file}")

    if all_series:
        overview_file = out_root / "all_instances_ratio_overview.png"
        draw_overview_chart(all_series, overview_file)
        print(f"Saved: {overview_file}")


if __name__ == "__main__":
    main()
