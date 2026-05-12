# -*- coding: utf-8 -*-
"""
可视化 c105 的 LLM 最优算子在 1000 代下的收敛过程。

输出:
- figures/c105_llm_best_convergence_1000.png
- figures/c105_llm_best_convergence_1000.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    current = Path(__file__).resolve()
    project_root = current.parents[2]
    sys.path.insert(0, str(project_root))

    from baseline_alns import ALNSVRPSolver
    from run_llm_optimize import load_solomon_data, inject_operators

    dataset_path = project_root / "data" / "1 Solomon Benchmark" / "c1" / "c105.txt"
    operator_path = project_root / "experiments_llm_optimize" / "c1_c105" / "round_4_accepted" / "code.py"

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    if not operator_path.exists():
        raise FileNotFoundError(f"LLM 最优算子代码不存在: {operator_path}")

    max_iter = 1000
    seed = 42

    data = load_solomon_data(str(dataset_path))
    operator_code = operator_path.read_text(encoding="utf-8")

    random.seed(seed)
    solver = ALNSVRPSolver(
        data=data,
        max_iter=max_iter,
        seed=seed,
        verbose=False,
    )

    ok = inject_operators(solver, operator_code)
    if not ok:
        raise RuntimeError("算子注入失败，无法执行可视化")

    best_solution, best_cost = solver.solve()

    figures_dir = current.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = figures_dir / "c105_llm_best_convergence_1000.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "current_cost", "best_cost"])
        for i, (c, b) in enumerate(zip(solver.cost_history, solver.best_cost_history), start=1):
            writer.writerow([i, f"{c:.6f}", f"{b:.6f}"])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    iters = list(range(1, len(solver.cost_history) + 1))

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300, constrained_layout=True)
    ax.plot(iters, solver.cost_history, color="#B0B0B0", linewidth=0.9, alpha=0.55, label="Current cost")
    ax.plot(iters, solver.best_cost_history, color="#0B3C6D", linewidth=1.8, label="Best cost")

    best_idx = min(range(len(solver.best_cost_history)), key=lambda i: solver.best_cost_history[i])
    bx = iters[best_idx]
    by = solver.best_cost_history[best_idx]
    ax.scatter([bx], [by], color="#B2182B", s=26, zorder=5)
    ax.annotate(
        f"Best={by:.2f}\nIter={bx}",
        xy=(bx, by),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=8,
        color="#B2182B",
        arrowprops={"arrowstyle": "->", "color": "#B2182B", "lw": 0.9},
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("C105 LLM-best Operator Convergence (1000 iterations)", fontweight="bold")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    png_path = figures_dir / "c105_llm_best_convergence_1000.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"BEST_COST={best_cost:.6f}")
    print(f"ROUTES={len(best_solution)}")
    print(f"CURVE_POINTS={len(solver.best_cost_history)}")
    print(f"PNG={png_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
