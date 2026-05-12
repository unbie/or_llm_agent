# -*- coding: utf-8 -*-
"""
一键批量运行 LLM-ALNS 强对比实验
覆盖: C101 / C105 / C201 / C205

默认采用“明显拉开 ALNS”的强配置：
- rounds=10
- iters=1500
- final-iters=3000
- eval-runs=3 (多 seed 稳健评估)

用法示例：
  python run_llm_optimize_strong_batch.py
  python run_llm_optimize_strong_batch.py --rounds 8 --iters 1200 --final-iters 2500 --eval-runs 5
  python run_llm_optimize_strong_batch.py --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


CASES = [
    ("c1", "c101"),
    ("c1", "c105"),
    ("c2", "c201"),
    ("c2", "c205"),
]


def run_one_case(
    case_type: str,
    dataset: str,
    rounds: int,
    iters: int,
    final_iters: int,
    eval_runs: int,
    seed: int,
    seed_step: int,
) -> int:
    cmd = [
        sys.executable,
        "run_llm_optimize.py",
        "--type", case_type,
        "--dataset", dataset,
        "--rounds", str(rounds),
        "--iters", str(iters),
        "--final-iters", str(final_iters),
        "--eval-runs", str(eval_runs),
        "--seed", str(seed),
        "--seed-step", str(seed_step),
    ]

    print("\n" + "=" * 78)
    print(f"[START] {case_type}/{dataset}")
    print("Command:", " ".join(cmd))
    print("=" * 78)

    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def read_summary_if_exists(case_type: str, dataset: str) -> dict | None:
    summary_path = Path("experiments_llm_optimize") / f"{case_type}_{dataset}" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="一键批量运行 C101/C105/C201/C205 强对比实验")
    parser.add_argument("--rounds", type=int, default=10, help="优化轮数，默认 10")
    parser.add_argument("--iters", type=int, default=1500, help="每轮迭代次数，默认 1500")
    parser.add_argument("--final-iters", type=int, default=3000, help="最终验证迭代，默认 3000")
    parser.add_argument("--eval-runs", type=int, default=3, help="每轮多seed评估次数，默认 3")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子，默认 42")
    parser.add_argument("--seed-step", type=int, default=97, help="多seed步长，默认 97")
    parser.add_argument("--force", action="store_true", help="即使存在 summary.json 也强制重跑")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一算例失败即停止（默认不中断，继续后续算例）",
    )
    args = parser.parse_args()

    batch_start = datetime.now().isoformat()
    print("=" * 78)
    print("LLM-ALNS 强对比批量实验启动")
    print(f"开始时间: {batch_start}")
    print(f"配置: rounds={args.rounds}, iters={args.iters}, final_iters={args.final_iters}, eval_runs={args.eval_runs}")
    print("算例: c1/c101, c1/c105, c2/c201, c2/c205")
    print("=" * 78)

    batch_results: list[dict] = []

    for idx, (case_type, dataset) in enumerate(CASES, start=1):
        print(f"\n[{idx}/{len(CASES)}] 准备运行 {case_type}/{dataset}")

        if not args.force:
            existed = read_summary_if_exists(case_type, dataset)
            if existed is not None:
                print(f"[SKIP] 已存在结果，跳过: experiments_llm_optimize/{case_type}_{dataset}/summary.json")
                batch_results.append(
                    {
                        "case": f"{case_type}/{dataset}",
                        "status": "skipped",
                        "reason": "summary_exists",
                        "best_cost": existed.get("best_cost"),
                        "total_improvement_pct": existed.get("total_improvement_pct"),
                    }
                )
                continue

        code = run_one_case(
            case_type=case_type,
            dataset=dataset,
            rounds=args.rounds,
            iters=args.iters,
            final_iters=args.final_iters,
            eval_runs=args.eval_runs,
            seed=args.seed,
            seed_step=args.seed_step,
        )

        summary = read_summary_if_exists(case_type, dataset)
        if code == 0 and summary is not None:
            batch_results.append(
                {
                    "case": f"{case_type}/{dataset}",
                    "status": "success",
                    "best_cost": summary.get("best_cost"),
                    "total_improvement_pct": summary.get("total_improvement_pct"),
                    "eval_runs": summary.get("eval_runs"),
                }
            )
            print(f"[OK] {case_type}/{dataset} 完成")
        else:
            batch_results.append(
                {
                    "case": f"{case_type}/{dataset}",
                    "status": "failed",
                    "return_code": code,
                }
            )
            print(f"[FAIL] {case_type}/{dataset} 失败，return_code={code}")
            if args.stop_on_error:
                break

    batch_end = datetime.now().isoformat()
    success_count = sum(1 for r in batch_results if r.get("status") == "success")
    fail_count = sum(1 for r in batch_results if r.get("status") == "failed")
    skip_count = sum(1 for r in batch_results if r.get("status") == "skipped")

    report = {
        "batch_name": "llm_optimize_strong_batch",
        "started_at": batch_start,
        "finished_at": batch_end,
        "config": {
            "rounds": args.rounds,
            "iters": args.iters,
            "final_iters": args.final_iters,
            "eval_runs": args.eval_runs,
            "seed": args.seed,
            "seed_step": args.seed_step,
            "force": args.force,
            "stop_on_error": args.stop_on_error,
        },
        "summary": {
            "success": success_count,
            "failed": fail_count,
            "skipped": skip_count,
            "total": len(batch_results),
        },
        "results": batch_results,
    }

    out_dir = Path("experiments_llm_optimize")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strong_batch_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("批量任务完成")
    print(f"成功: {success_count} | 失败: {fail_count} | 跳过: {skip_count} | 总计: {len(batch_results)}")
    print(f"报告文件: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
