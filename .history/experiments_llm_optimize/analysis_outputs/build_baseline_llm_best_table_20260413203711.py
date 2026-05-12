import csv
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_baseline_cost(instance_dir: Path):
    p = instance_dir / "round_0_baseline" / "result.json"
    if not p.exists():
        return None, None
    d = read_json(p)
    if "best_cost" in d:
        return float(d["best_cost"]), "round_0_baseline/result.json:best_cost"
    if "cost" in d:
        return float(d["cost"]), "round_0_baseline/result.json:cost"
    return None, None


def collect_llm_candidates(instance_dir: Path):
    cands = []

    # round_1+ 视为 LLM 运行轮次
    for p in sorted(instance_dir.glob("round_*/*result.json")):
        round_name = p.parent.name
        if round_name == "round_0_baseline":
            continue
        d = read_json(p)
        if "cost" in d:
            cands.append((float(d["cost"]), f"{round_name}/result.json:cost"))
        elif "best_cost" in d:
            cands.append((float(d["best_cost"]), f"{round_name}/result.json:best_cost"))

    # final_best 若存在，也视为 LLM 链路最终结果
    fp = instance_dir / "final_best" / "result.json"
    if fp.exists():
        d = read_json(fp)
        if "final_cost" in d:
            cands.append((float(d["final_cost"]), "final_best/result.json:final_cost"))

    return cands


def build_rows(exp_root: Path):
    rows = []
    for instance_dir in sorted(exp_root.glob("*_*")):
        if not instance_dir.is_dir():
            continue

        baseline_cost, baseline_source = get_baseline_cost(instance_dir)
        if baseline_cost is None:
            continue

        llm_cands = collect_llm_candidates(instance_dir)
        if not llm_cands:
            llm_best_cost = None
            llm_best_source = ""
            llm_improve_pct = None
        else:
            llm_best_cost, llm_best_source = min(llm_cands, key=lambda x: x[0])
            llm_improve_pct = (baseline_cost - llm_best_cost) / baseline_cost * 100.0

        rows.append(
            {
                "instance": instance_dir.name,
                "baseline_cost": baseline_cost,
                "llm_best_cost": llm_best_cost,
                "delta_cost": None if llm_best_cost is None else baseline_cost - llm_best_cost,
                "llm_improvement_pct": llm_improve_pct,
                "baseline_source": baseline_source,
                "llm_best_source": llm_best_source,
            }
        )

    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "baseline_cost",
                "llm_best_cost",
                "delta_cost",
                "llm_improvement_pct",
                "baseline_source",
                "llm_best_source",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "instance": r["instance"],
                    "baseline_cost": f"{r['baseline_cost']:.6f}",
                    "llm_best_cost": "" if r["llm_best_cost"] is None else f"{r['llm_best_cost']:.6f}",
                    "delta_cost": "" if r["delta_cost"] is None else f"{r['delta_cost']:.6f}",
                    "llm_improvement_pct": "" if r["llm_improvement_pct"] is None else f"{r['llm_improvement_pct']:.6f}",
                    "baseline_source": r["baseline_source"],
                    "llm_best_source": r["llm_best_source"],
                }
            )


def main():
    exp_root = Path(__file__).resolve().parents[1]
    out_csv = Path(__file__).resolve().parent / "instance_baseline_vs_llm_best.csv"

    rows = build_rows(exp_root)
    write_csv(rows, out_csv)

    print(f"Saved: {out_csv}")
    for r in rows:
        if r["llm_best_cost"] is None:
            print(f"{r['instance']}: baseline={r['baseline_cost']:.3f}, llm_best=NA")
        else:
            print(
                f"{r['instance']}: baseline={r['baseline_cost']:.3f}, "
                f"llm_best={r['llm_best_cost']:.3f}, "
                f"improve={r['llm_improvement_pct']:.3f}%"
            )


if __name__ == "__main__":
    main()
