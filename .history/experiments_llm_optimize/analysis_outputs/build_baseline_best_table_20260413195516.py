import csv
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_baseline_cost(instance_dir: Path):
    baseline_path = instance_dir / "round_0_baseline" / "result.json"
    if baseline_path.exists():
        data = read_json(baseline_path)
        if "best_cost" in data:
            return float(data["best_cost"]), str(baseline_path.name)
        if "cost" in data:
            return float(data["cost"]), str(baseline_path.name)

    summary_path = instance_dir / "summary.json"
    if summary_path.exists():
        data = read_json(summary_path)
        if "baseline_cost" in data:
            return float(data["baseline_cost"]), "summary.json"
    return None, None


def candidate_costs(instance_dir: Path):
    cands = []

    # final_best
    final_path = instance_dir / "final_best" / "result.json"
    if final_path.exists():
        d = read_json(final_path)
        if "final_cost" in d:
            cands.append((float(d["final_cost"]), "final_best/result.json:final_cost"))

    # 各轮 result.json
    for p in sorted(instance_dir.glob("round_*/*result.json")):
        d = read_json(p)
        if "cost" in d:
            cands.append((float(d["cost"]), f"{p.parent.name}/result.json:cost"))
        elif "best_cost" in d:
            cands.append((float(d["best_cost"]), f"{p.parent.name}/result.json:best_cost"))

    # summary.json best_cost 作为兜底
    summary_path = instance_dir / "summary.json"
    if summary_path.exists():
        d = read_json(summary_path)
        if "best_cost" in d:
            cands.append((float(d["best_cost"]), "summary.json:best_cost"))

    return cands


def build_table(root: Path):
    rows = []
    for instance_dir in sorted(root.glob("*_*")):
        if not instance_dir.is_dir():
            continue
        baseline_cost, baseline_source = extract_baseline_cost(instance_dir)
        if baseline_cost is None:
            continue

        cands = candidate_costs(instance_dir)
        if not cands:
            continue

        best_cost, best_source = min(cands, key=lambda x: x[0])
        improvement_pct = (baseline_cost - best_cost) / baseline_cost * 100.0 if baseline_cost else 0.0

        rows.append(
            {
                "instance": instance_dir.name,
                "baseline_cost": baseline_cost,
                "best_cost": best_cost,
                "improvement_pct": improvement_pct,
                "baseline_source": baseline_source,
                "best_source": best_source,
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
                "best_cost",
                "improvement_pct",
                "baseline_source",
                "best_source",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "instance": r["instance"],
                    "baseline_cost": f"{r['baseline_cost']:.6f}",
                    "best_cost": f"{r['best_cost']:.6f}",
                    "improvement_pct": f"{r['improvement_pct']:.6f}",
                    "baseline_source": r["baseline_source"],
                    "best_source": r["best_source"],
                }
            )


def main():
    exp_root = Path(__file__).resolve().parents[1]
    rows = build_table(exp_root)
    out_path = Path(__file__).resolve().parent / "instance_baseline_best_table.csv"
    write_csv(rows, out_path)
    print(f"Saved: {out_path}")
    for r in rows:
        print(
            f"{r['instance']}: baseline={r['baseline_cost']:.3f}, "
            f"best={r['best_cost']:.3f}, improve={r['improvement_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
