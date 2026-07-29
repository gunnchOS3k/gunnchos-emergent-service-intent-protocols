"""Generate paper tables from smoke results if present."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = ROOT / "results" / "smoke" / "smoke_summary_table.json"
    out = ROOT / "paper" / "tables" / "smoke_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not summary.exists():
        out.write_text("# Smoke results\n\n_No results yet — run `make smoke`._\n")
        print("NO_RESULTS")
        return
    data = json.loads(summary.read_text())
    lines = [
        "# Smoke results (SYNTHETIC_SIM)",
        "",
        f"Device label: `{data.get('device', {}).get('label')}`",
        "",
        "| config | seed | mean_return | eval_return | status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in data.get("rows", []):
        lines.append(
            f"| {row['config']} | {row['seed']} | {row['mean_return']:.4f} | "
            f"{row['eval_return']:.4f} | {row['status']} |"
        )
    out.write_text("\n".join(lines) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
