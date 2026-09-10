import json
import math
import os
import statistics
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("MICRODUCK_ARTIFACTS", repo / "artifacts"))
if not root.is_absolute():
    root = repo / root
root = root / "speed2"
out = root / "flight1000"
if len(sys.argv) == 1:
    paths = [root / "eval/zero13000_v2.5_t10_s123.json"]
    paths += sorted((root / "eval").glob("flight_run*_v2.5_t10_s123.json"))
    rows = [(path, json.loads(path.read_text())) for path in paths]
    eligible = [(path, row) for path, row in rows
                if math.isfinite(row["forward_speed_mps"]["mean"])
                and math.isfinite(row["survival_fraction"]) and row["survival_fraction"] >= 0.95]
    path, chosen = max(eligible, key=lambda item: item[1]["forward_speed_mps"]["mean"])
    result = {"checkpoint": chosen["checkpoint"], "screening_file": str(path),
              "screened_checkpoints": len(rows), "minimum_screening_survival": 0.95}
    (out / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(chosen["checkpoint"])
else:
    selected = json.loads((out / "selection.json").read_text())
    rows = [json.loads((root / "eval" / f"flight_selected_v2.5_t10_s{seed}.json").read_text())
            for seed in (123, 456, 789)]
    longer = json.loads((root / "eval/flight_selected_v2.5_t30_s456.json").read_text())
    speed = statistics.mean(row["forward_speed_mps"]["mean"] for row in rows)
    survival = statistics.mean(row["survival_fraction"] for row in rows)
    baseline = [json.loads((root / "eval" / f"zero13000_v2.5_t10_s{seed}.json").read_text())
                for seed in (123, 456, 789)]
    baseline_speed = statistics.mean(row["forward_speed_mps"]["mean"] for row in baseline)
    improved = math.isfinite(speed) and speed > baseline_speed and survival >= 0.95
    result = {**selected, "mean_speed_mps": speed, "survival_fraction": survival,
              "improved_over_retained_baseline": improved,
              "recommended_checkpoint": selected["checkpoint"] if improved else baseline[0]["checkpoint"],
              "long_rollout_speed_mps": longer["forward_speed_mps"]["mean"],
              "long_rollout_survival": longer["survival_fraction"],
              "target_reached": math.isfinite(speed) and speed >= 2.0 and survival >= 0.95
                  and math.isfinite(longer["forward_speed_mps"]["mean"])
                  and longer["forward_speed_mps"]["mean"] >= 2.0 and longer["survival_fraction"] >= 0.90,
              "target_definition": "At least 2.0 m/s over three 10-second seeds and a 30-second check; survival at least 95% and 90%, respectively"}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
