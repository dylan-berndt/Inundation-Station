"""Bring existing `checkpoints/*/metrics.json` files onto the current schema.

Two things were wrong in every file written before this change, both from
`test.ipynb`:

1. The per-gauge mean and standard deviation were stored under each other's
   names - `"targetDev"` held `targets.mean` and `"targetMean"` held
   `targets.deviation`. `compare.py` then computed KGE's variability and bias
   ratios against the wrong quantities, so every KGE ever reported from these
   files is wrong. Correcting it moves the median KGE of the two runs in the
   last comparison from 0.403/0.385 to 0.605/0.597.

2. Threshold column 0 was labelled a "1 year return period" but
   `max(1 - 1/1, 0.01)` resolves to the 1st percentile of the fitted
   log-Pearson III distribution - a low-flow threshold that about half of all
   days exceed. It is dropped, leaving [2, 5, 10] years.

The rewrite is idempotent (files carrying `"schema": 2` are skipped) and git
holds the originals, so `git checkout -- checkpoints` undoes it.

    python migrateMetrics.py              # rewrite every metrics.json
    python migrateMetrics.py --dry-run    # report what would change
"""

import argparse
import json
import os
from glob import glob

LEGACY_THRESHOLD_COUNT = 4
CURRENT_SCHEMA = 2


def migrateGauge(entry):
    if int(entry.get("schema", 1)) >= CURRENT_SCHEMA:
        return False

    entry["targetMean"], entry["targetDev"] = entry["targetDev"], entry["targetMean"]

    for key in ("tp", "fp", "fn"):
        rows = entry.get(key)
        if rows and len(rows[0]) == LEGACY_THRESHOLD_COUNT:
            entry[key] = [row[1:] for row in rows]

    entry["schema"] = CURRENT_SCHEMA
    return True


def migrateFile(path, dryRun=False):
    with open(path) as handle:
        metrics = json.load(handle)

    changed = sum(migrateGauge(entry) for entry in metrics.values())
    if changed and not dryRun:
        with open(path, "w") as handle:
            json.dump(metrics, handle, indent=4)

    return changed, len(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default="checkpoints")
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.root, "*", "metrics.json")))
    if not paths:
        print(f"no metrics.json found under {args.root}/")
        return

    for path in paths:
        changed, total = migrateFile(path, dryRun=args.dry_run)
        run = os.path.basename(os.path.dirname(path))
        state = "would migrate" if args.dry_run else "migrated"
        print(f"{run:40s} {state} {changed}/{total} gauges" if changed else f"{run:40s} already current")


if __name__ == "__main__":
    main()
