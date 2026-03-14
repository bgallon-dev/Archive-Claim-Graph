"""
Convert output JSON files in `out/` to CSV files in `csv_out/`.

For each .json file, a subfolder is created under csv_out/ and each
top-level key that holds a list or dict is written as its own CSV file.

Usage:
    python scripts/json_to_csv.py [--out-dir OUT] [--src-dir SRC]

Defaults:
    SRC = out/
    OUT = csv_out/
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten(obj: dict, prefix: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict into dot-separated keys."""
    result = {}
    for k, v in obj.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, key, sep))
        elif isinstance(v, list):
            result[key] = json.dumps(v)
        else:
            result[key] = v
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    # Collect all keys across all rows so sparse rows still get columns
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_file(json_path: Path, out_dir: Path) -> None:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Create a subfolder named after the JSON file (without extension)
    subfolder = out_dir / json_path.stem
    subfolder.mkdir(parents=True, exist_ok=True)

    if isinstance(data, list):
        # Top-level list: write a single CSV
        rows = [flatten(item) if isinstance(item, dict) else {"value": item} for item in data]
        write_csv(rows, subfolder / "data.csv")
        print(f"  -> {subfolder / 'data.csv'}  ({len(rows)} rows)")
        return

    if not isinstance(data, dict):
        print(f"  [skip] {json_path.name}: unsupported top-level type {type(data).__name__}")
        return

    for key, value in data.items():
        csv_path = subfolder / f"{key}.csv"
        if isinstance(value, list):
            rows = [flatten(item) if isinstance(item, dict) else {"value": item} for item in value]
            write_csv(rows, csv_path)
            print(f"  -> {csv_path}  ({len(rows)} rows)")
        elif isinstance(value, dict):
            rows = [flatten(value)]
            write_csv(rows, csv_path)
            print(f"  -> {csv_path}  (1 row)")
        else:
            # Scalar — write as a single-cell CSV
            csv_path.write_text(f"{key}\n{value}\n", encoding="utf-8")
            print(f"  -> {csv_path}  (scalar)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert output JSONs to CSVs.")
    parser.add_argument("--src-dir", default="out", help="Directory containing .json files")
    parser.add_argument("--out-dir", default="csv_out", help="Directory to write CSVs into")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    if not src.exists():
        print(f"Source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(src.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {src}", file=sys.stderr)
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    print(f"Source : {src.resolve()}")
    print(f"Output : {out.resolve()}")
    print()

    for json_path in json_files:
        print(f"Processing {json_path.name}")
        process_file(json_path, out)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
