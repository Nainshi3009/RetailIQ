"""Load and inspect a retail sales CSV stored in the project data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = Path("data") / "Sample - Superstore.csv"


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file as UTF-8, falling back to cp1252 when needed."""
    resolved_path = csv_path
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    try:
        return pd.read_csv(resolved_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(resolved_path, encoding="cp1252")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and inspect a retail sales CSV from the project data folder."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=(
            "CSV path, relative to the project root or absolute "
            "(default: data/Sample - Superstore.csv)"
        ),
    )
    args = parser.parse_args()

    csv_path = args.file
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    try:
        data = load_csv(args.file)
    except FileNotFoundError:
        print(
            f"Dataset not found: {csv_path}\n"
            "Place the downloaded CSV in the data folder or pass its path "
            "with --file. No data was generated.",
            file=sys.stderr,
        )
        return 1
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print(
            f"Could not read a valid CSV from {csv_path}: {error}",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        print(f"Could not access {csv_path}: {error}", file=sys.stderr)
        return 1

    print(f"Dataset: {csv_path}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print("\nColumn names:")
    for column in data.columns:
        print(f"- {column}")
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nSample records (first 5 rows):")
    print(data.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
