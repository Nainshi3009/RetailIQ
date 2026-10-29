"""Assess and clean the Superstore dataset without changing its source file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if __package__:
    from .load_data import DEFAULT_CSV_PATH, PROJECT_ROOT, load_csv
else:
    from load_data import DEFAULT_CSV_PATH, PROJECT_ROOT, load_csv


DEFAULT_CLEANED_PATH = Path("data") / "cleaned" / "superstore_cleaned.csv"
DEFAULT_REPORT_PATH = Path("reports") / "data_quality_report.json"
DATE_FORMAT = "%m/%d/%Y"
DATE_COLUMNS = ("Order Date", "Ship Date")
INTEGER_COLUMNS = ("Row ID", "Postal Code", "Quantity")
NUMERIC_COLUMNS = INTEGER_COLUMNS + ("Sales", "Discount", "Profit")
ORDER_LINE_KEY = ("Order ID", "Product ID")
EXAMPLE_LIMIT = 5


def _resolve_path(path: Path) -> Path:
    """Resolve a relative path from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_path(path: Path) -> str:
    """Use a project-relative report path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def snake_case_column_mapping(columns: Iterable[str]) -> dict[str, str]:
    """Create a stable original-to-snake-case mapping and reject collisions."""
    mapping = {
        column: re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", column))
        .strip("_")
        .lower()
        for column in columns
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Column names collide after snake_case standardization.")
    return mapping


def _examples(series: pd.Series, mask: pd.Series) -> list[dict[str, Any]]:
    """Return up to five original CSV row numbers and values for flagged rows."""
    examples: list[dict[str, Any]] = []
    for position, value in enumerate(series.to_numpy()):
        if bool(mask.iloc[position]):
            examples.append({"csv_row": position + 2, "value": str(value)})
            if len(examples) == EXAMPLE_LIMIT:
                break
    return examples


def _json_number(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    return int(value) if isinstance(value, int) else float(value)


def _analyze_numeric_column(
    source: pd.Series, cleaned: pd.Series, source_dtype: str
) -> dict[str, Any]:
    parsed = pd.to_numeric(source, errors="coerce")
    conversion_failures = source.notna() & parsed.isna()
    present = parsed.dropna()
    if present.empty:
        lower_fence = upper_fence = None
        below_fence = above_fence = pd.Series(False, index=parsed.index)
    else:
        first_quartile = present.quantile(0.25)
        third_quartile = present.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_fence = first_quartile - 1.5 * interquartile_range
        upper_fence = third_quartile + 1.5 * interquartile_range
        below_fence = parsed < lower_fence
        above_fence = parsed > upper_fence

    integer_value_failures = pd.Series(False, index=parsed.index)
    if source.name in INTEGER_COLUMNS:
        integer_value_failures = parsed.notna() & (parsed % 1 != 0)

    outside_discount_range = pd.Series(False, index=parsed.index)
    if source.name == "Discount":
        outside_discount_range = parsed.notna() & ~parsed.between(0, 1)

    return {
        "source_data_type": source_dtype,
        "cleaned_data_type": str(cleaned.dtype),
        "missing_count": int(source.isna().sum()),
        "conversion_failure_count": int(conversion_failures.sum()),
        "conversion_failure_examples": _examples(source, conversion_failures),
        "non_integer_count": int(integer_value_failures.sum()),
        "non_integer_examples": _examples(source, integer_value_failures),
        "negative_count": int((parsed < 0).sum()),
        "zero_count": int((parsed == 0).sum()),
        "outside_expected_range_count": int(outside_discount_range.sum()),
        "outside_expected_range_examples": _examples(source, outside_discount_range),
        "minimum": _json_number(present.min()) if not present.empty else None,
        "maximum": _json_number(present.max()) if not present.empty else None,
        "iqr_outliers": {
            "method": "values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR",
            "lower_fence": _json_number(lower_fence),
            "upper_fence": _json_number(upper_fence),
            "below_lower_fence_count": int(below_fence.sum()),
            "above_upper_fence_count": int(above_fence.sum()),
            "below_lower_fence_examples": _examples(source, below_fence),
            "above_upper_fence_examples": _examples(source, above_fence),
        },
    }


def _duplicate_order_line_report(data: pd.DataFrame) -> dict[str, Any]:
    available_keys = [column for column in ORDER_LINE_KEY if column in data.columns]
    if len(available_keys) != len(ORDER_LINE_KEY):
        return {
            "key_columns": list(available_keys),
            "duplicate_group_count": 0,
            "rows_in_duplicate_groups": 0,
            "excess_rows": 0,
            "examples": [],
            "status": "Not assessed because Order ID and Product ID are both required.",
        }

    group_sizes = (
        data.groupby(list(ORDER_LINE_KEY), dropna=False, sort=False)
        .size()
        .loc[lambda sizes: sizes > 1]
    )
    examples = [
        {
            "order_id": str(key[0]),
            "product_id": str(key[1]),
            "row_count": int(count),
        }
        for key, count in group_sizes.head(EXAMPLE_LIMIT).items()
    ]
    return {
        "key_columns": list(ORDER_LINE_KEY),
        "duplicate_group_count": int(len(group_sizes)),
        "rows_in_duplicate_groups": int(group_sizes.sum()),
        "excess_rows": int((group_sizes - 1).sum()),
        "examples": examples,
        "status": "Potential matches only; no order lines were removed.",
    }


def clean_and_assess(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert supported fields and return the cleaned frame and quality report."""
    original = data.copy()
    cleaned = data.copy()
    report: dict[str, Any] = {
        "row_count": int(len(original)),
        "column_count": int(len(original.columns)),
        "duplicate_rows": int(original.duplicated().sum()),
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in original.isna().sum().items()
        },
        "source_data_types": {
            str(column): str(dtype) for column, dtype in original.dtypes.items()
        },
        "date_format": DATE_FORMAT,
        "date_columns": {},
        "numeric_columns": {},
        "column_mapping": snake_case_column_mapping(
            str(column) for column in original.columns
        ),
        "potential_duplicate_order_lines": _duplicate_order_line_report(original),
        "investigation_guidance": [
            "IQR outliers, zero values, negative values, and potential duplicate order lines are flags for review, not automatic errors.",
            "Negative profit can be a legitimate loss and is retained.",
            "Discount values outside the inclusive range 0 to 1 are flagged for review.",
            "No rows are deleted, missing values are not replaced with zero, and outliers are not capped.",
        ],
    }

    for column in DATE_COLUMNS:
        if column not in original.columns:
            continue
        parsed = pd.to_datetime(
            original[column],
            format=DATE_FORMAT,
            errors="coerce",
            exact=True,
        )
        failures = original[column].notna() & parsed.isna()
        cleaned[column] = parsed
        report["date_columns"][column] = {
            "format": DATE_FORMAT,
            "missing_count": int(original[column].isna().sum()),
            "parse_failure_count": int(failures.sum()),
            "parse_failure_examples": _examples(original[column], failures),
        }

    for column in NUMERIC_COLUMNS:
        if column not in original.columns:
            continue
        parsed = pd.to_numeric(original[column], errors="coerce")
        if column in INTEGER_COLUMNS:
            finite_values = parsed.dropna()
            if finite_values.empty or (finite_values % 1 == 0).all():
                cleaned[column] = parsed.astype("Int64")
            else:
                cleaned[column] = parsed
        else:
            cleaned[column] = parsed.astype("float64")
        report["numeric_columns"][column] = _analyze_numeric_column(
            original[column],
            cleaned[column],
            str(original[column].dtype),
        )

    cleaned = cleaned.rename(columns=report["column_mapping"])
    report["cleaned_data_types"] = {
        str(column): str(dtype) for column, dtype in cleaned.dtypes.items()
    }
    return cleaned, report


def run_cleaning(
    source_path: Path = DEFAULT_CSV_PATH,
    cleaned_path: Path = DEFAULT_CLEANED_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Create the cleaned CSV and JSON report without modifying the source."""
    source = _resolve_path(source_path)
    output = _resolve_path(cleaned_path)
    report_output = _resolve_path(report_path)
    resolved_targets = (source.resolve(), output.resolve(), report_output.resolve())
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Source, cleaned dataset, and report paths must be different.")

    data = load_csv(source)
    cleaned, report = clean_and_assess(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False, date_format="%Y-%m-%d")
    report["source_file"] = _report_path(source)
    report["cleaned_file"] = _report_path(output)
    report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assess and clean the Superstore CSV without changing its source."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input CSV path (default: data/Sample - Superstore.csv)",
    )
    args = parser.parse_args()

    source = _resolve_path(args.file)
    try:
        report = run_cleaning(source)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError, OSError, ValueError) as error:
        print(f"Could not clean dataset {source}: {error}", file=sys.stderr)
        return 1

    print(f"Source dataset: {source}")
    print(f"Rows: {report['row_count']}")
    print(f"Columns: {report['column_count']}")
    print(f"Exact duplicate rows: {report['duplicate_rows']}")
    print(f"Cleaned dataset: {_resolve_path(DEFAULT_CLEANED_PATH)}")
    print(f"Quality report: {_resolve_path(DEFAULT_REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
