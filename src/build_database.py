"""Build or safely refresh the local SQLite database from the cleaned CSV."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from .load_data import PROJECT_ROOT
else:
    from load_data import PROJECT_ROOT


DEFAULT_CSV = Path("data") / "cleaned" / "superstore_cleaned.csv"
DEFAULT_DATABASE = Path("data") / "retailiq.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
TABLE_NAME = "retail_sales"
STAGING_TABLE = "retail_sales_staging"
EXPECTED_COLUMNS = [
    "row_id",
    "order_id",
    "order_date",
    "ship_date",
    "ship_mode",
    "customer_id",
    "customer_name",
    "segment",
    "country",
    "city",
    "state",
    "postal_code",
    "region",
    "product_id",
    "category",
    "sub_category",
    "product_name",
    "sales",
    "quantity",
    "discount",
    "profit",
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sqlite_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return value.item()
    return value


def _sql_statements(script: str) -> list[str]:
    """Split SQL on complete SQLite statements, ignoring comment semicolons."""
    statements: list[str] = []
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            if pending.strip():
                statements.append(pending)
            pending = ""
    if pending.strip():
        raise ValueError("SQL script contains an incomplete statement.")
    return statements


def build_database(
    csv_path: Path = DEFAULT_CSV,
    database_path: Path = DEFAULT_DATABASE,
) -> int:
    """Replace the database table transactionally using the cleaned CSV."""
    source = _resolve(csv_path)
    database = _resolve(database_path)
    if not source.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {source}")

    data = pd.read_csv(source)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
    unexpected_columns = [
        column for column in data.columns if column not in EXPECTED_COLUMNS
    ]
    if unexpected_columns:
        raise ValueError(
            "Cleaned dataset contains columns not present in the SQL schema: "
            + ", ".join(unexpected_columns)
        )
    if data.columns.tolist() != EXPECTED_COLUMNS:
        data = data[EXPECTED_COLUMNS]

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = _sql_statements(schema)
    if not statements:
        raise ValueError(f"No SQL statements found in {SCHEMA_PATH}")
    staging_table_statement, replacements = re.subn(
        rf"CREATE TABLE IF NOT EXISTS {TABLE_NAME}",
        f"CREATE TABLE {STAGING_TABLE}",
        statements[0],
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Could not prepare staging table from {SCHEMA_PATH}")

    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE}")
        connection.execute(staging_table_statement)
        placeholders = ", ".join("?" for _ in EXPECTED_COLUMNS)
        columns = ", ".join(f'"{column}"' for column in EXPECTED_COLUMNS)
        insert = (
            f"INSERT INTO {STAGING_TABLE} ({columns}) "
            f"VALUES ({placeholders})"
        )
        rows = (
            tuple(_sqlite_value(value) for value in row)
            for row in data.itertuples(index=False, name=None)
        )
        connection.executemany(insert, rows)
        database_row_count = connection.execute(
            f"SELECT COUNT(*) FROM {STAGING_TABLE}"
        ).fetchone()[0]
        if database_row_count != len(data):
            raise RuntimeError(
                f"Inserted {database_row_count} rows; expected {len(data)}."
            )
        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.execute(
            f"ALTER TABLE {STAGING_TABLE} RENAME TO {TABLE_NAME}"
        )
        for statement in statements[1:]:
            connection.execute(statement)
        connection.commit()
        return int(database_row_count)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the RetailIQ SQLite database from the cleaned CSV."
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_CSV, help="Cleaned CSV input")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database output",
    )
    args = parser.parse_args()
    try:
        row_count = build_database(args.file, args.database)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
            OSError, sqlite3.Error, ValueError, RuntimeError) as error:
        print(f"Could not build RetailIQ database: {error}", file=sys.stderr)
        return 1
    print(f"Database: {_resolve(args.database)}")
    print(f"Loaded rows: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
