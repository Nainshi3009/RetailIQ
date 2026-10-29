"""Execute named SQL analyses and write one CSV result per query."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

if __package__:
    from .build_database import DEFAULT_DATABASE, _resolve
    from .load_data import PROJECT_ROOT
else:
    from build_database import DEFAULT_DATABASE, _resolve
    from load_data import PROJECT_ROOT


DEFAULT_OUTPUT_DIRECTORY = Path("reports") / "sql"
ANALYSIS_SQL = PROJECT_ROOT / "sql" / "analysis.sql"
QUERY_PATTERN = re.compile(
    r"(?m)^-- name: ([a-z][a-z0-9_]*)\s*$"
)


def load_queries(sql_path: Path = ANALYSIS_SQL) -> dict[str, str]:
    """Parse named query blocks from analysis.sql."""
    content = sql_path.read_text(encoding="utf-8")
    matches = list(QUERY_PATTERN.finditer(content))
    if not matches:
        raise ValueError(f"No named SQL queries found in {sql_path}")
    queries: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        query = content[match.end():end].strip()
        if not query or not query.rstrip().endswith(";"):
            raise ValueError(f"Query {match.group(1)} is empty or lacks a semicolon.")
        queries[match.group(1)] = query
    if len(queries) != len(matches):
        raise ValueError(f"Duplicate query names found in {sql_path}")
    return queries


def run_sql_analysis(
    database_path: Path = DEFAULT_DATABASE,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    sql_path: Path = ANALYSIS_SQL,
) -> dict[str, pd.DataFrame]:
    """Run all named SQL queries and save their result sets as CSV files."""
    database = _resolve(database_path)
    output = _resolve(output_directory)
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database}")
    queries = load_queries(sql_path)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    with closing(sqlite3.connect(database)) as connection:
        for name, query in queries.items():
            result = pd.read_sql_query(query, connection)
            result.to_csv(output / f"{name}.csv", index=False)
            results[name] = result
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute RetailIQ SQL analyses and export result CSVs."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/retailiq.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="CSV output directory (default: reports/sql)",
    )
    args = parser.parse_args()
    try:
        results = run_sql_analysis(args.database, args.output)
    except (FileNotFoundError, OSError, sqlite3.Error, pd.errors.DatabaseError, ValueError) as error:
        print(f"Could not run SQL analysis: {error}", file=sys.stderr)
        return 1
    print(f"Database: {_resolve(args.database)}")
    print(f"Saved {len(results)} query result files to {_resolve(args.output)}")
    for name, result in results.items():
        print(f"- {name}.csv ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
