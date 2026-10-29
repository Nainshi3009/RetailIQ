"""Tests for SQLite database building and SQL analysis exports."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.build_database import EXPECTED_COLUMNS, build_database
from src.run_sql_analysis import run_sql_analysis


def fixture_data() -> pd.DataFrame:
    rows = [
        [1, "O-1", "2017-01-02", "2017-01-04", "First Class", "C-1", "Alice", "Consumer", "US", "A", "CA", 90001, "West", "P-1", "Furniture", "Chairs", "Chair", 100.0, 2, 0.1, 20.0],
        [2, "O-1", "2017-01-02", "2017-01-04", "First Class", "C-1", "Alice", "Consumer", "US", "A", "CA", 90001, "West", "P-2", "Technology", "Phones", "Phone", 50.0, 1, 0.2, -5.0],
        [3, "O-2", "2018-02-03", "2018-02-06", "Standard Class", "C-2", "Bob", "Corporate", "US", "B", "NY", 10001, "East", "P-3", "Furniture", "Tables", "Table", 0.0, 1, 0.0, 2.0],
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class SqlAnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "cleaned.csv"
        fixture_data().to_csv(self.csv_path, index=False)
        self.database_path = self.root / "retailiq.db"
        self.output_path = self.root / "reports" / "sql"

    def test_builds_database_and_matches_csv_row_count(self) -> None:
        count = build_database(self.csv_path, self.database_path)

        with closing(sqlite3.connect(self.database_path)) as connection:
            actual = connection.execute("SELECT COUNT(*) FROM retail_sales").fetchone()[0]
        self.assertEqual(count, len(fixture_data()))
        self.assertEqual(actual, len(fixture_data()))

    def test_analysis_returns_expected_totals_and_margin(self) -> None:
        build_database(self.csv_path, self.database_path)
        results = run_sql_analysis(self.database_path, self.output_path)

        totals = results["overall_totals"].iloc[0]
        self.assertEqual(totals["row_count"], 3)
        self.assertEqual(totals["total_sales"], 150.0)
        self.assertEqual(totals["total_profit"], 17.0)
        self.assertAlmostEqual(totals["overall_profit_margin_pct"], 17.0 / 150.0 * 100.0)
        self.assertEqual(len(results["top_10_products_by_sales"]), 3)
        self.assertEqual(len(results["negative_profit_products"]), 1)
        self.assertEqual(
            results["negative_profit_products"].iloc[0]["product_id"],
            "P-2",
        )

    def test_database_rebuild_is_safe_and_refreshes_rows(self) -> None:
        build_database(self.csv_path, self.database_path)
        fixture_data().iloc[:2].to_csv(self.csv_path, index=False)

        count = build_database(self.csv_path, self.database_path)

        with closing(sqlite3.connect(self.database_path)) as connection:
            actual = connection.execute("SELECT COUNT(*) FROM retail_sales").fetchone()[0]
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertEqual(count, 2)
        self.assertEqual(actual, 2)
        self.assertEqual(table_names, {"retail_sales"})

    def test_missing_input_is_reported_without_creating_database(self) -> None:
        missing = self.root / "missing.csv"
        with self.assertRaises(FileNotFoundError):
            build_database(missing, self.database_path)
        self.assertFalse(self.database_path.exists())

    def test_rejects_columns_outside_database_schema(self) -> None:
        data = fixture_data()
        data["unexpected"] = "not in SQL schema"
        data.to_csv(self.csv_path, index=False)

        with self.assertRaisesRegex(ValueError, "not present in the SQL schema"):
            build_database(self.csv_path, self.database_path)

        self.assertFalse(self.database_path.exists())

    def test_sql_analysis_writes_all_named_query_csvs(self) -> None:
        build_database(self.csv_path, self.database_path)
        results = run_sql_analysis(self.database_path, self.output_path)

        self.assertEqual(len(results), 12)
        for name, result in results.items():
            output = self.output_path / f"{name}.csv"
            self.assertTrue(output.exists())
            loaded = pd.read_csv(output)
            self.assertEqual(len(loaded), len(result))

    def test_zero_total_sales_has_null_profit_margin(self) -> None:
        data = fixture_data()
        data.loc[:, "sales"] = 0.0
        data.loc[:, "profit"] = 5.0
        data.to_csv(self.csv_path, index=False)
        build_database(self.csv_path, self.database_path)

        results = run_sql_analysis(self.database_path, self.output_path)

        self.assertTrue(pd.isna(results["overall_totals"].iloc[0]["overall_profit_margin_pct"]))


if __name__ == "__main__":
    unittest.main()
