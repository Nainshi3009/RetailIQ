"""Focused tests for dashboard loading, filters, metrics, and empty results."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.app import (
    DatabaseLoadError,
    build_business_insights,
    calculate_kpis,
    build_dashboard_charts,
    filter_sales_data,
    load_dashboard_data,
)
from src.build_database import EXPECTED_COLUMNS, build_database


def dashboard_fixture() -> pd.DataFrame:
    rows = [
        [1, "O-1", "2017-01-02", "2017-01-04", "First Class", "C-1", "Alice", "Consumer", "US", "A", "CA", 90001, "West", "P-1", "Furniture", "Chairs", "Chair", 100.0, 2, 0.1, 20.0],
        [2, "O-1", "2017-01-02", "2017-01-04", "First Class", "C-1", "Alice", "Consumer", "US", "A", "CA", 90001, "West", "P-2", "Technology", "Phones", "Phone", 50.0, 1, 0.2, -5.0],
        [3, "O-2", "2018-02-03", "2018-02-06", "Standard Class", "C-2", "Bob", "Corporate", "US", "B", "NY", 10001, "East", "P-3", "Furniture", "Tables", "Table", 0.0, 1, 0.0, 2.0],
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "cleaned.csv"
        self.database_path = self.root / "retailiq.db"
        dashboard_fixture().to_csv(self.csv_path, index=False)
        build_database(self.csv_path, self.database_path)

    def test_kpis_calculate_margin_orders_and_quantity(self) -> None:
        data = dashboard_fixture()
        metrics = calculate_kpis(data)

        self.assertEqual(metrics["total_sales"], 150.0)
        self.assertEqual(metrics["total_profit"], 17.0)
        self.assertAlmostEqual(metrics["profit_margin_pct"], 17 / 150 * 100)
        self.assertEqual(metrics["distinct_orders"], 2)
        self.assertEqual(metrics["quantity_sold"], 4)

    def test_filter_logic_combines_date_and_dimensions(self) -> None:
        data = load_dashboard_data(self.database_path)

        filtered = filter_sales_data(
            data,
            (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-31")),
            {"region": ["West"], "category": ["Furniture"]},
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["product_id"], "P-1")

    def test_empty_filter_result_returns_zero_totals_and_no_margin(self) -> None:
        data = filter_sales_data(
            load_dashboard_data(self.database_path),
            selections={"region": ["Not a region"]},
        )
        metrics = calculate_kpis(data)

        self.assertTrue(data.empty)
        self.assertEqual(metrics["total_sales"], 0.0)
        self.assertEqual(metrics["total_profit"], 0.0)
        self.assertIsNone(metrics["profit_margin_pct"])
        self.assertEqual(metrics["distinct_orders"], 0)
        self.assertIn("No records match", build_business_insights(data)[0])

    def test_empty_multiselect_selection_returns_no_rows(self) -> None:
        data = filter_sales_data(
            load_dashboard_data(self.database_path),
            selections={"region": []},
        )

        self.assertTrue(data.empty)

    def test_margin_is_none_when_sales_are_zero(self) -> None:
        data = dashboard_fixture()
        data["sales"] = 0.0

        self.assertIsNone(calculate_kpis(data)["profit_margin_pct"])

    def test_missing_or_invalid_database_has_helpful_error(self) -> None:
        with self.assertRaisesRegex(DatabaseLoadError, "was not found"):
            load_dashboard_data(self.root / "missing.db")

        invalid = self.root / "invalid.db"
        invalid.write_text("not a sqlite database", encoding="utf-8")
        with self.assertRaisesRegex(DatabaseLoadError, "Could not read"):
            load_dashboard_data(invalid)

    def test_database_with_missing_columns_is_rejected(self) -> None:
        incomplete = self.root / "incomplete.db"
        with closing(sqlite3.connect(incomplete)) as connection:
            connection.execute("CREATE TABLE retail_sales (sales REAL)")

        with self.assertRaisesRegex(DatabaseLoadError, "missing required columns"):
            load_dashboard_data(incomplete)

    def test_plotly_charts_use_transparent_glass_theme(self) -> None:
        charts = build_dashboard_charts(load_dashboard_data(self.database_path))

        self.assertIn("ship_mode", charts)
        self.assertEqual(charts["monthly"].layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(charts["monthly"].layout.plot_bgcolor, "rgba(0,0,0,0)")


if __name__ == "__main__":
    unittest.main()
