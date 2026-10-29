"""Tests for EDA aggregation, date handling, and output generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eda import aggregate_metrics, analyze_data, prepare_data, run_eda


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O-1", "O-1", "O-2"],
            "order_date": ["2017-01-02", "2017-01-02", "not-a-date"],
            "ship_date": ["2017-01-04", "2017-01-04", "2017-01-06"],
            "ship_mode": ["First Class", "First Class", "Standard Class"],
            "customer_id": ["C-1", "C-1", "C-2"],
            "segment": ["Consumer", "Consumer", "Corporate"],
            "region": ["West", "West", "East"],
            "product_id": ["P-1", "P-2", "P-3"],
            "product_name": ["Product 1", "Product 2", "Product 3"],
            "category": ["Furniture", "Technology", "Furniture"],
            "sub_category": ["Chairs", "Phones", "Tables"],
            "sales": [100.0, 50.0, 20.0],
            "profit": [10.0, -5.0, 2.0],
            "quantity": [2, 1, 1],
            "discount": [0.0, 0.2, 0.0],
        }
    )


class EdaTests(unittest.TestCase):
    def test_aggregate_metrics_sums_and_counts_orders(self) -> None:
        result = aggregate_metrics(sample_data(), "region").set_index("region")

        self.assertEqual(result.loc["West", "sales"], 150.0)
        self.assertEqual(result.loc["West", "profit"], 5.0)
        self.assertEqual(result.loc["West", "order_count"], 1)
        self.assertEqual(result.loc["West", "row_count"], 2)

    def test_aggregate_preserves_all_missing_metric_as_missing(self) -> None:
        data = sample_data()
        data.loc[:, "profit"] = float("nan")

        result = aggregate_metrics(data, "region")

        self.assertTrue(result["profit"].isna().all())

    def test_date_handling_reports_invalid_dates_and_temporal_excludes_them(self) -> None:
        prepared, quality = prepare_data(sample_data())
        _, summary, tables = analyze_data(sample_data())

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["order_date"]))
        self.assertEqual(quality["date_columns"]["order_date"]["parse_failure_count"], 1)
        self.assertEqual(summary["temporal_trends"]["by_year"][0]["year"], 2017)
        self.assertEqual(tables["year"]["row_count"].sum(), 2)

    def test_rejects_dataset_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_data(pd.DataFrame({"sales": [10]}))

    def test_summary_separates_low_sales_from_negative_profit(self) -> None:
        _, summary, _ = analyze_data(sample_data())

        self.assertEqual(summary["products"]["lowest_sales"][0]["product_id"], "P-3")
        lowest_profit = summary["products"]["lowest_profit"][0]
        self.assertEqual(lowest_profit["product_id"], "P-2")
        self.assertLess(lowest_profit["profit"], 0)
        self.assertEqual(
            summary["discount_profit_relationship"]["observation_count"],
            3,
        )
        self.assertIn(
            "do not establish",
            summary["discount_profit_relationship"]["interpretation"],
        )

    def test_writes_json_and_interactive_html_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cleaned.csv"
            sample_data().to_csv(source, index=False)
            chart_dir = root / "charts"
            summary_path = root / "reports" / "summary.json"

            summary = run_eda(source, chart_dir, summary_path)

            self.assertTrue(summary_path.exists())
            stored = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["dataset"]["row_count"], 3)
            self.assertTrue(stored["charts"])
            for filename in summary["charts"]:
                chart = chart_dir / filename
                self.assertTrue(chart.exists())
                self.assertIn("plotly", chart.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
