"""Tests for data cleaning and quality reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.clean_data import clean_and_assess, run_cleaning, snake_case_column_mapping


class CleanDataTests(unittest.TestCase):
    def test_parses_dates_with_explicit_format_and_reports_failures(self) -> None:
        data = pd.DataFrame(
            {
                "Order Date": ["1/2/2017", "not-a-date", None],
                "Ship Date": ["1/4/2017", "1/5/2017", "bad"],
            }
        )

        cleaned, report = clean_and_assess(data)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["order_date"]))
        self.assertEqual(cleaned.loc[0, "order_date"], pd.Timestamp("2017-01-02"))
        self.assertTrue(pd.isna(cleaned.loc[1, "order_date"]))
        self.assertEqual(report["date_columns"]["Order Date"]["parse_failure_count"], 1)
        self.assertEqual(report["date_columns"]["Order Date"]["missing_count"], 1)
        self.assertEqual(report["date_columns"]["Ship Date"]["parse_failure_count"], 1)

    def test_maps_columns_to_snake_case(self) -> None:
        mapping = snake_case_column_mapping(["Row ID", "Sub-Category", "Customer Name"])

        self.assertEqual(
            mapping,
            {
                "Row ID": "row_id",
                "Sub-Category": "sub_category",
                "Customer Name": "customer_name",
            },
        )

    def test_rejects_column_mapping_collisions(self) -> None:
        with self.assertRaises(ValueError):
            snake_case_column_mapping(["A B", "A-B"])

    def test_reports_missing_values_by_original_column(self) -> None:
        data = pd.DataFrame({"Customer Name": ["A", None], "Sales": [1.0, None]})

        _, report = clean_and_assess(data)

        self.assertEqual(report["missing_values_by_column"]["Customer Name"], 1)
        self.assertEqual(report["missing_values_by_column"]["Sales"], 1)

    def test_reports_exact_and_potential_duplicate_rows(self) -> None:
        data = pd.DataFrame(
            {
                "Order ID": ["O-1", "O-1", "O-2"],
                "Product ID": ["P-1", "P-1", "P-2"],
                "Sales": [10.0, 10.0, 20.0],
            }
        )

        _, report = clean_and_assess(data)

        self.assertEqual(report["duplicate_rows"], 1)
        potential = report["potential_duplicate_order_lines"]
        self.assertEqual(potential["duplicate_group_count"], 1)
        self.assertEqual(potential["rows_in_duplicate_groups"], 2)
        self.assertEqual(potential["excess_rows"], 1)
        self.assertIn("no order lines were removed", potential["status"])

    def test_converts_numeric_values_and_reports_invalid_tokens(self) -> None:
        data = pd.DataFrame(
            {
                "Sales": ["12.5", "not-a-number", None],
                "Quantity": ["2", "invalid", "3"],
                "Discount": ["0.2", "1.2", "-0.1"],
                "Profit": [-5.0, 0.0, 10.0],
            }
        )

        cleaned, report = clean_and_assess(data)

        self.assertEqual(report["numeric_columns"]["Sales"]["conversion_failure_count"], 1)
        self.assertEqual(report["numeric_columns"]["Sales"]["missing_count"], 1)
        self.assertEqual(report["numeric_columns"]["Quantity"]["conversion_failure_count"], 1)
        self.assertEqual(report["numeric_columns"]["Discount"]["outside_expected_range_count"], 2)
        self.assertEqual(report["numeric_columns"]["Profit"]["negative_count"], 1)
        self.assertEqual(report["numeric_columns"]["Profit"]["zero_count"], 1)
        self.assertTrue(pd.isna(cleaned.loc[1, "sales"]))
        self.assertEqual(cleaned.loc[0, "sales"], 12.5)
        self.assertIn("no rows are deleted", " ".join(report["investigation_guidance"]).lower())

    def test_writes_output_files_and_creates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            output = root / "nested" / "cleaned.csv"
            report_path = root / "reports" / "quality.json"
            source.write_text(
                "Order ID,Product ID,Order Date,Sales,Quantity\n"
                "O-1,P-1,1/2/2017,12.5,2\n",
                encoding="utf-8",
            )
            original_bytes = source.read_bytes()

            report = run_cleaning(source, output, report_path)

            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertTrue(output.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(report["row_count"], 1)
            self.assertIn("order_date", pd.read_csv(output).columns)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["column_mapping"]["Order Date"], "order_date")

    def test_refuses_to_overwrite_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("Sales\n12.5\n", encoding="utf-8")
            original_bytes = source.read_bytes()

            with self.assertRaises(ValueError):
                run_cleaning(source, source, Path(directory) / "quality.json")

            self.assertEqual(source.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
