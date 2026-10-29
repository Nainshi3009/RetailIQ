"""Focused tests for RetailIQ's advanced, explainable analytics."""

from __future__ import annotations

import unittest

import pandas as pd

from src.advanced_analytics import (
    aggregate_discount_bubbles,
    aggregate_heatmaps,
    aggregate_treemap,
    calculate_customer_rfm,
    calculate_product_pareto,
    estimate_discount_scenario,
    find_profit_leaks,
    outlier_summary,
    profit_contributions,
    score_profit_anomalies,
)
from src.app import _build_advanced_charts, calculate_kpis, filter_sales_data
from tests.test_dashboard import dashboard_fixture


def advanced_fixture() -> pd.DataFrame:
    data = dashboard_fixture().copy()
    data["order_date"] = pd.to_datetime(data["order_date"])
    return data


class AdvancedAnalyticsTests(unittest.TestCase):
    def test_heatmap_and_treemap_aggregations_use_sales_profit_and_margin(self) -> None:
        data = advanced_fixture()
        month_sales, month_profit, region_category = aggregate_heatmaps(data)
        treemap, excluded = aggregate_treemap(data)
        bubbles = aggregate_discount_bubbles(data)
        subcategory_bubbles = aggregate_discount_bubbles(data, "sub_category")

        self.assertEqual(month_sales.sum().sum(), 150.0)
        self.assertEqual(month_profit.sum().sum(), 17.0)
        self.assertEqual(len(region_category), 3)
        self.assertTrue(
            region_category.loc[region_category["sales"].ne(0), "profit_margin_pct"].notna().all()
        )
        self.assertTrue(region_category.loc[region_category["sales"].eq(0), "profit_margin_pct"].isna().all())
        self.assertEqual(excluded, 1)
        self.assertEqual(treemap["sales"].sum(), 150.0)
        self.assertEqual(bubbles["row_count"].sum(), len(data))
        self.assertEqual(subcategory_bubbles["row_count"].sum(), len(data))
        with self.assertRaisesRegex(ValueError, "support product or sub_category"):
            aggregate_discount_bubbles(data, "order_line")

    def test_advanced_chart_builder_supports_product_and_subcategory_bubbles(self) -> None:
        data = advanced_fixture()

        product_chart = _build_advanced_charts(data)
        subcategory_chart = _build_advanced_charts(data, "sub_category")

        self.assertIn("Discount and product profitability", product_chart["discount_bubbles"].layout.title.text)
        self.assertIn(
            "Discount and sub category profitability",
            subcategory_chart["discount_bubbles"].layout.title.text,
        )
        self.assertEqual(len(product_chart["discount_bubbles"].data), 1)
        self.assertEqual(len(subcategory_chart["discount_bubbles"].data), 1)

    def test_discount_chart_passes_requested_aggregation_to_bubble_builder(self) -> None:
        data = advanced_fixture()

        for aggregation in ("product", "sub_category"):
            with self.subTest(aggregation=aggregation):
                chart = _build_advanced_charts(data, aggregation)["discount_bubbles"]
                expected = aggregate_discount_bubbles(data, aggregation)

                self.assertEqual(len(chart.data[0].x), len(expected))

    def test_pareto_ranks_products_and_uses_selected_sales_denominator(self) -> None:
        data = advanced_fixture()
        pareto = calculate_product_pareto(data)

        self.assertEqual(pareto.iloc[0]["product_id"], "P-1")
        self.assertAlmostEqual(pareto.iloc[0]["sales_contribution_pct"], 100 * 100 / 150)
        self.assertAlmostEqual(pareto.iloc[-1]["cumulative_sales_pct"], 100.0)

    def test_pareto_handles_zero_total_sales(self) -> None:
        data = advanced_fixture()
        data["sales"] = 0.0

        pareto = calculate_product_pareto(data)

        self.assertTrue(pareto["sales_contribution_pct"].isna().all())
        self.assertTrue(pareto["cumulative_sales_pct"].isna().all())

    def test_category_and_subcategory_waterfalls_reconcile(self) -> None:
        data = advanced_fixture()
        total = data["profit"].sum()

        for dimension in ("category", "sub_category"):
            contributions = profit_contributions(data, dimension)
            self.assertAlmostEqual(contributions["profit"].sum(), total)
        self.assertIn("category", profit_contributions(data, "sub_category").columns)

    def test_outlier_summary_uses_1_5_iqr_and_empty_input(self) -> None:
        data = advanced_fixture()
        result = outlier_summary(data)
        empty = outlier_summary(data.iloc[0:0])

        self.assertEqual(result["sales"]["valid_count"], 3)
        self.assertGreaterEqual(result["profit"]["outlier_count"], 0)
        self.assertEqual(empty["sales"]["outlier_count"], 0)
        self.assertIsNone(empty["sales"]["lower_fence"])

    def test_rfm_uses_as_of_date_and_distinct_orders(self) -> None:
        data = advanced_fixture()
        data.loc[1, "order_id"] = data.loc[0, "order_id"]
        rfm, info = calculate_customer_rfm(data)
        west_customer = rfm.loc[rfm["customer_id"].eq("C-1")].iloc[0]

        self.assertEqual(info["as_of_date"], pd.Timestamp("2018-02-03"))
        self.assertEqual(west_customer["frequency_orders"], 1)
        self.assertEqual(west_customer["recency_days"], 397)
        self.assertEqual(west_customer["monetary_sales"], 150.0)
        self.assertEqual(len(info["segment_rules"]), 4)

    def test_rfm_excludes_missing_customer_ids_and_handles_empty_data(self) -> None:
        data = advanced_fixture()
        data.loc[:, "customer_id"] = None
        rfm, info = calculate_customer_rfm(data)
        empty, empty_info = calculate_customer_rfm(data.iloc[0:0])

        self.assertTrue(rfm.empty)
        self.assertEqual(info["excluded_missing_customer_rows"], 3)
        self.assertTrue(empty.empty)
        self.assertIsNone(empty_info["as_of_date"])

    def test_profit_leaks_distinguish_negative_profit_from_low_margin(self) -> None:
        data = pd.DataFrame(
            [
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-LOSS",
                    "product_name": "Loss leader",
                    "sales": 500.0,
                    "profit": -50.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "West",
                    "discount": 0.2,
                    "order_id": "LOSS-1",
                },
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-LOSS",
                    "product_name": "Loss leader",
                    "sales": 500.0,
                    "profit": -50.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "West",
                    "discount": 0.2,
                    "order_id": "LOSS-2",
                },
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-LOW",
                    "product_name": "Low margin",
                    "sales": 500.0,
                    "profit": 20.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "East",
                    "discount": 0.1,
                    "order_id": "LOW-1",
                },
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-LOW",
                    "product_name": "Low margin",
                    "sales": 500.0,
                    "profit": 20.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "East",
                    "discount": 0.1,
                    "order_id": "LOW-2",
                },
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-HIGH",
                    "product_name": "High margin",
                    "sales": 1000.0,
                    "profit": 500.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "South",
                    "discount": 0.0,
                    "order_id": "HIGH-1",
                },
                {
                    **advanced_fixture().iloc[0].to_dict(),
                    "product_id": "P-HIGH",
                    "product_name": "High margin",
                    "sales": 1000.0,
                    "profit": 500.0,
                    "category": "Technology",
                    "sub_category": "Phones",
                    "region": "South",
                    "discount": 0.0,
                    "order_id": "HIGH-2",
                },
            ]
        )
        findings = find_profit_leaks(data, minimum_group_rows=2)

        loss = findings.loc[
            findings["finding_type"].eq("High-sales negative-profit product")
        ]
        low_margin = findings.loc[
            findings["finding_type"].eq("High-sales product below category margin")
        ]
        self.assertFalse(loss.empty)
        self.assertTrue((loss["profit"] < 0).all())
        self.assertFalse(low_margin.empty)
        self.assertTrue((low_margin["profit"] >= 0).all())
        self.assertTrue(low_margin["impact_basis"].str.contains("not recoverable savings").all())

    def test_anomaly_scoring_requires_peer_cells_and_minimum_sample(self) -> None:
        base = advanced_fixture().iloc[0].copy()
        rows = []
        for index, (region, profit) in enumerate(
            (("East", 30.0), ("West", 20.0), ("South", 10.0), ("Central", -50.0))
        ):
            row = base.copy()
            row["region"] = region
            row["category"] = "Office Supplies"
            row["sub_category"] = "Binders"
            row["discount"] = 0.2
            row["sales"] = 100.0
            row["profit"] = profit
            row["order_id"] = f"O-{index}"
            rows.append(row)
        data = pd.DataFrame(rows)

        scores = score_profit_anomalies(data, minimum_rows=1)
        low_cell = scores.loc[scores["region"].eq("Central")].iloc[0]

        self.assertLess(low_cell["robust_deviation_score"], -2.5)
        self.assertTrue(bool(low_cell["unusual_low_margin"]))
        self.assertEqual(scores["row_count"].min(), 1)
        small_group_scores = score_profit_anomalies(data, minimum_rows=2)
        self.assertTrue(small_group_scores["robust_deviation_score"].isna().all())
        self.assertFalse(small_group_scores["unusual_low_margin"].any())

    def test_filtered_analytics_match_filtered_dashboard_kpis(self) -> None:
        data = advanced_fixture()
        filtered = filter_sales_data(data, selections={"region": ["West"], "category": ["Furniture"]})

        kpis = calculate_kpis(filtered)
        pareto = calculate_product_pareto(filtered)
        contributions = profit_contributions(filtered, "category")
        treemap, _ = aggregate_treemap(filtered)

        self.assertEqual(kpis["total_sales"], 100.0)
        self.assertEqual(float(pareto["sales"].sum()), kpis["total_sales"])
        self.assertEqual(float(contributions["profit"].sum()), kpis["total_profit"])
        self.assertEqual(float(treemap["sales"].sum()), kpis["total_sales"])

    def test_zero_sales_leak_groups_are_not_assigned_margins(self) -> None:
        data = advanced_fixture()
        data["sales"] = 0.0

        findings = find_profit_leaks(data)

        self.assertTrue(findings.empty)

    def test_empty_analytics_and_missing_required_columns_are_handled(self) -> None:
        empty = advanced_fixture().iloc[0:0]
        self.assertTrue(find_profit_leaks(empty).empty)
        self.assertTrue(score_profit_anomalies(empty).empty)
        self.assertTrue(calculate_product_pareto(empty)["cumulative_sales_pct"].isna().all())
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            calculate_product_pareto(pd.DataFrame({"sales": [10.0]}))

    def test_discount_scenario_formula_and_assumptions(self) -> None:
        data = advanced_fixture().iloc[:2].copy()
        result = estimate_discount_scenario(data, 0.1)

        self.assertAlmostEqual(result["baseline_sales"], 150.0)
        self.assertAlmostEqual(result["baseline_profit"], 15.0)
        self.assertAlmostEqual(result["scenario_sales_estimate"], 156.25)
        self.assertAlmostEqual(result["scenario_profit_estimate"], 21.25)
        self.assertAlmostEqual(result["profit_difference_estimate"], 6.25)
        self.assertIn("not a forecast", result["assumption"])

    def test_scenario_rejects_invalid_discount_and_zero_denominators_are_safe(self) -> None:
        data = advanced_fixture().iloc[:1].copy()
        data["sales"] = 0.0
        with self.assertRaisesRegex(ValueError, "Historical sales are zero"):
            estimate_discount_scenario(data, 0.1)
        with self.assertRaisesRegex(ValueError, "less than 1"):
            estimate_discount_scenario(data, 1.0)


if __name__ == "__main__":
    unittest.main()
