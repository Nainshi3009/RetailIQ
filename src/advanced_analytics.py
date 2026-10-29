"""Pure, explainable calculations for the RetailIQ advanced analytics views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CORE_COLUMNS = {
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "product_id",
    "product_name",
    "category",
    "sub_category",
    "sales",
    "profit",
    "quantity",
    "discount",
}
RFM_SEGMENT_RULES = (
    "High-value active: recency at or below the 25th percentile and both "
    "distinct-order frequency and monetary sales at or above their 75th percentiles.",
    "Frequent active: recency at or below the median and frequency at or above "
    "the 75th percentile.",
    "At-risk, lower activity: recency at or above the 75th percentile and "
    "frequency at or below the median.",
    "Other: customers not meeting the preceding rules.",
)


def _validate(data: pd.DataFrame, columns: set[str] = CORE_COLUMNS) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def _margin(profit: pd.Series, sales: pd.Series) -> pd.Series:
    return profit.div(sales.where(sales.ne(0))).mul(100)


def aggregate_heatmaps(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return month/category sales and profit pivots plus region/category metrics."""
    _validate(data, CORE_COLUMNS | {"region"})
    dated = data.loc[data["order_date"].notna()].copy()
    dated["month"] = dated["order_date"].dt.month
    monthly = (
        dated.groupby(["month", "category"], dropna=False, observed=True)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    monthly["month"] = pd.Categorical(
        monthly["month"],
        categories=range(1, 13),
        ordered=True,
    )
    month_sales = monthly.pivot(index="month", columns="category", values="sales")
    month_profit = monthly.pivot(index="month", columns="category", values="profit")
    region_category = (
        data.groupby(["region", "category"], dropna=False, observed=True)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), row_count=("sales", "size"))
        .reset_index()
    )
    region_category["profit_margin_pct"] = _margin(
        region_category["profit"], region_category["sales"]
    )
    return month_sales, month_profit, region_category


def aggregate_treemap(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aggregate category/sub-category/product values; treemap area is sales."""
    _validate(data)
    grouped = (
        data.groupby(
            ["category", "sub_category", "product_id", "product_name"],
            dropna=False,
            observed=True,
        )
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            discount=("discount", "mean"),
            row_count=("sales", "size"),
        )
        .reset_index()
    )
    grouped["profit_margin_pct"] = _margin(grouped["profit"], grouped["sales"])
    excluded = int(grouped["sales"].le(0).sum())
    return grouped.loc[grouped["sales"] > 0].copy(), excluded


def aggregate_discount_bubbles(
    data: pd.DataFrame,
    aggregation: str = "product",
) -> pd.DataFrame:
    """Aggregate at product or sub-category and observed discount to avoid overlap."""
    _validate(data)
    if aggregation not in {"product", "sub_category"}:
        raise ValueError("Discount bubbles support product or sub_category aggregation.")
    dimensions = (
        ["product_id", "product_name", "category", "sub_category", "discount"]
        if aggregation == "product"
        else ["category", "sub_category", "discount"]
    )
    grouped = (
        data.groupby(dimensions, dropna=False, observed=True)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            row_count=("order_id", "size"),
        )
        .reset_index()
    )
    grouped["profit_margin_pct"] = _margin(grouped["profit"], grouped["sales"])
    return grouped


def calculate_product_pareto(data: pd.DataFrame) -> pd.DataFrame:
    """Rank products by sales and calculate contribution to selected total sales."""
    _validate(data)
    products = (
        data.groupby(["product_id", "product_name"], dropna=False, observed=True)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), row_count=("sales", "size"))
        .reset_index()
        .sort_values(["sales", "product_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    total_sales = float(data["sales"].sum())
    products["rank"] = np.arange(1, len(products) + 1)
    if total_sales > 0:
        products["sales_contribution_pct"] = products["sales"].div(total_sales).mul(100)
        products["cumulative_sales_pct"] = products["sales_contribution_pct"].cumsum()
    else:
        products["sales_contribution_pct"] = np.nan
        products["cumulative_sales_pct"] = np.nan
    return products


def profit_contributions(data: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate positive and negative profit contributions by dimension."""
    if by not in {"category", "sub_category"}:
        raise ValueError("Profit contributions support category or sub_category.")
    _validate(data, CORE_COLUMNS | {by})
    dimensions = ["category"] if by == "category" else ["category", "sub_category"]
    return (
        data.groupby(dimensions, dropna=False, observed=True)
        .agg(profit=("profit", "sum"), sales=("sales", "sum"), row_count=("profit", "size"))
        .reset_index()
    )


def outlier_summary(data: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Summarize 1.5 x IQR flags; these statistical flags are not data errors."""
    _validate(data)
    summary: dict[str, dict[str, float | int | None]] = {}
    for column in ("sales", "profit", "discount", "quantity"):
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            summary[column] = {
                "q1": None,
                "q3": None,
                "lower_fence": None,
                "upper_fence": None,
                "outlier_count": 0,
                "valid_count": 0,
            }
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        summary[column] = {
            "q1": q1,
            "q3": q3,
            "lower_fence": low,
            "upper_fence": high,
            "outlier_count": int(((values < low) | (values > high)).sum()),
            "valid_count": int(values.count()),
        }
    return summary


def calculate_customer_rfm(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate filtered customer RFM metrics; frequency counts distinct orders."""
    _validate(data)
    valid_dates = data["order_date"].dropna()
    as_of = valid_dates.max().normalize() if not valid_dates.empty else None
    identified = data.loc[data["customer_id"].notna()].copy()
    identified = identified.loc[identified["customer_id"].astype(str).str.strip().ne("")]
    if identified.empty or as_of is None:
        empty = pd.DataFrame(
            columns=[
                "customer_id",
                "recency_days",
                "frequency_orders",
                "monetary_sales",
                "segment",
            ]
        )
        return empty, {
            "as_of_date": as_of,
            "excluded_missing_customer_rows": int(len(data) - len(identified)),
            "segment_rules": list(RFM_SEGMENT_RULES),
        }

    customers = (
        identified.groupby("customer_id", observed=True)
        .agg(
            last_purchase=("order_date", "max"),
            frequency_orders=("order_id", "nunique"),
            monetary_sales=("sales", "sum"),
        )
        .reset_index()
    )
    customers["recency_days"] = (
        as_of - customers["last_purchase"].dt.normalize()
    ).dt.days
    recency_25 = float(customers["recency_days"].quantile(0.25))
    recency_50 = float(customers["recency_days"].median())
    recency_75 = float(customers["recency_days"].quantile(0.75))
    frequency_50 = float(customers["frequency_orders"].quantile(0.50))
    frequency_75 = float(customers["frequency_orders"].quantile(0.75))
    monetary_75 = float(customers["monetary_sales"].quantile(0.75))
    high_value_active = (
        customers["recency_days"].le(recency_25)
        & customers["frequency_orders"].ge(frequency_75)
        & customers["monetary_sales"].ge(monetary_75)
    )
    frequent_active = (
        ~high_value_active
        & customers["recency_days"].le(recency_50)
        & customers["frequency_orders"].ge(frequency_75)
    )
    at_risk = (
        ~high_value_active
        & ~frequent_active
        & customers["recency_days"].ge(recency_75)
        & customers["frequency_orders"].le(frequency_50)
    )
    customers["segment"] = np.select(
        [high_value_active, frequent_active, at_risk],
        ["High-value active", "Frequent active", "At-risk, lower activity"],
        default="Other",
    )
    customers = customers.drop(columns="last_purchase").sort_values(
        ["monetary_sales", "customer_id"], ascending=[False, True], kind="stable"
    )
    return customers.reset_index(drop=True), {
        "as_of_date": as_of,
        "excluded_missing_customer_rows": int(len(data) - len(identified)),
        "segment_rules": list(RFM_SEGMENT_RULES),
        "cutoffs": {
            "recency_days_25th_percentile": recency_25,
            "recency_days_median": recency_50,
            "recency_days_75th_percentile": recency_75,
            "frequency_orders_median": frequency_50,
            "frequency_orders_75th_percentile": frequency_75,
            "monetary_sales_75th_percentile": monetary_75,
        },
    }


def score_profit_anomalies(
    data: pd.DataFrame,
    minimum_rows: int = 10,
    unusual_score: float = -2.5,
) -> pd.DataFrame:
    """Score region/category/sub-category/discount margins against peer cells."""
    _validate(data)
    dimensions = ["region", "category", "sub_category", "discount"]
    cells = (
        data.groupby(dimensions, dropna=False, observed=True)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            row_count=("profit", "size"),
        )
        .reset_index()
    )
    cells["profit_margin_pct"] = _margin(cells["profit"], cells["sales"])
    cells["peer_median_margin_pct"] = np.nan
    cells["peer_mad_margin_pct"] = np.nan
    cells["robust_deviation_score"] = np.nan
    cells["unusual_low_margin"] = False
    for _, indexes in cells.groupby(["category", "sub_category"], dropna=False).groups.items():
        peer_cells = cells.loc[indexes]
        for index in indexes:
            peers = peer_cells.loc[peer_cells.index != index, "profit_margin_pct"].dropna()
            if len(peers) < 3 or cells.loc[index, "row_count"] < minimum_rows:
                continue
            baseline = float(peers.median())
            mad = float((peers - baseline).abs().median())
            cells.loc[index, "peer_median_margin_pct"] = baseline
            cells.loc[index, "peer_mad_margin_pct"] = mad
            if mad > 0 and cells.loc[index, "sales"] != 0:
                score = (
                    cells.loc[index, "profit_margin_pct"] - baseline
                ) / (1.4826 * mad)
                cells.loc[index, "robust_deviation_score"] = score
                cells.loc[index, "unusual_low_margin"] = (
                    score <= unusual_score
                )
    return cells


def find_profit_leaks(data: pd.DataFrame, minimum_group_rows: int = 20) -> pd.DataFrame:
    """Return evidence-backed product, region/category, and discount investigations."""
    _validate(data)
    finding_columns = [
        "finding_type",
        "subject",
        "sales",
        "profit",
        "margin_pct",
        "baseline_margin_pct",
        "margin_gap_pp",
        "quantity",
        "row_count",
        "order_count",
        "financial_impact",
        "impact_basis",
        "explanation",
        "next_question",
    ]
    findings: list[dict[str, Any]] = []

    products = (
        data.groupby(
            ["category", "product_id", "product_name"],
            dropna=False,
            observed=True,
        )
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            row_count=("sales", "size"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    total_negative = float(products.loc[products["profit"] < 0, "profit"].sum())
    high_sales_negative_ids: set[tuple[Any, Any]] = set()
    category_totals = (
        data.groupby("category", dropna=False, observed=True)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"))
    )
    for category, group in products.groupby("category", dropna=False, observed=True):
        category_total = category_totals.loc[category]
        sales_median = float(group["sales"].median())
        for row in group.itertuples(index=False):
            peer_sales = float(category_total["sales"] - row.sales)
            peer_profit = float(category_total["profit"] - row.profit)
            if (
                row.row_count < 2
                or row.sales < sales_median
                or row.sales <= 0
                or peer_sales <= 0
            ):
                continue
            baseline = peer_profit / peer_sales * 100
            margin = row.profit / row.sales * 100
            if margin >= baseline:
                continue
            loss = row.profit < 0
            if loss:
                high_sales_negative_ids.add((category, row.product_id))
            impact = float(row.profit) if loss else float(row.sales * (baseline - margin) / 100)
            findings.append(
                {
                    "finding_type": "High-sales negative-profit product"
                    if loss
                    else "High-sales product below category margin",
                    "subject": f"{row.product_name} ({row.product_id})",
                    "sales": float(row.sales),
                    "profit": float(row.profit),
                    "margin_pct": margin,
                    "baseline_margin_pct": baseline,
                    "margin_gap_pp": margin - baseline,
                    "quantity": int(row.quantity),
                    "row_count": int(row.row_count),
                    "order_count": int(row.order_count),
                    "financial_impact": impact,
                    "impact_basis": (
                        f"Observed negative profit; {abs(row.profit / total_negative * 100):.1f}% "
                        "of negative-profit product total"
                    )
                    if loss and total_negative < 0
                    else "Observed negative profit"
                    if loss
                    else "Gap to category margin; not recoverable savings",
                    "explanation": "High-sales product trails the filtered "
                    "sales-weighted margin of other products in its category."
                    if not loss
                    else "Product has negative aggregate profit despite material sales.",
                    "next_question": "Are discount, fulfillment cost, or product mix "
                    "contributing to the observed margin gap?",
                }
            )

    region_category = (
        data.groupby(["region", "category"], dropna=False, observed=True)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            row_count=("sales", "size"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    for category, group in region_category.groupby("category", dropna=False):
        for row in group.itertuples(index=False):
            if row.row_count < minimum_group_rows or row.sales <= 0:
                continue
            peers = group.loc[group["region"] != row.region]
            peer_sales = float(peers["sales"].sum())
            if peers.empty or peer_sales <= 0:
                continue
            baseline = float(peers["profit"].sum() / peer_sales * 100)
            margin = float(row.profit / row.sales * 100)
            if margin >= baseline:
                continue
            findings.append(
                {
                    "finding_type": "Region/category margin below peer regions",
                    "subject": f"{row.region} / {category}",
                    "sales": float(row.sales),
                    "profit": float(row.profit),
                    "margin_pct": margin,
                    "baseline_margin_pct": baseline,
                    "margin_gap_pp": margin - baseline,
                    "quantity": int(row.quantity),
                    "row_count": int(row.row_count),
                    "order_count": int(row.order_count),
                    "financial_impact": float(row.sales * (baseline - margin) / 100),
                    "impact_basis": "Gap to other regions' category margin; "
                    "not recoverable savings",
                    "explanation": f"Margin is below the other regions' weighted "
                    f"{category} margin, based on {row.row_count} order lines.",
                    "next_question": "Do local pricing, product mix, or operating "
                    "conditions explain the regional difference?",
                }
            )

    discount_groups = (
        data.groupby("discount", dropna=False, observed=True)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            row_count=("sales", "size"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    total_sales = float(data["sales"].sum())
    for row in discount_groups.itertuples(index=False):
        peer_sales = total_sales - float(row.sales)
        peer_profit = float(data["profit"].sum()) - float(row.profit)
        if row.row_count < minimum_group_rows or row.sales <= 0 or peer_sales <= 0:
            continue
        baseline_discount_margin = peer_profit / peer_sales * 100
        margin = float(row.profit / row.sales * 100)
        if margin >= baseline_discount_margin:
            continue
        findings.append(
            {
                "finding_type": "Discount level associated with lower margin",
                "subject": f"{row.discount:.0%} observed discount",
                "sales": float(row.sales),
                "profit": float(row.profit),
                "margin_pct": margin,
                "baseline_margin_pct": baseline_discount_margin,
                "margin_gap_pp": margin - baseline_discount_margin,
                "quantity": int(row.quantity),
                "row_count": int(row.row_count),
                "order_count": int(row.order_count),
                "financial_impact": float(row.sales * (baseline_discount_margin - margin) / 100),
                "impact_basis": "Observed margin gap; descriptive association, "
                "not causal impact",
                "explanation": f"Observed margin is below the weighted margin of "
                f"other discount levels across {row.row_count} order lines.",
                "next_question": "Which products and customer/order contexts occur "
                "at this discount level?",
            }
        )

    negative_products = products.loc[products["profit"] < 0].copy()
    total_negative = float(negative_products["profit"].sum())
    if total_negative < 0:
        for row in negative_products.loc[
            negative_products["row_count"] >= 2
        ].itertuples(index=False):
            if (row.category, row.product_id) in high_sales_negative_ids:
                continue
            margin = float(row.profit / row.sales * 100) if row.sales > 0 else np.nan
            findings.append(
                {
                    "finding_type": "Product contributing to negative-profit pool",
                    "subject": f"{row.product_name} ({row.product_id})",
                    "sales": float(row.sales),
                    "profit": float(row.profit),
                    "margin_pct": margin,
                    "baseline_margin_pct": None,
                    "margin_gap_pp": None,
                    "quantity": int(row.quantity),
                    "row_count": int(row.row_count),
                    "order_count": int(row.order_count),
                    "financial_impact": float(row.profit),
                    "impact_basis": "Observed negative profit; "
                    f"{abs(row.profit / total_negative * 100):.1f}% of "
                    "negative-profit product total",
                    "explanation": "Aggregate product profit is negative; "
                    "this is an observed loss, not by itself a statistical anomaly.",
                    "next_question": "Which order lines, discounts, or cost assumptions "
                    "should be reviewed for this product?",
                }
            )

    result = pd.DataFrame(findings, columns=finding_columns)
    if not result.empty:
        result["priority_order"] = np.where(result["profit"] < 0, 0, 1)
        result["loss_rank"] = result["profit"].where(result["profit"] < 0, 0)
        result["gap_rank"] = -result["financial_impact"].where(result["profit"] >= 0, 0)
        result = result.sort_values(
            ["priority_order", "loss_rank", "gap_rank", "sales"],
            ascending=[True, True, True, False],
            kind="stable",
        ).drop(columns=["priority_order", "loss_rank", "gap_rank"])
    return result.reset_index(drop=True)


def estimate_discount_scenario(
    data: pd.DataFrame,
    scenario_discount: float,
) -> dict[str, float | int | str]:
    """Estimate a uniform discount change under fixed volume and cost assumptions."""
    _validate(data)
    if not 0 <= scenario_discount < 1:
        raise ValueError("Scenario discount must be at least 0 and less than 1.")
    if data.empty:
        raise ValueError("No historical records are available for this scenario.")
    if data[["sales", "profit", "discount"]].isna().any().any():
        raise ValueError("Scenario requires non-missing sales, profit, and discount.")
    if (data["discount"] < 0).any() or (data["discount"] >= 1).any():
        raise ValueError("Historical discounts must be between 0 inclusive and 1 exclusive.")
    if (data["sales"] < 0).any():
        raise ValueError("Negative sales prevent a credible fixed-volume scenario.")

    gross_sales = float((data["sales"] / (1 - data["discount"])).sum())
    baseline_sales = float(data["sales"].sum())
    if gross_sales <= 0 or baseline_sales <= 0:
        raise ValueError(
            "Historical sales are zero, so the selected rows do not support a discount scenario."
        )
    baseline_profit = float(data["profit"].sum())
    scenario_sales = gross_sales * (1 - scenario_discount)
    scenario_profit = baseline_profit + (scenario_sales - baseline_sales)
    weighted_discount = (
        1 - baseline_sales / gross_sales if gross_sales > 0 else 0.0
    )
    return {
        "row_count": int(len(data)),
        "baseline_sales": baseline_sales,
        "baseline_profit": baseline_profit,
        "baseline_margin_pct": baseline_profit / baseline_sales * 100
        if baseline_sales != 0
        else float("nan"),
        "baseline_weighted_discount": weighted_discount,
        "scenario_discount": float(scenario_discount),
        "scenario_sales_estimate": scenario_sales,
        "scenario_profit_estimate": scenario_profit,
        "profit_difference_estimate": scenario_profit - baseline_profit,
        "assumption": "Historical quantity, list prices, product mix, and unit costs "
        "remain unchanged; only the selected discount changes. This is a hypothetical "
        "mechanical estimate, not a forecast or causal estimate.",
    }
