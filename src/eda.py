"""Explore the cleaned Superstore dataset and write charts and a JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __package__:
    from .load_data import PROJECT_ROOT
else:
    from load_data import PROJECT_ROOT


DEFAULT_DATASET = Path("data") / "cleaned" / "superstore_cleaned.csv"
DEFAULT_CHART_DIRECTORY = Path("reports") / "eda"
DEFAULT_SUMMARY = Path("reports") / "eda_summary.json"
DATE_FORMAT = "%Y-%m-%d"
NUMERIC_COLUMNS = ("sales", "profit", "quantity", "discount")
REQUIRED_COLUMNS = (
    "order_id",
    "order_date",
    "ship_date",
    "ship_mode",
    "customer_id",
    "segment",
    "region",
    "product_id",
    "product_name",
    "category",
    "sub_category",
    *NUMERIC_COLUMNS,
)
GROUPINGS = {
    "region": ["region"],
    "category": ["category"],
    "sub_category": ["category", "sub_category"],
    "customer_segment": ["segment"],
    "shipping_mode": ["ship_mode"],
}
PRODUCT_LIMIT = 10


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime(DATE_FORMAT)
    return value


def validate_columns(data: pd.DataFrame) -> None:
    """Raise a useful error when the cleaned dataset lacks required fields."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate fields and parse dates/numeric values while recording failures."""
    validate_columns(data)
    prepared = data.copy()
    date_quality: dict[str, dict[str, int]] = {}
    for column in ("order_date", "ship_date"):
        original = prepared[column]
        parsed = pd.to_datetime(original, format=DATE_FORMAT, errors="coerce", exact=True)
        failures = original.notna() & parsed.isna()
        date_quality[column] = {
            "missing_count": int(original.isna().sum()),
            "parse_failure_count": int(failures.sum()),
        }
        prepared[column] = parsed

    numeric_quality: dict[str, dict[str, int]] = {}
    for column in NUMERIC_COLUMNS:
        original = prepared[column]
        numeric = pd.to_numeric(original, errors="coerce")
        numeric_quality[column] = {
            "missing_count": int(original.isna().sum()),
            "conversion_failure_count": int((original.notna() & numeric.isna()).sum()),
        }
        prepared[column] = numeric
    return prepared, {
        "date_format": DATE_FORMAT,
        "date_columns": date_quality,
        "numeric_columns": numeric_quality,
    }


def aggregate_metrics(data: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    """Aggregate sales/profit and transaction counts for one or more dimensions."""
    groups = [by] if isinstance(by, str) else list(by)
    return (
        data.groupby(groups, dropna=False, observed=True)
        .agg(
            sales=("sales", lambda values: values.sum(min_count=1)),
            profit=("profit", lambda values: values.sum(min_count=1)),
            quantity=("quantity", lambda values: values.sum(min_count=1)),
            order_count=("order_id", "nunique"),
            row_count=("order_id", "size"),
        )
        .reset_index()
        .sort_values(groups, kind="stable")
        .reset_index(drop=True)
    )


def _ranked_products(data: pd.DataFrame, metric: str, ascending: bool) -> list[dict[str, Any]]:
    ranked = (
        data.groupby(["product_id", "product_name"], dropna=False, observed=True)
        .agg(
            sales=("sales", lambda values: values.sum(min_count=1)),
            profit=("profit", lambda values: values.sum(min_count=1)),
            quantity=("quantity", lambda values: values.sum(min_count=1)),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values([metric, "product_id"], ascending=[ascending, True], kind="stable")
    )
    return _json_safe(ranked.head(PRODUCT_LIMIT).to_dict(orient="records"))


def _descriptive_statistics(data: pd.DataFrame) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for column in NUMERIC_COLUMNS:
        values = data[column].dropna()
        statistics[column] = {
            "unit": "dataset monetary units" if column in ("sales", "profit") else "units"
            if column == "quantity"
            else "discount proportion (0 to 1)",
            "valid_count": int(values.count()),
            "missing_count": int(data[column].isna().sum()),
            "mean": _json_safe(values.mean()) if len(values) else None,
            "standard_deviation": _json_safe(values.std()) if len(values) > 1 else None,
            "minimum": _json_safe(values.min()) if len(values) else None,
            "25th_percentile": _json_safe(values.quantile(0.25)) if len(values) else None,
            "median": _json_safe(values.median()) if len(values) else None,
            "75th_percentile": _json_safe(values.quantile(0.75)) if len(values) else None,
            "maximum": _json_safe(values.max()) if len(values) else None,
            "total": _json_safe(values.sum()) if len(values) else None,
        }
    return statistics


def _discount_analysis(data: pd.DataFrame) -> dict[str, Any]:
    valid = data[["discount", "profit"]].dropna()
    grouped = (
        valid.groupby("discount", observed=True)
        .agg(
            row_count=("profit", "size"),
            mean_profit=("profit", "mean"),
            median_profit=("profit", "median"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values("discount")
    )
    pearson = valid["discount"].corr(valid["profit"], method="pearson")
    ranked_discount = valid["discount"].rank(method="average")
    ranked_profit = valid["profit"].rank(method="average")
    spearman = ranked_discount.corr(ranked_profit, method="pearson")
    return {
        "observation_count": int(len(valid)),
        "pearson_correlation": _json_safe(pearson),
        "spearman_correlation": _json_safe(spearman),
        "by_discount_rate": _json_safe(grouped.to_dict(orient="records")),
        "interpretation": (
            "These are observational associations at the order-line level. "
            "They do not establish that discounts cause changes in profit; "
            "product mix, order characteristics, and other factors may differ."
        ),
    }


def analyze_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    """Calculate actual-data summary metrics and chart-ready aggregation tables."""
    prepared, quality = prepare_data(data)
    valid_dates = prepared.loc[prepared["order_date"].notna()].copy()
    if valid_dates.empty:
        raise ValueError("No valid order_date values are available for EDA.")

    valid_dates["year"] = valid_dates["order_date"].dt.year
    valid_dates["month"] = valid_dates["order_date"].dt.to_period("M").astype(str)
    tables: dict[str, pd.DataFrame] = {
        "year": aggregate_metrics(valid_dates, "year"),
        "month": aggregate_metrics(valid_dates, "month"),
    }
    for name, columns in GROUPINGS.items():
        tables[name] = aggregate_metrics(prepared, columns)

    products_by_sales = {
        "highest": _ranked_products(prepared, "sales", ascending=False),
        "lowest": _ranked_products(prepared, "sales", ascending=True),
    }
    products_by_profit = {
        "highest": _ranked_products(prepared, "profit", ascending=False),
        "lowest": _ranked_products(prepared, "profit", ascending=True),
    }
    negative_profit_products = [
        product
        for product in products_by_profit["lowest"]
        if product["profit"] is not None and product["profit"] < 0
    ]
    category_performance = tables["category"]
    category_rankings = {
        f"{direction}_{metric}": _json_safe(
            category_performance.sort_values(
                [metric, "category"],
                ascending=[ascending, True],
                kind="stable",
            )
            .head(len(category_performance))
            .to_dict(orient="records")
        )
        for direction, ascending in (("highest", False), ("lowest", True))
        for metric in ("sales", "profit")
    }
    temporal = {
        "by_year": _json_safe(tables["year"].to_dict(orient="records")),
        "by_month": _json_safe(tables["month"].to_dict(orient="records")),
    }
    summary = {
        "dataset": {
            "row_count": int(len(prepared)),
            "column_count": int(len(prepared.columns)),
            "order_date_period": {
                "start": valid_dates["order_date"].min().strftime(DATE_FORMAT),
                "end": valid_dates["order_date"].max().strftime(DATE_FORMAT),
            },
            "monetary_unit": (
                "Dataset monetary units; the dataset does not specify a currency symbol."
            ),
        },
        "data_quality": quality,
        "descriptive_statistics": _descriptive_statistics(prepared),
        "temporal_trends": temporal,
        "sales_profit_by_region": _json_safe(tables["region"].to_dict(orient="records")),
        "sales_profit_by_category": _json_safe(
            tables["category"].to_dict(orient="records")
        ),
        "category_rankings": category_rankings,
        "sales_profit_by_sub_category": _json_safe(
            tables["sub_category"].to_dict(orient="records")
        ),
        "sales_profit_by_customer_segment": _json_safe(
            tables["customer_segment"].to_dict(orient="records")
        ),
        "sales_profit_by_shipping_mode": _json_safe(
            tables["shipping_mode"].to_dict(orient="records")
        ),
        "products": {
            "highest_sales": products_by_sales["highest"],
            "lowest_sales": products_by_sales["lowest"],
            "highest_profit": products_by_profit["highest"],
            "lowest_profit": products_by_profit["lowest"],
            "lowest_profit_negative_products": negative_profit_products,
            "ranking_note": (
                "Sales rankings use summed sales; profit rankings use summed profit. "
                "A low-sales product is not necessarily loss-making. Negative profit "
                "is shown separately and is not removed."
            ),
        },
        "discount_profit_relationship": _discount_analysis(prepared),
        "order_trends": {
            "distinct_order_count": int(prepared["order_id"].nunique()),
            "monthly_distinct_orders": _json_safe(
                valid_dates.groupby("month", observed=True)["order_id"]
                .nunique()
                .rename("order_count")
                .reset_index()
                .to_dict(orient="records")
            ),
            "note": (
                "Order counts are distinct Order ID values assigned to the order date month."
            ),
        },
        "analytical_limitations": [
            "The analysis describes this dataset only; it does not establish representativeness beyond the records provided.",
            "Sales and profit use dataset monetary units because no currency symbol or currency metadata is supplied.",
            "Discount-profit relationships are observational and do not establish causation.",
            "Product performance is aggregated by product ID and product name; rankings do not adjust for exposure, availability, or time on sale.",
            "Invalid order dates are excluded from time-based analyses and counted in data_quality; invalid numeric values are excluded from the relevant pandas sums/statistics and counted in data_quality.",
            "Potential duplicate order lines are retained; aggregation counts every input row.",
        ],
    }
    return prepared, summary, tables


def _make_charts(
    data: pd.DataFrame, tables: dict[str, pd.DataFrame], chart_directory: Path
) -> list[str]:
    chart_directory.mkdir(parents=True, exist_ok=True)
    charts: list[tuple[str, go.Figure]] = []

    monthly = tables["month"]
    trends = make_subplots(specs=[[{"secondary_y": True}]])
    trends.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["sales"], name="Sales", mode="lines+markers"),
        secondary_y=False,
    )
    trends.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["profit"], name="Profit", mode="lines+markers"),
        secondary_y=True,
    )
    trends.update_layout(title="Monthly Sales and Profit", xaxis_title="Order month")
    trends.update_yaxes(title_text="Sales (dataset monetary units)", secondary_y=False)
    trends.update_yaxes(title_text="Profit (dataset monetary units)", secondary_y=True)
    charts.append(("monthly_sales_profit.html", trends))

    charts.append(
        (
            "regional_sales_profit.html",
            px.bar(
                tables["region"].melt(
                    id_vars="region",
                    value_vars=["sales", "profit"],
                    var_name="metric",
                    value_name="amount",
                ),
                x="region",
                y="amount",
                color="metric",
                barmode="group",
                title="Sales and Profit by Region",
                labels={"amount": "Dataset monetary units", "region": "Region"},
            ),
        )
    )
    category_data = tables["sub_category"].copy()
    category_data["category_sub_category"] = (
        category_data["category"] + " / " + category_data["sub_category"]
    )
    charts.append(
        (
            "category_subcategory_performance.html",
            px.bar(
                category_data.melt(
                    id_vars=["category", "sub_category", "category_sub_category"],
                    value_vars=["sales", "profit"],
                    var_name="metric",
                    value_name="amount",
                ),
                x="amount",
                y="category_sub_category",
                color="metric",
                barmode="group",
                orientation="h",
                title="Sales and Profit by Product Sub-Category",
                labels={"amount": "Dataset monetary units", "category_sub_category": "Category / sub-category"},
            ),
        )
    )
    charts.append(
        (
            "customer_segment_performance.html",
            px.bar(
                tables["customer_segment"].melt(
                    id_vars="segment",
                    value_vars=["sales", "profit"],
                    var_name="metric",
                    value_name="amount",
                ),
                x="segment",
                y="amount",
                color="metric",
                barmode="group",
                title="Sales and Profit by Customer Segment",
                labels={"amount": "Dataset monetary units", "segment": "Customer segment"},
            ),
        )
    )
    charts.append(
        (
            "discount_profit_relationship.html",
            px.scatter(
                data,
                x="discount",
                y="profit",
                color="category",
                hover_data=["product_name", "sales", "quantity"],
                title="Order-Line Discount and Profit (Association, Not Causation)",
                labels={
                    "discount": "Discount proportion",
                    "profit": "Profit (dataset monetary units)",
                },
                opacity=0.55,
            ),
        )
    )
    charts.append(
        (
            "shipping_mode_performance.html",
            px.bar(
                tables["shipping_mode"].melt(
                    id_vars="ship_mode",
                    value_vars=["sales", "profit"],
                    var_name="metric",
                    value_name="value",
                ),
                x="ship_mode",
                y="value",
                color="metric",
                barmode="group",
                title="Sales and Profit by Shipping Mode",
                labels={
                    "ship_mode": "Shipping mode",
                    "value": "Dataset monetary units",
                },
            ),
        )
    )
    charts.append(
        (
            "shipping_mode_orders.html",
            px.bar(
                tables["shipping_mode"],
                x="ship_mode",
                y="order_count",
                title="Distinct Orders by Shipping Mode",
                labels={"ship_mode": "Shipping mode", "order_count": "Distinct orders"},
            ),
        )
    )

    filenames: list[str] = []
    for filename, figure in charts:
        figure.write_html(
            chart_directory / filename,
            include_plotlyjs="cdn",
            full_html=True,
        )
        filenames.append(filename)
    return filenames


def run_eda(
    dataset_path: Path = DEFAULT_DATASET,
    chart_directory: Path = DEFAULT_CHART_DIRECTORY,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    """Analyze the cleaned dataset, writing HTML charts and a JSON summary."""
    dataset = _resolve(dataset_path)
    charts_path = _resolve(chart_directory)
    summary_file = _resolve(summary_path)
    data = pd.read_csv(dataset)
    prepared, summary, tables = analyze_data(data)
    summary["charts"] = _make_charts(prepared, tables, charts_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explore the cleaned Superstore dataset and generate reports."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_DATASET,
        help="Cleaned CSV path (default: data/cleaned/superstore_cleaned.csv)",
    )
    args = parser.parse_args()
    try:
        summary = run_eda(args.file)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError) as error:
        print(f"Could not complete EDA: {error}", file=sys.stderr)
        return 1
    print(f"Dataset: {_resolve(args.file)}")
    print(f"Rows: {summary['dataset']['row_count']}")
    print(f"Columns: {summary['dataset']['column_count']}")
    print(f"Order date period: {summary['dataset']['order_date_period']}")
    print(f"Summary: {_resolve(DEFAULT_SUMMARY)}")
    print(f"Charts: {_resolve(DEFAULT_CHART_DIRECTORY)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
