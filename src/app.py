"""Streamlit dashboard for interactive RetailIQ business analytics."""

from __future__ import annotations

from html import escape
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

if __package__:
    from .advanced_analytics import (
        RFM_SEGMENT_RULES,
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
    from .build_database import DEFAULT_DATABASE, _resolve
    from .eda import aggregate_metrics
    from .load_data import PROJECT_ROOT
else:
    from advanced_analytics import (
        RFM_SEGMENT_RULES,
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
    from build_database import DEFAULT_DATABASE, _resolve
    from eda import aggregate_metrics
    from load_data import PROJECT_ROOT


REQUIRED_COLUMNS = {
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
    "sales",
    "profit",
    "quantity",
    "discount",
}
NUMERIC_COLUMNS = ("sales", "profit", "quantity", "discount")
FILTER_COLUMNS = {
    "Region": "region",
    "Category": "category",
    "Customer segment": "segment",
    "Ship mode": "ship_mode",
}
PALETTE = {
    "navy": "#0B1727",
    "blue": "#73A7FF",
    "cyan": "#78E5F2",
    "teal": "#49D5BD",
    "violet": "#B7A0FF",
    "gold": "#F5C56B",
    "red": "#FF8294",
    "muted": "#9CAFC5",
}
THEME_PATH = Path(__file__).resolve().with_name("dashboard.css")


class DatabaseLoadError(RuntimeError):
    """Raised when the dashboard cannot safely read its SQLite database."""


def load_dashboard_data(database_path: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Read required business fields from the existing database in read-only mode."""
    database = _resolve(database_path)
    if not database.is_file():
        raise DatabaseLoadError(
            f"RetailIQ database was not found at {database}. "
            "Build it with: .\\.venv\\Scripts\\python.exe .\\src\\build_database.py"
        )

    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "retail_sales" not in tables:
                raise DatabaseLoadError(
                    f"Database {database} does not contain the retail_sales table. "
                    "Rebuild it with src/build_database.py."
                )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(retail_sales)")
            }
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise DatabaseLoadError(
                    "Database table retail_sales is missing required columns: "
                    + ", ".join(missing)
                )
            selected_columns = ", ".join(f'"{name}"' for name in sorted(REQUIRED_COLUMNS))
            data = pd.read_sql_query(
                f"SELECT {selected_columns} FROM retail_sales",
                connection,
            )
    except DatabaseLoadError:
        raise
    except sqlite3.Error as error:
        raise DatabaseLoadError(
            f"Could not read RetailIQ database {database}: {error}"
        ) from error

    for column in ("order_date", "ship_date"):
        parsed = pd.to_datetime(data[column], format="%Y-%m-%d", errors="coerce")
        if data[column].notna().ne(parsed.notna()).any():
            raise DatabaseLoadError(
                f"Database contains invalid {column} values; rebuild it from the cleaned CSV."
            )
        data[column] = parsed

    for column in NUMERIC_COLUMNS:
        original = data[column]
        parsed = pd.to_numeric(original, errors="coerce")
        if original.notna().ne(parsed.notna()).any():
            raise DatabaseLoadError(
                f"Database contains invalid numeric values in {column}; "
                "rebuild it from the cleaned CSV."
            )
        data[column] = parsed
    return data


def filter_sales_data(
    data: pd.DataFrame,
    date_range: tuple[date | pd.Timestamp, date | pd.Timestamp] | None = None,
    selections: dict[str, list[str] | tuple[str, ...] | set[str]] | None = None,
) -> pd.DataFrame:
    """Apply inclusive date bounds and selected dimension values consistently."""
    filtered = data.copy()
    if date_range is not None:
        start, end = (pd.Timestamp(value) for value in date_range)
        filtered = filtered.loc[
            filtered["order_date"].between(start, end, inclusive="both")
        ]
    for column, values in (selections or {}).items():
        if column not in FILTER_COLUMNS.values():
            raise ValueError(f"Unsupported dashboard filter column: {column}")
        filtered = filtered.loc[filtered[column].isin(values)]
    return filtered.copy()


def calculate_kpis(data: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate filtered dashboard KPIs, returning no margin when sales are zero."""
    sales = float(data["sales"].sum(min_count=1)) if not data.empty else 0.0
    profit = float(data["profit"].sum(min_count=1)) if not data.empty else 0.0
    quantity = float(data["quantity"].sum(min_count=1)) if not data.empty else 0.0
    return {
        "total_sales": sales,
        "total_profit": profit,
        "profit_margin_pct": profit / sales * 100.0 if sales != 0 else None,
        "distinct_orders": int(data["order_id"].nunique()),
        "quantity_sold": int(quantity),
    }


def _group_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return aggregate_metrics(data, column)


def _product_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["product_id", "product_name"], dropna=False, observed=True)
        .agg(
            sales=("sales", lambda values: values.sum(min_count=1)),
            profit=("profit", lambda values: values.sum(min_count=1)),
            quantity=("quantity", lambda values: values.sum(min_count=1)),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )


def build_business_insights(data: pd.DataFrame) -> list[str]:
    """Create concise insights supported by the current filtered data."""
    if data.empty:
        return ["No records match the selected filters. Widen the filters to view insights."]

    insights: list[str] = []
    kpis = calculate_kpis(data)
    regions = _group_summary(data, "region")
    if not regions.empty:
        leader = regions.sort_values("profit", ascending=False).iloc[0]
        insights.append(
            f"{leader['region']} leads filtered regional profit at "
            f"{leader['profit']:,.2f} dataset units from "
            f"{leader['sales']:,.2f} in sales."
        )

    categories = _group_summary(data, "category")
    if not categories.empty:
        categories["profit_margin_pct"] = np.where(
            categories["sales"] != 0,
            categories["profit"] / categories["sales"] * 100,
            np.nan,
        )
        category = categories.sort_values(
            ["profit_margin_pct", "category"], ascending=[False, True]
        ).iloc[0]
        if pd.notna(category["profit_margin_pct"]):
            insights.append(
                f"{category['category']} has the highest filtered category margin "
                f"at {category['profit_margin_pct']:.2f}% "
                f"({category['profit']:,.2f} profit / {category['sales']:,.2f} sales)."
            )

    products = _product_summary(data)
    negative = products.loc[products["profit"] < 0].sort_values("profit")
    if not negative.empty:
        insights.append(
            f"{len(negative)} products have negative total profit in this selection; "
            f"their combined product-level losses are {negative['profit'].sum():,.2f} "
            "dataset units."
        )
    else:
        insights.append("No products have negative total profit in this selection.")

    if data["discount"].nunique() > 1 and len(data) > 1:
        correlation = data["discount"].corr(data["profit"])
        if pd.notna(correlation):
            insights.append(
                f"Order-line discount and profit have a Pearson correlation of "
                f"{correlation:.3f} in this selection. This describes association, "
                "not causation."
            )
    insights.append(
        f"The selected records contain {kpis['distinct_orders']:,} distinct orders "
        f"and {kpis['quantity_sold']:,} units sold."
    )
    return insights


def _empty_chart(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title)
    figure.add_annotation(
        text="No data for the selected filters",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": PALETTE["muted"]},
    )
    return _apply_chart_theme(figure)


def _apply_chart_theme(figure: go.Figure) -> go.Figure:
    """Apply a shared, transparent Plotly theme that blends with glass panels."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#DCEBFA", "size": 12},
        title={
            "font": {"size": 14, "color": "#F0F7FF"},
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.15,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#C1D0E1", "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
        height=390,
        margin={"l": 18, "r": 18, "t": 96, "b": 28},
        hoverlabel={
            "bgcolor": "#14263C",
            "bordercolor": "rgba(120,229,242,.35)",
            "font": {"color": "#F2F8FF", "family": "Inter, Segoe UI, sans-serif"},
        },
        colorway=[
            PALETTE["cyan"],
            PALETTE["teal"],
            PALETTE["blue"],
            PALETTE["violet"],
            PALETTE["gold"],
            PALETTE["red"],
        ],
    )
    figure.update_xaxes(
        gridcolor="rgba(166,194,222,.10)",
        linecolor="rgba(166,194,222,.18)",
        tickfont={"color": "#AFC2D7", "size": 10},
        title_font={"color": "#C1D0E1", "size": 11},
        zerolinecolor="rgba(166,194,222,.16)",
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor="rgba(166,194,222,.10)",
        linecolor="rgba(166,194,222,.18)",
        tickfont={"color": "#AFC2D7", "size": 10},
        title_font={"color": "#C1D0E1", "size": 11},
        zerolinecolor="rgba(166,194,222,.16)",
        automargin=True,
    )
    return figure


def build_dashboard_charts(data: pd.DataFrame) -> dict[str, go.Figure]:
    """Build Plotly charts for all required business perspectives."""
    if data.empty:
        return {
            name: _empty_chart(title)
            for name, title in (
                ("monthly", "Monthly Sales and Profit"),
                ("category", "Sales and Profit by Category / Sub-Category"),
                ("region", "Regional Performance"),
                ("segment", "Customer Segment Performance"),
                ("ship_mode", "Shipping Mode Performance"),
                ("top_sales", "Top Products by Sales"),
                ("top_profit", "Top Products by Profit"),
                ("negative_profit", "Negative-Profit Products"),
                ("discount", "Discount versus Profitability"),
            )
        }

    charts: dict[str, go.Figure] = {}
    monthly = data.assign(month=data["order_date"].dt.to_period("M").astype(str))
    monthly_summary = (
        monthly.groupby("month", observed=True)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    charts["monthly"] = px.line(
        monthly_summary,
        x="month",
        y=["sales", "profit"],
        markers=True,
        title="Monthly Sales and Profit",
        labels={"month": "Order month", "value": "Dataset monetary units", "variable": "Metric"},
        color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"]],
    )
    charts["monthly"].update_traces(line={"shape": "spline", "width": 2.5}, marker={"size": 5})

    category = aggregate_metrics(data, ["category", "sub_category"])
    category["category_sub_category"] = (
        category["category"] + " / " + category["sub_category"]
    )
    charts["category"] = px.bar(
        category.melt(
            id_vars=["category", "category_sub_category"],
            value_vars=["sales", "profit"],
            var_name="metric",
            value_name="amount",
        ),
        x="amount",
        y="category_sub_category",
        color="metric",
        barmode="group",
        orientation="h",
        title="Sales and Profit by Category / Sub-Category",
        labels={
            "amount": "Dataset monetary units",
            "category_sub_category": "Category / sub-category",
        },
        color_discrete_sequence=[PALETTE["cyan"], PALETTE["violet"]],
    )

    charts["region"] = px.bar(
        _group_summary(data, "region").melt(
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
        labels={"amount": "Dataset monetary units"},
        color_discrete_sequence=[PALETTE["cyan"], PALETTE["teal"]],
    )
    charts["segment"] = px.bar(
        _group_summary(data, "segment").melt(
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
        labels={"segment": "Customer segment", "amount": "Dataset monetary units"},
        color_discrete_sequence=[PALETTE["cyan"], PALETTE["violet"]],
    )
    charts["ship_mode"] = px.bar(
        _group_summary(data, "ship_mode").melt(
            id_vars="ship_mode",
            value_vars=["sales", "profit"],
            var_name="metric",
            value_name="amount",
        ),
        x="ship_mode",
        y="amount",
        color="metric",
        barmode="group",
        title="Sales and Profit by Shipping Mode",
        labels={"ship_mode": "Shipping mode", "amount": "Dataset monetary units"},
        color_discrete_sequence=[PALETTE["cyan"], PALETTE["teal"]],
    )

    products = _product_summary(data)
    top_sales = products.nlargest(10, "sales").sort_values("sales")
    top_profit = products.nlargest(10, "profit").sort_values("profit")
    negative = products.loc[products["profit"] < 0].nsmallest(10, "profit").sort_values("profit")
    charts["top_sales"] = px.bar(
        top_sales,
        x="sales",
        y="product_name",
        orientation="h",
        title="Top Products by Sales",
        labels={"sales": "Sales (dataset monetary units)", "product_name": "Product"},
        hover_data=["product_id", "profit", "order_count"],
        color_discrete_sequence=[PALETTE["blue"]],
    )
    charts["top_profit"] = px.bar(
        top_profit,
        x="profit",
        y="product_name",
        orientation="h",
        title="Top Products by Profit",
        labels={"profit": "Profit (dataset monetary units)", "product_name": "Product"},
        hover_data=["product_id", "sales", "order_count"],
        color_discrete_sequence=[PALETTE["teal"]],
    )
    charts["negative_profit"] = px.bar(
        negative,
        x="profit",
        y="product_name",
        orientation="h",
        title="Products with Negative Total Profit (Worst 10)",
        labels={"profit": "Total profit (dataset monetary units)", "product_name": "Product"},
        hover_data=["product_id", "sales", "order_count"],
        color_discrete_sequence=[PALETTE["red"]],
    )
    charts["discount"] = px.scatter(
        data,
        x="discount",
        y="profit",
        color="category",
        hover_data=["product_name", "sales", "quantity", "region"],
        title="Discount versus Order-Line Profitability",
        labels={
            "discount": "Discount proportion",
            "profit": "Profit (dataset monetary units)",
        },
        opacity=0.55,
        color_discrete_sequence=[PALETTE["cyan"], PALETTE["teal"], PALETTE["violet"]],
    )
    for figure in charts.values():
        _apply_chart_theme(figure)
    return charts


def _format_amount(value: float) -> str:
    return f"{value:,.2f}"


def _render_chart(chart_id: str, figure: go.Figure) -> None:
    """Place a responsive interactive chart inside a reusable glass panel."""
    with st.container(key=f"chart-{chart_id}"):
        title = figure.layout.title.text or "Interactive retail analytics chart"
        st.plotly_chart(
            figure,
            width="stretch",
            config={"responsive": True},
            alt=f"Interactive visualization: {title}",
        )


def _render_section_heading(title: str, subtitle: str | None = None) -> None:
    """Render a consistent section title with optional explanatory context."""
    subtitle_markup = (
        f'<p class="section-subtitle">{escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f'<div class="section-title-group">'
        f'<div class="section-heading">{escape(title)}</div>'
        f"{subtitle_markup}</div>",
        unsafe_allow_html=True,
    )


def _mobile_data_card(
    title: str,
    identifier: str,
    metrics: tuple[tuple[str, str], ...],
) -> str:
    """Format one accessible summary card for narrow-screen table views."""
    metric_markup = "".join(
        f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
        for label, value in metrics
    )
    return (
        '<article class="mobile-data-card">'
        '<div class="mobile-data-card-heading">'
        f"<h4>{escape(title)}</h4><p>{escape(identifier)}</p>"
        "</div>"
        f'<dl class="mobile-data-card-metrics">{metric_markup}</dl>'
        "</article>"
    )


def _render_insight_card(index: int, insight: str) -> None:
    """Render data-derived insight copy as an accessible glass card."""
    labels = ("PERFORMANCE", "MARGIN", "PRODUCT HEALTH", "ORDER PATTERN", "SUMMARY")
    label = labels[min(index, len(labels) - 1)]
    with st.container(key=f"insight-card-{index}"):
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">{label}</p>
                <p class="insight-copy">{escape(insight)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_advanced_charts(
    data: pd.DataFrame,
    bubble_aggregation: str = "product",
) -> dict[str, go.Figure]:
    """Build advanced interactive figures from the currently filtered records."""
    monthly_sales, monthly_profit, region_category = aggregate_heatmaps(data)
    month_names = {
        month: pd.Timestamp(year=2000, month=month, day=1).strftime("%b")
        for month in range(1, 13)
    }
    sales_pivot = (
        monthly_sales.reindex(range(1, 13))
        .rename(index=month_names)
    )
    profit_pivot = (
        monthly_profit.reindex(range(1, 13))
        .rename(index=month_names)
    )
    margin_pivot = region_category.pivot(
        index="region", columns="category", values="profit_margin_pct"
    )
    figures: dict[str, go.Figure] = {
        "sales_heatmap": px.imshow(
            sales_pivot,
            labels={"x": "Category", "y": "Calendar month", "color": "Sales"},
            color_continuous_scale="Blues",
            text_auto=".2s",
            aspect="auto",
            title="Sales by calendar month and category",
        ),
        "profit_heatmap": px.imshow(
            profit_pivot,
            labels={"x": "Category", "y": "Calendar month", "color": "Profit"},
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".2s",
            aspect="auto",
            title="Profit by calendar month and category",
        ),
        "margin_heatmap": px.imshow(
            margin_pivot,
            labels={"x": "Category", "y": "Region", "color": "Margin (%)"},
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".1f",
            aspect="auto",
            title="Profit margin by region and category",
        ),
    }

    treemap, excluded_treemap_groups = aggregate_treemap(data)
    treemap["product_label"] = (
        treemap["product_name"].astype(str) + " (" + treemap["product_id"].astype(str) + ")"
    )
    if treemap.empty:
        figures["treemap"] = _empty_chart("Product hierarchy · no positive sales groups")
    else:
        figures["treemap"] = px.treemap(
            treemap,
            path=["category", "sub_category", "product_label"],
            values="sales",
            color="profit_margin_pct",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data={
                "sales": ":,.2f",
                "profit": ":,.2f",
                "profit_margin_pct": ":.2f",
                "quantity": True,
                "discount": ":.2f",
                "row_count": True,
                "product_label": False,
            },
            title="Product hierarchy · area = sales, color = profit margin",
        )
        figures["treemap"].update_traces(
            root_color="rgba(0,0,0,0)",
            tiling={"packing": "squarify"},
        )
    figures["treemap"].add_annotation(
        text=f"Product groups with non-positive sales excluded from area chart: {excluded_treemap_groups}",
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": PALETTE["muted"], "size": 10},
    )

    bubbles = aggregate_discount_bubbles(data, bubble_aggregation)
    bubble_hover_name = (
        "product_name" if bubble_aggregation == "product" else "sub_category"
    )
    bubble_hover_data = [
        column
        for column in (
            "product_id",
            "category",
            "sub_category",
            "sales",
            "profit",
            "quantity",
            "row_count",
            "discount",
        )
        if column in bubbles.columns
    ]
    figures["discount_bubbles"] = px.scatter(
        bubbles,
        x="discount",
        y="profit_margin_pct",
        size="sales",
        size_max=30,
        color="profit",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_name=bubble_hover_name,
        hover_data=bubble_hover_data,
        title=f"Discount and {bubble_aggregation.replace('_', ' ')} profitability",
        labels={
            "discount": "Observed discount proportion",
            "profit_margin_pct": "Profit margin (%)",
            "sales": "Sales (bubble area)",
            "profit": "Profit (color)",
        },
        opacity=0.72,
    )
    figures["discount_bubbles"].add_hline(
        y=0, line_dash="dot", line_color=PALETTE["muted"], opacity=0.7
    )

    pareto = calculate_product_pareto(data)
    pareto_chart = make_subplots(specs=[[{"secondary_y": True}]])
    pareto_chart.add_trace(
        go.Bar(
            x=pareto["rank"],
            y=pareto["sales"],
            name="Product sales",
            marker_color=PALETTE["blue"],
            customdata=pareto[
                ["product_id", "product_name", "sales_contribution_pct"]
            ].to_numpy(),
            hovertemplate=(
                "Rank: %{x}<br>%{customdata[1]}<br>Sales: %{y:,.2f}"
                "<br>Product ID: %{customdata[0]}"
                "<br>Sales share: %{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    pareto_chart.add_trace(
        go.Scatter(
            x=pareto["rank"],
            y=pareto["cumulative_sales_pct"],
            name="Cumulative sales share",
            mode="lines",
            line={"color": PALETTE["teal"], "width": 2.5},
            customdata=pareto[["product_name"]].to_numpy(),
            hovertemplate="Rank %{x}<br>%{customdata[0]}<br>"
            "Cumulative sales share: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    pareto_chart.add_hline(
        y=80,
        line_dash="dash",
        line_color=PALETTE["gold"],
        annotation_text="80% reference",
        annotation_position="top left",
        secondary_y=True,
    )
    pareto_chart.update_layout(
        title="Product sales Pareto · products ranked by sales",
        xaxis_title="Product rank (hover for product name)",
        yaxis_title="Sales (dataset units)",
        xaxis={"tickangle": -45, "tickfont": {"size": 8}},
    )
    pareto_chart.update_yaxes(title_text="Cumulative share (%)", secondary_y=True, range=[0, 105])
    figures["pareto"] = pareto_chart

    for dimension in ("category", "sub_category"):
        contributions = profit_contributions(data, dimension)
        if dimension == "sub_category":
            contributions["label"] = (
                contributions["category"].astype(str)
                + " / "
                + contributions["sub_category"].astype(str)
            )
        else:
            contributions["label"] = contributions["category"].astype(str)
        total_profit = float(data["profit"].sum())
        figure = go.Figure(
            go.Waterfall(
                name="Profit contribution",
                orientation="v",
                measure=["relative"] * len(contributions) + ["total"],
                x=contributions["label"].tolist() + ["Filtered total"],
                y=contributions["profit"].tolist() + [total_profit],
                text=[f"{value:,.0f}" for value in contributions["profit"]]
                + [f"{total_profit:,.0f}"],
                textposition="outside",
                increasing={"marker": {"color": PALETTE["teal"]}},
                decreasing={"marker": {"color": PALETTE["red"]}},
                totals={"marker": {"color": PALETTE["blue"]}},
                connector={"line": {"color": "rgba(166,194,222,.3)"}},
                customdata=contributions[["sales", "row_count"]].to_numpy().tolist()
                + [[float(data["sales"].sum()), len(data)]],
                hovertemplate=(
                    "%{x}<br>Profit contribution: %{y:,.2f}<br>"
                    "Sales: %{customdata[0]:,.2f}<br>"
                    "Order lines: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            title=f"Profit contribution by {dimension.replace('_', ' ')} · "
            f"filtered total {total_profit:,.2f}",
            yaxis_title="Profit (dataset monetary units)",
            xaxis_title=dimension.replace("_", " ").title(),
        )
        figures[f"waterfall_{dimension}"] = figure

    for column in ("sales", "profit", "discount", "quantity"):
        figure = px.box(
            data,
            y=column,
            points="outliers",
            title=f"{column.replace('_', ' ').title()} distribution",
            labels={column: column.replace("_", " ").title()},
            color_discrete_sequence=[PALETTE["cyan"]],
            hover_data=["category", "region"],
        )
        figures[f"outlier_{column}"] = figure

    rfm, rfm_info = calculate_customer_rfm(data)
    if not rfm.empty:
        rfm_counts = (
            rfm.groupby("segment", observed=True)
            .agg(customer_count=("customer_id", "size"), sales=("monetary_sales", "sum"))
            .reset_index()
        )
        figures["rfm_segments"] = px.bar(
            rfm_counts,
            x="segment",
            y="customer_count",
            color="segment",
            hover_data=["sales"],
            title="Filtered customer RFM segment counts",
            labels={"customer_count": "Customers", "segment": "Rule-based segment"},
            color_discrete_sequence=[
                PALETTE["teal"],
                PALETTE["cyan"],
                PALETTE["red"],
                PALETTE["violet"],
            ],
        )
    else:
        figures["rfm_segments"] = _empty_chart("Filtered customer RFM segment counts")

    for figure in figures.values():
        _apply_chart_theme(figure)
    return figures


def _render_advanced_analytics(data: pd.DataFrame) -> None:
    _render_section_heading(
        "Advanced analytics",
        "Explore seasonality, product mix, discount patterns, profit drivers, and customer value.",
    )
    if data.empty:
        st.info("No records match the active filters. Widen the filters to view advanced analyses.")
        return
    bubble_choice = st.selectbox(
        "Discount analysis detail",
        ["Product and observed discount", "Sub-category and observed discount"],
        key="discount-analysis-detail",
    )
    bubble_aggregation = (
        "product" if bubble_choice.startswith("Product") else "sub_category"
    )
    figures = _build_advanced_charts(data, bubble_aggregation)
    _render_section_heading(
        "Seasonality and product hierarchy",
        "Heatmaps pool calendar months across selected years. Treemap area shows positive product sales; "
        "color shows profit margin. Non-positive-sales groups are excluded.",
    )
    for row in (("sales_heatmap", "profit_heatmap"), ("margin_heatmap", "treemap")):
        left, right = st.columns(2)
        with left:
            _render_chart(row[0].replace("_", "-"), figures[row[0]])
        with right:
            _render_chart(row[1].replace("_", "-"), figures[row[1]])

    _render_section_heading(
        "Discount response and product concentration",
        "Compare observed discount-level profitability with product sales concentration; neither view implies causation.",
    )
    left, right = st.columns(2)
    with left:
        _render_chart("discount-bubbles", figures["discount_bubbles"])
        st.caption(
            f"Each bubble aggregates one {bubble_aggregation.replace('_', ' ')} at one observed discount level. "
            "Size represents sales; color represents profit. The view describes association, not causation."
        )
    with right:
        _render_chart("product-pareto", figures["pareto"])
        st.caption(
            "Sales contribution and cumulative share use all selected records. "
            "The 80% line is a comparison reference, not a guaranteed business rule."
        )

    _render_section_heading(
        "Profit contribution",
        "Waterfalls reconcile positive and negative group contributions to total filtered profit.",
    )
    left, right = st.columns(2)
    for column, dimension in ((left, "category"), (right, "sub_category")):
        with column:
            _render_chart(
                f"waterfall-{dimension}",
                figures[f"waterfall_{dimension}"],
            )
            contributions = profit_contributions(data, dimension)
            total = float(data["profit"].sum())
            if not np.isclose(contributions["profit"].sum(), total):
                st.error(f"The {dimension} profit contributions do not reconcile to filtered profit.")
            else:
                st.caption(f"Contribution check: {contributions['profit'].sum():,.2f} "
                           f"= filtered profit {total:,.2f}.")

    _render_section_heading(
        "Distribution analysis",
        "Box plots flag values beyond the 1.5 × IQR fences for investigation; no observations are removed.",
    )
    outliers = outlier_summary(data)
    outlier_columns = st.columns(2)
    for index, measure in enumerate(("sales", "profit", "discount", "quantity")):
        with outlier_columns[index % 2]:
            _render_chart(f"distribution-{measure}", figures[f"outlier_{measure}"])
            finding = outliers[measure]
            st.caption(
                f"{finding['outlier_count']:,} values outside "
                f"[{finding['lower_fence']:,.3f}, {finding['upper_fence']:,.3f}] "
                "under the 1.5 × IQR rule."
                if finding["lower_fence"] is not None
                else "No valid values are available."
            )

    _render_section_heading(
        "Customer RFM",
        "Rule-based recency, frequency, and monetary summaries describe the selected historical records.",
    )
    rfm, info = calculate_customer_rfm(data)
    left, right = st.columns([1, 1.5])
    with left:
        _render_chart("rfm-segments", figures["rfm_segments"])
    with right:
        st.markdown("**Customer RFM sample**")
        rfm_sample = rfm.head(20)
        with st.container(key="rfm-table"):
            st.dataframe(
                rfm_sample,
                hide_index=True,
                width="stretch",
                alt="Top filtered customer RFM records",
            )
        with st.container(key="mobile-rfm-cards"):
            st.markdown(
                "".join(
                    _mobile_data_card(
                        f"Customer {row.customer_id}",
                        str(row.segment),
                        (
                            ("Monetary sales", f"{row.monetary_sales:,.2f}"),
                            ("Distinct orders", f"{row.frequency_orders:,}"),
                            ("Recency", f"{row.recency_days:,} days"),
                        ),
                    )
                    for row in rfm_sample.itertuples(index=False)
                ),
                unsafe_allow_html=True,
            )
        st.caption(
            f"Recency is measured from {info['as_of_date']:%Y-%m-%d} "
            "to each customer's last filtered order; frequency counts distinct order IDs. "
            f"Rows without a customer ID are excluded: {info['excluded_missing_customer_rows']:,}."
        )
        with st.expander("RFM segment rules and limits"):
            for rule in RFM_SEGMENT_RULES:
                st.write(f"- {rule}")
            st.write(
                "Segments are descriptive of the selected historical records, not verified "
                "future customer behavior. Date and dimension filters change these scores."
            )


def _render_evidence_card(finding: pd.Series, index: int) -> None:
    with st.container(key=f"leak-finding-{index}"):
        st.markdown(
            f'<div class="evidence-card"><p class="insight-label">{escape(str(finding["finding_type"]))}</p>'
            f'<h4>{escape(str(finding["subject"]))}</h4></div>',
            unsafe_allow_html=True,
        )
        st.write(
            f"**Observed:** sales {finding['sales']:,.2f}; profit "
            f"{finding['profit']:,.2f}; margin "
            f"{finding['margin_pct']:.2f}%"
            if pd.notna(finding["margin_pct"])
            else f"**Observed:** sales {finding['sales']:,.2f}; profit {finding['profit']:,.2f}; margin unavailable"
        )
        if pd.notna(finding["baseline_margin_pct"]):
            st.write(
                f"**Peer/portfolio baseline:** {finding['baseline_margin_pct']:.2f}% "
                f"({finding['margin_gap_pp']:+.2f} percentage points)"
            )
        st.write(
            f"**Sample:** {int(finding['row_count']):,} order lines, "
            f"{int(finding['order_count']):,} distinct orders; "
            f"{int(finding['quantity']):,} units."
        )
        st.write(
            f"**Financial impact:** {finding['financial_impact']:,.2f} "
            f"({finding['impact_basis']})."
        )
        st.write(f"**Why investigate:** {finding['explanation']}")
        st.write(f"**Next question:** {finding['next_question']}")


def _render_profit_leak_detective(data: pd.DataFrame) -> None:
    _render_section_heading(
        "Profit Leak Detective",
        "A rule-based investigation queue grounded in filtered order lines—not proof of error or recoverable savings.",
    )
    if data.empty:
        st.info("No records match the active filters. Widen the filters to investigate profitability.")
        return

    findings = find_profit_leaks(data)
    if findings.empty:
        st.success("No findings met the documented sample and comparison thresholds in this selection.")
    else:
        st.caption(
            f"{len(findings):,} findings met the minimum group/sample rules. "
            "High-sales products require at least 2 lines and sales at or above category median; "
            "region/category and discount groups require at least 20 lines; negative-profit pool "
            "products require at least 2 lines. Product comparisons use other products in the same "
            "category; regional comparisons use other regions within that category. Discount groups "
            "are descriptive and not causal. Data-quality flags are tracked separately in "
            "reports/data_quality_report.json."
        )
        for index, (_, finding) in enumerate(findings.head(12).iterrows()):
            with st.expander(f"{index + 1}. {finding['finding_type']} · {finding['subject']}"):
                _render_evidence_card(finding, index)
        st.dataframe(
            findings.head(100),
            hide_index=True,
            width="stretch",
            alt="Ranked evidence-backed profit investigation findings",
        )

    _render_section_heading(
        "Profit anomaly map",
        "Compare each eligible cell’s margin with peer cells in the same category and sub-category.",
    )
    anomaly_rows = score_profit_anomalies(data)
    scorable = anomaly_rows.loc[
        anomaly_rows["robust_deviation_score"].notna()
        & anomaly_rows["sales"].gt(0)
    ].copy()
    if scorable.empty:
        st.info(
            "There are not enough comparable region/category/sub-category/discount peer cells "
            "with non-zero margin dispersion to compute a robust anomaly score."
        )
    else:
        anomaly_figure = px.scatter(
            scorable,
            x="peer_median_margin_pct",
            y="profit_margin_pct",
            color="robust_deviation_score",
            size="row_count",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data=[
                "region",
                "category",
                "sub_category",
                "discount",
                "sales",
                "profit",
                "row_count",
                "peer_median_margin_pct",
                "robust_deviation_score",
                "unusual_low_margin",
            ],
            title="Observed margin vs. comparable peer-cell median",
            labels={
                "peer_median_margin_pct": "Peer median margin (%)",
                "profit_margin_pct": "Observed margin (%)",
                "robust_deviation_score": "Robust deviation score",
                "row_count": "Order-line count",
            },
        )
        low = min(scorable["peer_median_margin_pct"].min(), scorable["profit_margin_pct"].min())
        high = max(scorable["peer_median_margin_pct"].max(), scorable["profit_margin_pct"].max())
        anomaly_figure.add_shape(
            type="line",
            x0=low,
            y0=low,
            x1=high,
            y1=high,
            line={"color": PALETTE["muted"], "dash": "dot"},
        )
        _apply_chart_theme(anomaly_figure)
        _render_chart("profit-anomaly-map", anomaly_figure)
        unusual = int(scorable["unusual_low_margin"].sum())
        st.caption(
            f"Robust deviation is (cell margin − median peer-cell margin) / "
            f"(1.4826 × median absolute deviation). Peer cells share category and sub-category; "
            f"cells need at least 10 order lines and 3 peer cells. {unusual} scored cells fell "
            "at or below −2.5. Unusual values warrant review and are not necessarily incorrect."
        )
        st.dataframe(
            scorable.sort_values("robust_deviation_score").head(30),
            hide_index=True,
            width="stretch",
            alt="Lowest robust profit margin anomaly scores",
        )

    _render_section_heading(
        "What-if discount simulator",
        "Estimate a hypothetical uniform discount scenario using the selected historical records and stated assumptions.",
    )
    categories = sorted(data["category"].dropna().astype(str).unique())
    if not categories:
        st.info("Category is unavailable for the current filtered data.")
        return
    controls = st.columns(3)
    with controls[0]:
        category = st.selectbox("Scenario category", categories, key="scenario-category")
    category_rows = data.loc[data["category"].astype(str).eq(category)]
    subcategories = sorted(category_rows["sub_category"].dropna().astype(str).unique())
    with controls[1]:
        subcategory = st.selectbox(
            "Scenario sub-category",
            ["All sub-categories", *subcategories],
            key="scenario-sub-category",
        )
    scenario_rows = category_rows
    if subcategory != "All sub-categories":
        scenario_rows = scenario_rows.loc[
            scenario_rows["sub_category"].astype(str).eq(subcategory)
        ]
    scenario_supported = not scenario_rows.empty and not (
        (scenario_rows["discount"] < 0).any()
        or (scenario_rows["discount"] >= 1).any()
        or (scenario_rows["sales"] < 0).any()
        or scenario_rows[["sales", "profit", "discount"]].isna().any().any()
    )
    if not scenario_supported:
        default_discount = 0.0
    else:
        gross_sales = (
            scenario_rows["sales"] / (1 - scenario_rows["discount"])
        ).sum()
        baseline_sales = float(scenario_rows["sales"].sum())
        default_discount = (
            1 - baseline_sales / gross_sales if gross_sales > 0 else 0.0
        )
    with controls[2]:
        scenario_discount_pct = st.slider(
            "Hypothetical uniform discount",
            min_value=0,
            max_value=80,
            value=int(round(min(max(default_discount, 0.0), 0.8) * 100)),
            step=1,
            format="%d%%",
            key="scenario-discount",
        )
    scenario_discount = scenario_discount_pct / 100
    if scenario_rows.empty:
        st.info("No historical rows match this scenario selection.")
    else:
        try:
            scenario = estimate_discount_scenario(scenario_rows, scenario_discount)
        except ValueError as error:
            st.warning(f"Scenario estimate unavailable: {error}")
        else:
            metrics = st.columns(4)
            metrics[0].metric("Baseline sales", f"{scenario['baseline_sales']:,.2f}")
            metrics[1].metric("Baseline profit", f"{scenario['baseline_profit']:,.2f}")
            metrics[2].metric("Scenario profit estimate", f"{scenario['scenario_profit_estimate']:,.2f}")
            metrics[3].metric("Estimated profit difference", f"{scenario['profit_difference_estimate']:+,.2f}")
            st.caption(
                f"Scenario covers {scenario['row_count']:,} historical order lines; "
                f"weighted baseline discount is {scenario['baseline_weighted_discount']:.1%}, "
                f"scenario discount is {scenario['scenario_discount']:.1%}. "
                f"{scenario['assumption']}"
            )

def render_dashboard(database_path: Path = DEFAULT_DATABASE) -> None:
    """Render the Streamlit app; database failures are surfaced to the user."""
    st.set_page_config(
        page_title="RetailIQ | Business Analytics",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
    try:
        data = load_dashboard_data(database_path)
    except DatabaseLoadError as error:
        st.error(str(error))
        st.info(
            "The dashboard reads the SQLite database only. It does not substitute "
            "sample or generated data."
        )
        st.stop()

    st.markdown(
        """
        <div class="hero">
          <div class="brand-mark"><span></span>Retail intelligence, refined</div>
          <h1><span>RetailIQ</span> Business Analytics</h1>
          <p>One clear view of sales, margin, and product performance — powered by your actual retail data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    min_date = data["order_date"].min().date()
    max_date = data["order_date"].max().date()
    with st.sidebar:
        with st.container(key="filter-panel"):
            st.markdown('<p class="section-kicker">Refine your view</p>', unsafe_allow_html=True)
            st.caption("All metrics and charts follow these selections.")
            st.markdown('<p class="filter-group-label">Date coverage</p>', unsafe_allow_html=True)
            chosen_dates = st.date_input(
                "Order date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(chosen_dates, tuple) and len(chosen_dates) == 2:
                date_range = chosen_dates
            elif isinstance(chosen_dates, (date, pd.Timestamp)):
                date_range = (chosen_dates, chosen_dates)
            else:
                date_range = (min_date, max_date)

            selections: dict[str, list[str]] = {}
            st.markdown('<p class="filter-group-label">Dimensions</p>', unsafe_allow_html=True)
            for label, column in FILTER_COLUMNS.items():
                options = sorted(data[column].dropna().astype(str).unique().tolist())
                selections[column] = st.multiselect(
                    label,
                    options,
                    default=options,
                )
            st.caption("Currency is not specified in the source; values use dataset units.")

    filtered = filter_sales_data(data, date_range, selections)
    kpis = calculate_kpis(filtered)
    st.markdown(
        f"Showing {len(filtered):,} order lines from "
        f"{pd.Timestamp(date_range[0]).strftime('%b %d, %Y')} to "
        f"{pd.Timestamp(date_range[1]).strftime('%b %d, %Y')}.",
        unsafe_allow_html=False,
    )
    metric_columns = st.columns(5)
    metric_specs = (
        (
            "sales",
            "Total sales",
            _format_amount(kpis["total_sales"]),
            "Sum of sales across the currently filtered order lines. Values use dataset monetary units; the source does not specify a currency.",
        ),
        (
            "profit",
            "Total profit",
            _format_amount(kpis["total_profit"]),
            "Sum of profit across the currently filtered order lines. Negative-profit lines remain included.",
        ),
        (
            "margin",
            "Profit margin",
            f"{kpis['profit_margin_pct']:.2f}%"
            if kpis["profit_margin_pct"] is not None
            else "N/A",
            "Filtered total profit divided by filtered total sales, multiplied by 100. Unavailable when filtered sales are zero.",
        ),
        (
            "orders",
            "Distinct orders",
            f"{kpis['distinct_orders']:,}",
            "Count of unique order IDs represented by the currently filtered order lines.",
        ),
        (
            "quantity",
            "Quantity sold",
            f"{kpis['quantity_sold']:,}",
            "Sum of quantity across the currently filtered order lines.",
        ),
    )
    for column, (key, label, value, definition) in zip(metric_columns, metric_specs):
        with column.container(key=f"kpi-{key}"):
            st.metric(label, value, help=definition)

    overview_tab, advanced_tab, detective_tab = st.tabs(
        ["Executive overview", "Advanced analytics", "Profit Leak Detective"],
        on_change="rerun",
        key="retailiq-views",
    )
    if overview_tab.open:
        with overview_tab:
            charts = build_dashboard_charts(filtered)
            _render_section_heading(
                "Performance trends",
                "Track monthly movement and compare sales and profit across retail dimensions.",
            )
            _render_chart("monthly", charts["monthly"])
            left, right = st.columns(2)
            with left:
                _render_chart("category", charts["category"])
            with right:
                _render_chart("region", charts["region"])
            left, right = st.columns(2)
            with left:
                _render_chart("segment", charts["segment"])
            with right:
                _render_chart("ship-mode", charts["ship_mode"])

            _render_section_heading(
                "Discount and profitability",
                "Order-line discount and profit patterns are descriptive associations, not causal effects.",
            )
            left, right = st.columns(2)
            with left:
                _render_chart("discount", charts["discount"])
            with right:
                with st.container(key="discount-note"):
                    st.markdown(
                        """
                        <div class="insight-card">
                          <p class="insight-label">How to read this view</p>
                          <p class="insight-copy">Each point is an order line. Discount and profit patterns are
                          descriptive associations; they do not show that discounts cause profit changes.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            _render_section_heading(
                "Product performance",
                "Review top contributors and products with negative total profit in the current selection.",
            )
            left, right = st.columns(2)
            with left:
                _render_chart("top-sales", charts["top_sales"])
                products = _product_summary(filtered)
                top_products = products.nlargest(10, "sales").sort_values(
                    "sales",
                    ascending=False,
                )
                with st.container(key="product-table"):
                    st.dataframe(
                        top_products,
                        column_config={
                            "product_id": st.column_config.TextColumn("Product ID", width="small"),
                            "product_name": st.column_config.TextColumn("Product", width="medium"),
                            "sales": st.column_config.NumberColumn("Sales", width="small"),
                            "profit": st.column_config.NumberColumn("Profit", width="small"),
                            "quantity": st.column_config.NumberColumn("Units", width="small"),
                            "order_count": st.column_config.NumberColumn("Orders", width="small"),
                        },
                        width="stretch",
                        hide_index=True,
                        alt="Top ten products ranked by sales",
                    )
                with st.container(key="mobile-product-cards"):
                    st.markdown(
                        "".join(
                            _mobile_data_card(
                                str(row.product_name),
                                f"Product ID {row.product_id}",
                                (
                                    ("Sales", f"{row.sales:,.2f}"),
                                    ("Profit", f"{row.profit:,.2f}"),
                                    ("Quantity", f"{row.quantity:,.0f}"),
                                    ("Distinct orders", f"{row.order_count:,}"),
                                ),
                            )
                            for row in top_products.itertuples(index=False)
                        ),
                        unsafe_allow_html=True,
                    )
            with right:
                _render_chart("top-profit", charts["top_profit"])
                _render_chart("negative-profit", charts["negative_profit"])

            _render_section_heading("Business insights")
            insight_columns = st.columns(2)
            for index, insight in enumerate(build_business_insights(filtered)):
                with insight_columns[index % 2]:
                    _render_insight_card(index, insight)
            st.caption(
                "Discount-profit relationships are descriptive associations. They do not establish causation."
            )
    if advanced_tab.open:
        with advanced_tab:
            _render_advanced_analytics(filtered)
    if detective_tab.open:
        with detective_tab:
            _render_profit_leak_detective(filtered)
    st.caption(
        f"Data source: {Path(database_path).name} · "
        f"{data['order_date'].min():%Y-%m-%d} to {data['order_date'].max():%Y-%m-%d}"
    )


if __name__ == "__main__":
    render_dashboard()
