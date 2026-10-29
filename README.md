# RetailIQ

RetailIQ is a retail business and data analytics portfolio project. It turns
order-line data into an auditable workflow—from data-quality checks and
exploratory analysis through SQL reporting and an interactive dashboard—so a
retail analyst can investigate sales, profitability, discount patterns, and
customer activity using a consistent set of filters.

The project is designed to make analytical assumptions visible. Findings are
calculated from the selected records, descriptive comparisons are not
presented as causal effects, and hypothetical scenario results are identified
as estimates rather than forecasts.

## Objectives

RetailIQ demonstrates a reproducible, auditable retail analytics workflow. Its
objectives are to validate and clean order-line data without silently deleting
questionable records, produce traceable exploratory and SQL summaries from
the same cleaned data, provide an interactive dashboard for investigation, and
make analytical assumptions explicit so results support decisions without
overstating causality.

## Features

- Strict CSV loading and reproducible cleaning with a machine-readable data
  quality report; flagged rows and outliers are reviewed, not silently removed.
- Exploratory statistics and interactive charts across time, products,
  regions, customer segments, and shipping modes.
- A local SQLite database and named SQL analyses with CSV exports.
- A filterable Streamlit dashboard with executive KPIs, interactive Plotly
  charts, business insights, and responsive midnight-navy glass styling.
- Advanced seasonal heatmaps, product hierarchy treemap, discount bubbles,
  sales Pareto, profit waterfalls, distribution analysis, and customer RFM.
- Profit Leak Detective: evidence-backed peer comparisons, a robust anomaly
  view, and an explicitly hypothetical discount simulator.

## Technology stack

| Technology | Use |
| --- | --- |
| Python | Data workflow, calculations, dashboard, and tests |
| Pandas and NumPy | Cleaning, aggregation, metrics, and analytical methods |
| SQLite | Local retail-sales database and SQL analysis |
| Streamlit | Interactive dashboard and filters |
| Plotly | Interactive charts |
| `unittest` | Automated regression and workflow tests |

SQLite is included with Python; there is no separate database server.

## Dataset and attribution

The project uses the **Sample Superstore** order-line dataset. The raw source
input is `data/Sample - Superstore.csv`. The cleaning workflow creates
`data/cleaned/superstore_cleaned.csv`, which is then used to build the local
SQLite database and analytical reports. The dataset contains order and ship
dates, products, customers, regions, sales, profit, quantity, and discount.
Its monetary figures are reported in dataset units because the source does not
specify a currency symbol.

A community-hosted copy is available from Kaggle at
[Superstore Dataset (Final)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).
That page is a download reference, not a claim that this project created or
owns the data. Check the source uploader's current terms and attribution
requirements before redistributing the dataset. The loader reads the raw CSV
without changing it.

## Architecture

```text
data/Sample - Superstore.csv
        |
        +--> src/load_data.py --> src/clean_data.py
                                    |--> data/cleaned/superstore_cleaned.csv
                                    |--> reports/data_quality_report.json
                                    |
                                    +--> src/eda.py --> reports/eda_summary.json
                                    |                 reports/eda/*.html
                                    |
                                    +--> src/build_database.py
                                          |--> data/retailiq.db (local, ignored)
                                          |      |--> src/run_sql_analysis.py
                                          |      |      |--> sql/analysis.sql
                                          |      |      +--> reports/sql/*.csv
                                          |      +--> src/app.py
                                          |              +--> src/advanced_analytics.py
                                          +--> tests/
```

`src/app.py` reads the local SQLite database in read-only mode. Dashboard
filters are applied to the loaded order lines before KPI, chart, and
investigation calculations. The analytics layer does not update the database.
The database is ignored by Git and is regenerated from the cleaned CSV with
`src/build_database.py` when it is absent or stale.

### Inputs and generated artifacts

| Classification | Path | Role |
| --- | --- | --- |
| Raw input | `data/Sample - Superstore.csv` | Source order-line dataset; the workflow reads it without modifying it. |
| Generated cleaned input | `data/cleaned/superstore_cleaned.csv` | Normalized dataset produced by `src/clean_data.py`; consumed by the database and reporting steps. |
| Generated local build artifact | `data/retailiq.db` | Ignored SQLite database built from the cleaned CSV; it is not included in the repository. |
| Generated EDA artifacts | `reports/data_quality_report.json`, `reports/eda_summary.json`, `reports/eda/*.html` | Data-quality and exploratory-analysis outputs from the cleaning and EDA workflows. |
| Generated SQL exports | `reports/sql/*.csv` | CSV result exports from the named queries in `sql/analysis.sql`. |

The cleaned data and reports are retained in this working copy for inspection;
the existing `.gitignore` does not blanket-ignore `data/` or `reports/`.

## Windows PowerShell setup

Use Python 3.12 or a compatible Python installation. From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is restricted by local policy, skip activation and
use the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project requirements are Pandas, Streamlit, and Plotly. NumPy is installed
as a Pandas dependency; SQLite and `unittest` are part of Python's standard
library.

## Run the project

Run commands from the project root.

### Launch the dashboard

The local database may already exist in a working copy, but it is ignored by
Git and is not assumed to be included in a checkout. To regenerate it from
the cleaned CSV, run the builder first; then start Streamlit:

```powershell
# Regenerate the ignored local database when it is absent or stale:
.\.venv\Scripts\python.exe .\src\build_database.py
# Launch the dashboard:
.\.venv\Scripts\python.exe -m streamlit run .\src\app.py
```

The app reads `data\retailiq.db` in read-only mode and does not substitute
sample or generated records when the database is unavailable. The builder
expects `data\cleaned\superstore_cleaned.csv`; run the cleaning workflow below
first if that file is absent.

### Run the tests and syntax checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .\src .\tests
```

### Run the data workflow

```powershell
.\.venv\Scripts\python.exe .\src\load_data.py
.\.venv\Scripts\python.exe .\src\clean_data.py
.\.venv\Scripts\python.exe .\src\eda.py
.\.venv\Scripts\python.exe .\src\build_database.py
.\.venv\Scripts\python.exe .\src\run_sql_analysis.py
```

The cleaning, EDA, database, and SQL commands regenerate their documented
outputs. `load_data.py` only inspects the raw input; it does not create the
cleaned CSV. Do not rerun the data workflow when you only want to launch the
dashboard or run the tests.

## Dashboard tabs

The dashboard in `src/app.py` has three tabs. The date, region, category,
customer-segment, and shipping-mode filters apply to all of them:

- **Executive overview:** filtered KPIs, business insights, time trends,
  regional/category/segment/shipping performance, and product summaries.
- **Advanced analytics:** seasonal heatmaps, product hierarchy treemap,
  discount bubbles, sales Pareto, profit waterfalls, distribution/outlier
  views, and customer RFM segments.
- **Profit Leak Detective:** ranked rule-based findings, peer comparisons,
  robust anomaly scoring, and a hypothetical discount simulator.

## Last verification run

The last local verification run produced these results; they describe that
run and are not a guarantee about future environments or data:

- 51 automated tests passed.
- Python source and test compilation passed.
- The dashboard loaded 9,994 rows from the local database.
- Standard and advanced chart generation passed for product and sub-category
  discount-bubble views.
- Streamlit started successfully and returned HTTP 200 in a headless smoke
  check.

## Important metrics and methods

- **Sales** and **profit** are sums of the corresponding fields over the
  filtered order lines. Negative profit remains in the total.
- **Profit margin** is filtered total profit divided by filtered total sales,
  multiplied by 100. It is unavailable when total sales are zero; margins are
  computed from aggregate profit and sales, not averaged across lines.
- **Distinct orders** counts unique order IDs represented by the selected
  lines. A dimension filter can include only some lines from an order.
- **Quantity sold** is the sum of the quantity field over selected lines.
- **Discount** is the recorded order-line discount proportion. Group-level
  discount views show observed associations; they do not establish that
  discounts caused a change in profit.
- **Customer RFM** uses recency in days from each customer's last selected
  order to the latest selected order date; frequency is distinct selected
  order IDs; monetary value is selected sales. Percentile-based segments are
  descriptive of historical selected data, not predictions.
- **Profit Leak Detective** compares eligible product margins to the
  sales-weighted margin of other products in the same category, regional
  margins to other regions in the same category, and discount-level margins
  to other observed discount levels. Sample thresholds are applied. A
  margin-gap amount is a comparison basis—not confirmed recoverable savings.
- **Anomaly scores** use a robust deviation from peer-cell median margins;
  unusual scores indicate cases to review, not data errors.
- **Discount scenarios** hold quantity, list prices, product mix, and unit
  costs constant. They are mechanical estimates under stated assumptions, not
  forecasts or causal estimates.

## Screenshots

No screenshots are included yet. To add portfolio images, run the dashboard
locally, select the desired tab and filter state, capture genuine screenshots
at a readable desktop size, and save the original captures under
`docs/screenshots/`. Use names such as `executive-overview.png`,
`advanced-analytics.png`, and `profit-leak-detective.png`; then add standard
Markdown image links here. Do not draw charts, edit values, or present
mockups as screenshots of the running application.

## Limitations and future enhancements

- This is a historical sample dataset with no currency specified and no
  information about product costs beyond the provided profit field.
- Discount, margin, and regional comparisons are observational and may reflect
  differences in product mix, customer context, or other unmeasured factors.
- RFM segments and anomaly flags are descriptive rules; they have not been
  validated as predictive models.
- The discount simulator's fixed-volume and fixed-cost assumptions are
  simplifying assumptions, not demand or cost estimates.
- The app is designed for local portfolio use. No hosted deployment, user
  authentication, or multi-user database service is configured.

Possible future work includes time-aware validation of demand or retention
models, documented scenario sensitivity analysis, automated data-quality
checks in continuous integration, and screenshots captured from the verified
local dashboard.

## Analytics details

The dashboard separates the **Executive overview**, **Advanced analytics**,
and **Profit Leak Detective** into tabs. The existing date, region, category,
customer-segment, and shipping-mode filters apply to every tab. Advanced
calculations live in `src\advanced_analytics.py` and use the filtered rows
loaded from SQLite; no advanced view changes the database.

The Advanced analytics tab adds:

- Calendar-month/category heatmaps for sales and profit, and a region/category
  profit-margin heatmap. Calendar months pool matching months across selected
  years, so use the view to compare seasonal patterns rather than year-specific
  monthly totals. A zero-sales group's margin remains unavailable.
- A category → sub-category → product treemap. Area is aggregated sales and
  color is profit margin; groups with zero or negative total sales are excluded
  from this area encoding and counted in the chart annotation.
- A discount bubble chart aggregated by product or sub-category and observed
  discount. Bubble size represents sales, vertical position is profit margin,
  and color represents profit. It describes associations, not causes.
- Product sales Pareto contribution and cumulative share. Percentages use total
  sales in the current filtered selection; the 80% line is only a reference.
- Category and category/sub-category profit waterfalls. The contribution
  values are checked against total filtered profit before they are presented.
- Sales, profit, discount, and quantity box plots. Outlier markers use the
  1.5 × IQR rule and are investigation signals, not confirmed errors; no values
  are removed.
- Customer RFM metrics: recency is days since the customer's last filtered
  order relative to the latest filtered order date; frequency counts distinct
  order IDs; monetary is filtered sales. The documented percentile rules assign
  descriptive segments, not predictions of future customer behavior.

No Sankey chart is included: the available order table records attributes of
each order line, not a meaningful directed journey or transition, so connecting
region and category would suggest a flow the data does not establish.

Profit Leak Detective presents a ranked, rule-based investigation queue with
sales, profit, margin, comparison baseline, sample size, financial-impact
basis, and suggested next questions. Product margins are compared with other
products in the same category; candidate products must have at least two
order lines and sales at or above the category's median product sales.
Region/category groups need at least 20 order lines and are compared with the
other regions in that category. Discount groups need at least 20 lines and are
compared with other observed discount levels; this is descriptive, not causal.
Negative-profit product findings require at least two lines. A calculated
margin gap is a comparison amount, not recoverable savings. The anomaly map
compares region/category/sub-category/discount cells with peer cells in the
same category and sub-category; a cell needs at least 10 lines and three peers
to be scored. Its robust score is `(cell margin - peer median margin) /
(1.4826 × peer median absolute deviation)`; scores at or below `-2.5` are
flagged for review, not labeled as data errors.

The discount simulator uses selected historical rows and estimates a
hypothetical uniform discount scenario. It infers pre-discount sales as
`observed sales / (1 - observed discount)` and calculates scenario sales at the
chosen discount. Estimated profit is baseline profit plus the change in sales,
assuming quantity, list prices, product mix, and unit costs remain unchanged.
The result is a mechanical scenario, not a forecast or causal estimate; it is
unavailable when the selected rows do not support those assumptions.

If Streamlit reports that the database is missing, run the database build
command above. If it reports an invalid database or missing columns, rebuild
the database from the cleaned CSV. The dashboard does not create or silently
substitute business data.
