-- RetailIQ analyses use the loaded cleaned rows in retail_sales.
-- Profit margin is total profit / total sales * 100.
-- NULLIF protects every margin calculation against division by zero.

-- name: overall_totals
SELECT
    COUNT(*) AS row_count,
    SUM(sales) AS total_sales,
    SUM(profit) AS total_profit,
    SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS overall_profit_margin_pct
FROM retail_sales;

-- name: monthly_trends
SELECT
    strftime('%Y-%m', order_date) AS year_month,
    SUM(sales) AS total_sales,
    SUM(profit) AS total_profit,
    SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
    COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY strftime('%Y-%m', order_date)
ORDER BY year_month;

-- name: yearly_trends
SELECT
    strftime('%Y', order_date) AS year,
    SUM(sales) AS total_sales,
    SUM(profit) AS total_profit,
    SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
    COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY strftime('%Y', order_date)
ORDER BY year;

-- name: by_region
SELECT region, SUM(sales) AS total_sales, SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY region
ORDER BY total_sales DESC;

-- name: by_category
SELECT category, SUM(sales) AS total_sales, SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY category
ORDER BY total_sales DESC;

-- name: by_sub_category
SELECT category, sub_category, SUM(sales) AS total_sales,
       SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY category, sub_category
ORDER BY total_sales DESC;

-- name: top_10_products_by_sales
SELECT product_id, product_name, SUM(sales) AS total_sales,
       SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY product_id, product_name
ORDER BY total_sales DESC, product_id
LIMIT 10;

-- name: top_10_products_by_profit
SELECT product_id, product_name, SUM(sales) AS total_sales,
       SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY product_id, product_name
ORDER BY total_profit DESC, product_id
LIMIT 10;

-- name: negative_profit_products
SELECT product_id, product_name, SUM(sales) AS total_sales,
       SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY product_id, product_name
HAVING SUM(profit) < 0
ORDER BY total_profit, product_id;

-- name: by_customer_segment
SELECT segment, SUM(sales) AS total_sales, SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY segment
ORDER BY total_sales DESC;

-- name: by_shipping_mode
SELECT ship_mode, SUM(sales) AS total_sales, SUM(profit) AS total_profit,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct,
       COUNT(DISTINCT order_id) AS order_count
FROM retail_sales
GROUP BY ship_mode
ORDER BY total_sales DESC;

-- name: by_discount
-- This is a descriptive association, not a causal estimate.
SELECT discount, COUNT(*) AS order_line_count,
       SUM(sales) AS total_sales, SUM(profit) AS total_profit,
       AVG(profit) AS average_profit_per_line,
       SUM(profit) / NULLIF(SUM(sales), 0) * 100.0 AS profit_margin_pct
FROM retail_sales
GROUP BY discount
ORDER BY discount;
