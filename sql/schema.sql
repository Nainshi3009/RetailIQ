-- RetailIQ cleaned Superstore records.
-- Dates are stored as ISO-8601 text (YYYY-MM-DD); amounts are SQLite REALs.
CREATE TABLE IF NOT EXISTS retail_sales (
    row_id INTEGER,
    order_id TEXT,
    order_date TEXT,
    ship_date TEXT,
    ship_mode TEXT,
    customer_id TEXT,
    customer_name TEXT,
    segment TEXT,
    country TEXT,
    city TEXT,
    state TEXT,
    postal_code INTEGER,
    region TEXT,
    product_id TEXT,
    category TEXT,
    sub_category TEXT,
    product_name TEXT,
    sales REAL,
    quantity INTEGER,
    discount REAL,
    profit REAL
);

CREATE INDEX IF NOT EXISTS idx_retail_sales_order_date
    ON retail_sales(order_date);
CREATE INDEX IF NOT EXISTS idx_retail_sales_region
    ON retail_sales(region);
CREATE INDEX IF NOT EXISTS idx_retail_sales_category
    ON retail_sales(category, sub_category);
CREATE INDEX IF NOT EXISTS idx_retail_sales_product
    ON retail_sales(product_id);
