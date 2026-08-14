# Data Pipeline Module

This module implements the data engineering pipeline that turns raw scraped data into a clean relational store.

## Workflow

1. **Extraction (Mocking)**: The `pipeline.py` script generates JSON arrays simulating raw, scraped data from Zepto (products, customers, and orders). We intentionally inject **dirty data**, such as negative prices, missing values, and duplicate records, to simulate real-world data engineering challenges.
2. **Transformation**: The raw JSON files are loaded into Pandas DataFrames. We perform robust data cleaning:
   - Deduplication of records based on unique IDs.
   - Anomaly removal (dropping orders with negative total amounts and products with negative prices).
   - Missing value imputation (filling missing weather data).
   - Type-casting (converting string dates to proper datetime objects).
3. **Loading**: The cleaned DataFrames are written into normalized tables (`products`, `customers`, `orders`) inside a local `sqlite3` database (`zepto.db`).

## Execution

Run the script from the root directory:
```bash
python data_pipeline/pipeline.py
```
This will create a `raw_data/` folder containing the JSONs, and a `zepto.db` SQLite database file at the root of the project.
