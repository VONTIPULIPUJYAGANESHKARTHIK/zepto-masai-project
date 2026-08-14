# Zepto Data & AI Platform Capstone

This repository contains the complete end-to-end implementation for the Zepto Data & Analytics Capstone Project. It is divided into three interconnected modules demonstrating data engineering, advanced analytics, and generative AI.

## Setup Instructions

A consolidated `requirements.txt` is provided in the root directory for all three modules. 
To install the dependencies, it is recommended to create a virtual environment, then run:

```bash
pip install -r requirements.txt
```

---

## Module 1: Data Pipeline (`/data_pipeline`)

This module implements a complete web scraping and data engineering pipeline. It simulates the extraction of catalogue pricing data before it reaches an analytics dashboard.

### Features
- **Scraping**: Uses `requests` and `BeautifulSoup` to scrape exactly 1,000 books across all 50 pages of the `books.toscrape.com` catalogue.
- **Cleaning**: Parses numerical ratings (e.g. 'One' to 1), converts availability to boolean, and cleanly handles messy/missing rows.
- **Transformation**: Converts prices from GBP (£) to INR (₹) using a strict, fixed exchange rate of `105.50`.
- **Relational Storage**: Stores the cleaned data in a normalized `sqlite3` database (`zepto.db`) using two tables (`books` and `categories`).
- **Validation**: Executes 5 required SQL aggregate/join queries and validates the results by comparing them directly against equivalent Pandas DataFrame `merge` operations to ensure absolute parity.

### Execution
Run the pipeline script from the root directory:
```bash
python data_pipeline/pipeline.py
```

---

## Module 2: Analytics Pipeline (`/analytics`)

This module implements an end-to-end analyst-to-data-scientist workflow on the classic **Titanic dataset**, fulfilling all required statistical profiling and machine learning tasks.

### Features
- **Exploratory Data Analysis (`01_eda.ipynb`)**: Loads the dataset via Seaborn, profiles missing values, performs univariate/bivariate analysis, computes a correlation heatmap, builds a multivariate data story of survival likelihood, and validates a z-score standardization. It saves the cleaned data to `titanic.csv` as an offline fallback.
- **Predictive Modeling (`02_modeling.ipynb`)**: Reads the offline fallback CSV and executes a modeling pipeline. It implements a strictly separated `ColumnTransformer` (fit on train only), evaluates three classifiers side-by-side (Logistic Regression, Decision Tree, Random Forest), handles class imbalance via `SMOTE` vs `class_weight`, tunes the Random Forest via `GridSearchCV`, and performs a linear regression side-task to predict fares.
- **Deployment Artifact**: Saves the full `ColumnTransformer` and best estimator as a single `joblib` artifact.

### Documentation & Interpretations
All required written interpretations (missing value strategies, skewness, correlation analysis, data story conclusions, and the final model deployment recommendation) are documented directly inside the module's dedicated README:
**[View Analytics Documentation](analytics/README.md)**

### Execution
Open the Jupyter Notebooks inside the `/analytics` directory and execute all cells sequentially.

---

## Module 3: Support Assistant (`/support_assistant`)

This module implements a complete GenAI Retrieval-Augmented Generation (RAG) service for Zepto's policy corpus, orchestrated via LangGraph and exposed through a FastAPI endpoint.

It features a strict `MOCK_LLM` toggle that allows the pipeline to run completely offline without any API keys, using deterministic mock logic for grading, while preserving the ability to run a live LLM (like Groq) via an environment variable.

### Documentation & Architecture
The full architecture breakdown, including the prompt schema and mock transcripts, is documented directly inside the module's dedicated README:
**[View Support Assistant Documentation](support_assistant/README.md)**

### Execution
1. **Setup**: Run `python support_assistant/ingest.py` to embed the documents and initialize ChromaDB.
2. **Server**: Run `uvicorn support_assistant.assistant:app --host 0.0.0.0 --port 7860`.
3. **Docker**: 
   ```bash
   docker build -t zepto-support .
   docker run -p 7860:7860 zepto-support
   ```