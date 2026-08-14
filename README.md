# Zepto Data & AI Platform Capstone

This repository contains the end-to-end Zepto Data & AI Platform implementation, which includes three interconnected modules:
1. `/data_pipeline`: A data engineering pipeline that simulates scraping Zepto raw data, cleans it, and stores it in a relational SQLite database.
2. `/analytics`: An analytics pipeline (Jupyter Notebook) that profiles the customer dataset end-to-end and builds a Random Forest model to predict delivery times.
3. `/support_assistant`: A grounded GenAI support assistant using local HuggingFace models to answer policy questions based on Zepto's documents.

## Setup Instructions

A consolidated `requirements.txt` is provided in the root directory for all three modules. 
To install the dependencies, it is recommended to create a virtual environment, then run:

```bash
pip install -r requirements.txt
```

*(Note: The support assistant requires PyTorch and Transformers, which will download a small language model on its first run.)*

## Running the Modules End to End

To see the full pipeline in action, follow these steps in order:

### 1. Data Pipeline
Generates the raw mock data and builds the relational store (`zepto.db`).
```bash
python data_pipeline/pipeline.py
```

### 2. Analytics
After generating the `zepto.db`, you can run the analytics notebook to see the profiling and modeling.
```bash
jupyter notebook analytics/analytics_end_to_end.ipynb
```
Run all cells in the notebook to view the Exploratory Data Analysis (EDA) and the predictive model evaluation.

### 3. Support Assistant
Run the GenAI CLI assistant to ask questions about Zepto's policies (e.g., "What happens if a delivery takes more than 20 minutes?").
```bash
python support_assistant/assistant.py
```

## Design Decisions

### Data Pipeline
- **Decision**: Python script generating JSONs mimicking scraped data, followed by Pandas cleaning and SQLite for the relational store.
- **Reasoning**: SQLite provides a robust, zero-configuration relational database that is perfect for local end-to-end demonstrations without requiring a separate database server. Pandas allows for vectorized cleaning of the raw JSON arrays.

### Analytics
- **Decision**: Jupyter Notebook using `scikit-learn` Random Forest Regressor.
- **Reasoning**: Notebooks provide an excellent format for combining code, visualizations (EDA), and Markdown interpretation. A Random Forest model handles non-linear relationships (like weather impacts) very well for predicting continuous variables like delivery time.

### Support Assistant
- **Decision**: `transformers` Question-Answering pipeline using `distilbert-base-cased-distilled-squad`.
- **Reasoning**: The project requires the use of strictly free services. By utilizing a local HuggingFace QA model, we can perform grounded document question-answering on the `zepto_policies.txt` text without relying on paid APIs like OpenAI, adhering strictly to the capstone constraints.