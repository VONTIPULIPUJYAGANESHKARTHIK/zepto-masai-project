
# Zepto Data & AI Platform Capstone

This repository contains the end-to-end Zepto Data & AI Platform implementation, which includes three interconnected modules:
1. `/data_pipeline`: A data engineering pipeline that simulates scraping Zepto raw data, cleans it, and stores it in a relational SQLite database.
2. `/analytics`: An analytics pipeline (Jupyter Notebook) that profiles the customer dataset end-to-end and builds a Random Forest model to predict delivery times.
3. `/support_assistant`: A grounded GenAI support assistant using local HuggingFace models to answer policy questions based on Zepto's documents.

---

## Setup Instructions

A consolidated `requirements.txt` is provided in the root directory for all three modules. 
To install the dependencies, it is recommended to create a virtual environment, then run:

```bash
pip install -r requirements.txt
```

*(Note: The support assistant requires PyTorch and Transformers, which will download a small language model on its first run.)*

## Docker Support (Recommended)

You can run the entire platform end-to-end inside a Docker container without needing to set up a local Python environment.

1. **Build the image**:
```bash
docker build -t zepto-ai-platform .
```
2. **Run the container**:
```bash
docker run -it zepto-ai-platform
```
This will automatically execute the data pipeline, train the ML models, and query the GenAI assistant, printing all the sample results directly to your terminal.

---

## Module 1: Data Pipeline

This module implements the data engineering pipeline that turns raw scraped data into a clean relational store.

### Workflow
1. **Extraction (Mocking)**: The `pipeline.py` script generates JSON arrays simulating raw, scraped data from Zepto (products, customers, and orders). We intentionally inject **dirty data**, such as negative prices, missing values, and duplicate records, to simulate real-world data engineering challenges.
2. **Transformation**: The raw JSON files are loaded into Pandas DataFrames. We perform robust data cleaning:
   - Deduplication of records based on unique IDs.
   - Anomaly removal (dropping orders with negative total amounts and products with negative prices).
   - Missing value imputation (filling missing weather data).
   - Type-casting (converting string dates to proper datetime objects).
3. **Loading**: The cleaned DataFrames are written into normalized tables (`products`, `customers`, `orders`) inside a local `sqlite3` database (`zepto.db`).

### Execution
Run the script from the root directory:
```bash
python data_pipeline/pipeline.py
```
This will create a `raw_data/` folder containing the JSONs, and a `zepto.db` SQLite database file at the root of the project.

### Design Decisions
- **Decision**: Python script generating JSONs mimicking scraped data, followed by Pandas cleaning and SQLite for the relational store.
- **Reasoning**: SQLite provides a robust, zero-configuration relational database that is perfect for local end-to-end demonstrations without requiring a separate database server. Pandas allows for vectorized cleaning of the raw JSON arrays.

---

## Module 2: Analytics Pipeline

This module implements an end-to-end analyst-to-data-scientist workflow on the classic **Titanic dataset**.

### Workflow
1. **`01_eda.ipynb`**: Loads the dataset via `sns.load_dataset('titanic')`, profiles missing values, performs univariate/bivariate analysis, computes a correlation heatmap, builds a multivariate data story of survival likelihood, and validates a z-score standardization. It saves the cleaned data to `titanic.csv` as an offline fallback.
2. **`02_modeling.ipynb`**: Reads the offline fallback CSV and continues straight into the predictive modeling pipeline. It implements a strictly separated `ColumnTransformer` (fit on train only), evaluates three classifiers side-by-side, tests SMOTE vs Class Weight balancing, tunes a Random Forest via `GridSearchCV`, and performs a linear regression side-task to predict fares.

### Documentation & Interpretations
All required written interpretations (missing value strategies, skewness, correlation analysis, data story conclusions, and the final model deployment recommendation) are documented directly inside the module's dedicated README:
**[View Analytics Documentation](analytics/README.md)**

### Execution
Open the Jupyter Notebooks inside the `/analytics` directory and execute all cells sequentially:
1. `analytics/01_eda.ipynb`
2. `analytics/02_modeling.ipynb`

### Design Decisions
- **Decision**: Jupyter Notebook using `scikit-learn` Random Forest Regressor.
- **Reasoning**: Notebooks provide an excellent format for combining code, visualizations (EDA), and Markdown interpretation. A Random Forest model handles non-linear relationships (like weather impacts) very well for predicting continuous variables like delivery time.

---

## Module 3: Support Assistant

This module implements a GenAI support assistant that answers policy questions grounded in Zepto's documents.

### Workflow
This module utilizes a **Retrieval-Augmented Generation (RAG)** architecture using entirely local, free, open-source models (avoiding paid APIs like OpenAI).

1. **Knowledge Base**: We defined a mock Zepto policy document (`zepto_policies.txt`) which contains details about delivery guarantees, cancellations, and premium memberships.
2. **Document Chunking**: The `assistant.py` script loads the policy document and splits it into logical, manageable chunks (paragraphs).
3. **Semantic Retrieval**: When a user asks a question, we use `sentence-transformers` (`all-MiniLM-L6-v2`) to create vector embeddings of the chunks and the user's question. We calculate cosine similarity to retrieve the single most relevant chunk of context.
4. **Generation (Question Answering)**: The retrieved chunk is passed alongside the user's question to a local HuggingFace QA model (`distilbert-base-cased-distilled-squad`). The model reads the specific context and extracts the exact answer.

### Execution
Run the CLI assistant script from the root directory:
```bash
python support_assistant/assistant.py
```
*(Note: The first run will automatically download the language models from HuggingFace).*

### Design Decisions
- **Decision**: `transformers` AutoModelForQuestionAnswering pipeline using `distilbert-base-cased-distilled-squad`.
- **Reasoning**: The project requires the use of strictly free services. By utilizing a local HuggingFace QA model, we can perform grounded document question-answering on the `zepto_policies.txt` text without relying on paid APIs, adhering strictly to the capstone constraints.

---

## Sample Results

Below are sample outputs generated by the analytics and support assistant modules:

### Analytics Pipeline Output
```text
Training Random Forest with GridSearchCV...
Random Forest MAE: 1.70 mins
Feature Importance (distance_km): 0.39
Feature Importance (weather_Rain): 0.40
Feature Importance (weather_Traffic): 0.21
```

### Support Assistant (RAG) Output
```text
Question: What happens if a delivery takes more than 20 minutes?
Answer: customers are eligible for a 10 % discount on their next order

Question: How much is the Zepto Pass?
Answer: Rs. 299 per month
```