import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Zepto Data & AI Capstone: Analytics Module
This notebook profiles the scraped dataset and builds predictive models, acting as the analytics pipeline."""

code1 = """import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')"""

text2 = """## 1. Data Loading
We load the `books` and `categories` tables from the SQLite database."""

code2 = """conn = sqlite3.connect('../zepto.db')
books_df = pd.read_sql("SELECT * FROM books", conn)
cats_df = pd.read_sql("SELECT * FROM categories", conn)
df = pd.merge(books_df, cats_df, on='category_id')
conn.close()

df.head()"""

text3 = """## 2. Exploratory Data Analysis (EDA)
Let's visualize the distribution of prices and the impact of ratings."""

code3 = """plt.figure(figsize=(10, 5))
sns.histplot(df['price_gbp'], bins=20, kde=True, color='purple')
plt.title('Distribution of Book Prices (GBP)')
plt.xlabel('Price (GBP)')
plt.ylabel('Count')
plt.show()"""

code4 = """plt.figure(figsize=(10, 5))
sns.boxplot(x='rating', y='price_gbp', data=df, palette='Set2')
plt.title('Price Distribution by Star Rating')
plt.xlabel('Star Rating (1-5)')
plt.ylabel('Price (GBP)')
plt.show()"""

text4 = """## 3. Predictive Modeling
We will train a model to predict the **price** of a book based on its rating and category.
First, we encode the categorical variables."""

code4_2 = """# Encode category
df_encoded = pd.get_dummies(df, columns=['category_name'], drop_first=True)

X = df_encoded.drop(columns=['book_id', 'title', 'price_gbp', 'price_inr', 'category_id'])
y = df_encoded['price_gbp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""

text5 = """### Baseline Model: Linear Regression"""

code5 = """lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print(f"Linear Regression MAE: £{mean_absolute_error(y_test, preds):.2f}")"""

text6 = """### Advanced Model: Random Forest with GridSearchCV"""

code6 = """rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
rf_preds = best_rf.predict(X_test)

print(f"Best RF Params: {grid.best_params_}")
print(f"Random Forest MAE: £{mean_absolute_error(y_test, rf_preds):.2f}")"""

text7 = """### Feature Importances"""

code7 = """importances = best_rf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feat_df.head(10), palette='viridis')
plt.title('Top 10 Feature Importances for Price Prediction')
plt.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4_2),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(text7),
    nbf.v4.new_code_cell(code7)
]

with open('analytics/analytics_end_to_end.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
