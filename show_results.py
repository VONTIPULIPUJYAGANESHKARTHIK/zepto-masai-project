import os
import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Test Analytics Model
def run_analytics_test():
    print("\n--- ANALYTICS PIPELINE RESULTS ---")
    conn = sqlite3.connect('zepto.db')
    orders_df = pd.read_sql('SELECT * FROM orders', conn)
    conn.close()
    
    model_df = pd.get_dummies(orders_df, columns=['weather'], drop_first=True)
    X = model_df[['distance_km', 'weather_Rain', 'weather_Traffic']]
    y = model_df['delivery_time_mins']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest with GridSearchCV...")
    param_grid = {'n_estimators': [50], 'max_depth': [10]} # Simplified for quick run
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    rf_pred = best_rf.predict(X_test)
    
    print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_pred):.2f} mins")
    
    importances = best_rf.feature_importances_
    for name, imp in zip(X.columns, importances):
        print(f"Feature Importance ({name}): {imp:.2f}")

# 2. Test RAG Assistant
def run_assistant_test():
    print("\n--- GENAI ASSISTANT (RAG) RESULTS ---")
    # Add assistant path to sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, 'support_assistant'))
    
    import assistant
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from sentence_transformers import SentenceTransformer
    
    policy_path = os.path.join(base_dir, "support_assistant", "zepto_policies.txt")
    chunks = assistant.load_and_chunk_policies(policy_path)
    
    print("Loading models (this takes a few seconds)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    question = "What happens if a delivery takes more than 20 minutes?"
    print(f"\nQuestion: {question}")
    answer = assistant.get_answer_rag(question, chunks, embedder, tokenizer, model)
    print(f"Answer: {answer}")
    
    question2 = "How much is the Zepto Pass?"
    print(f"\nQuestion: {question2}")
    answer2 = assistant.get_answer_rag(question2, chunks, embedder, tokenizer, model)
    print(f"Answer: {answer2}")

if __name__ == "__main__":
    try:
        run_analytics_test()
        run_assistant_test()
    except Exception as e:
        print(f"Error occurred: {e}")
