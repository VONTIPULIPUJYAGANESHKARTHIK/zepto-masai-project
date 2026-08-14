import sqlite3
import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta

def generate_mock_raw_data(output_dir):
    """Simulates pulling raw data from an external source with anomalies (dirty data)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Scraping raw Zepto data (simulating dirty data)...")
    
    # 1. Generate Mock Products (Catalog)
    products = [
        {"product_id": 1, "name": "Milk 1L", "category": "Dairy", "price": 60},
        {"product_id": 2, "name": "Bread", "category": "Bakery", "price": 40},
        {"product_id": 3, "name": "Eggs 12 pcs", "category": "Dairy", "price": 80},
        {"product_id": 4, "name": "Apples 1kg", "category": "Fruits", "price": 150},
        {"product_id": 5, "name": "Potato 1kg", "category": "Vegetables", "price": 30},
        {"product_id": 6, "name": "Cola 2L", "category": "Beverages", "price": 90},
        {"product_id": 7, "name": "Error Product", "category": "Unknown", "price": -50} # Anomaly: Negative price
    ]
    
    # 2. Generate Mock Customers
    customers = []
    for i in range(1, 101):
        customers.append({
            "customer_id": i,
            "name": f"Customer_{i}",
            "signup_date": (datetime.now() - timedelta(days=random.randint(10, 365))).strftime("%Y-%m-%d"),
            "premium_member": random.choice([True, False])
        })
        
    # Duplicate a customer to simulate dirty data
    customers.append(customers[0])
        
    # 3. Generate Mock Orders
    orders = []
    for i in range(1, 1001):
        order_time = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        distance_km = round(random.uniform(0.5, 5.0), 2)
        weather = random.choice(["Clear", "Rain", "Traffic"])
        
        # Calculate delivery time realistically based on distance and weather
        delivery_time_mins = 10 + (distance_km * 2.5)
        if weather == "Rain":
            delivery_time_mins += random.uniform(8, 18)
        elif weather == "Traffic":
            delivery_time_mins += random.uniform(5, 12)
            
        delivery_time_mins = round(delivery_time_mins, 1)
        
        # Sometime we have missing data (to simulate raw data issues)
        if random.random() < 0.05:
            weather = None
            
        # Simulate realistic amount based on distance loosely
        total_amount = round(random.uniform(50, 500) + (distance_km * 10), 2)
        
        orders.append({
            "order_id": i,
            "customer_id": random.randint(1, 100),
            "order_time": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "distance_km": distance_km,
            "weather": weather,
            "delivery_time_mins": delivery_time_mins,
            "total_amount": total_amount
        })
        
    # Introduce anomalies in orders
    orders.append(orders[10]) # Duplicate order
    orders[15]['total_amount'] = -100 # Negative amount anomaly
        
    # Save to raw JSON files
    with open(os.path.join(output_dir, "raw_products.json"), "w") as f:
        json.dump(products, f)
    with open(os.path.join(output_dir, "raw_customers.json"), "w") as f:
        json.dump(customers, f)
    with open(os.path.join(output_dir, "raw_orders.json"), "w") as f:
        json.dump(orders, f)
        
    print("Raw dirty data generated successfully.")

def clean_and_store_data(raw_dir, db_path):
    """Cleans the raw data (deduplication, anomaly removal) and stores it in SQLite."""
    print("Cleaning data and building relational store...")
    
    # Load raw data
    with open(os.path.join(raw_dir, "raw_products.json"), "r") as f:
        products_df = pd.DataFrame(json.load(f))
    with open(os.path.join(raw_dir, "raw_customers.json"), "r") as f:
        customers_df = pd.DataFrame(json.load(f))
    with open(os.path.join(raw_dir, "raw_orders.json"), "r") as f:
        orders_df = pd.DataFrame(json.load(f))
        
    # --- Data Cleaning ---
    
    # 1. Deduplication
    initial_counts = (len(products_df), len(customers_df), len(orders_df))
    products_df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
    customers_df.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)
    orders_df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
    
    # 2. Anomaly Removal (Negative values)
    products_df = products_df[products_df['price'] >= 0]
    orders_df = orders_df[orders_df['total_amount'] >= 0]
    
    # 3. Missing Value Imputation
    orders_df['weather'] = orders_df['weather'].fillna('Clear')
    
    # 4. Type Enforcement
    orders_df['order_time'] = pd.to_datetime(orders_df['order_time'])
    
    final_counts = (len(products_df), len(customers_df), len(orders_df))
    print(f"Cleaning complete. Removed {(initial_counts[0]-final_counts[0])} products, " 
          f"{(initial_counts[1]-final_counts[1])} customers, "
          f"{(initial_counts[2]-final_counts[2])} orders.")
    
    # --- Storage ---
    conn = sqlite3.connect(db_path)
    
    # Write to relational store
    products_df.to_sql("products", conn, if_exists="replace", index=False)
    customers_df.to_sql("customers", conn, if_exists="replace", index=False)
    orders_df.to_sql("orders", conn, if_exists="replace", index=False)
    
    conn.close()
    print(f"Data successfully cleaned and stored in {db_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "raw_data")
    
    # Run pipeline
    generate_mock_raw_data(raw_data_dir)
    
    db_file = os.path.join(base_dir, "..", "zepto.db")
    clean_and_store_data(raw_data_dir, db_file)
