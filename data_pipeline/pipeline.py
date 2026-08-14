import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import os

os.makedirs("data_pipeline/raw_data", exist_ok=True)
print("Starting to scrape books.toscrape.com for all 1000 books...")

# 1. Get all category links from the homepage
home_url = "http://books.toscrape.com/index.html"
home_resp = requests.get(home_url)
soup = BeautifulSoup(home_resp.content, 'html.parser')

category_list = soup.find('div', class_='side_categories').find('ul').find('li').find('ul').find_all('a')
categories = []
for a in category_list:
    cat_name = a.text.strip()
    cat_url = "http://books.toscrape.com/" + a['href']
    categories.append((cat_name, cat_url))

all_books = []

# 2. Iterate through every category and handle pagination
for cat_name, base_cat_url in categories:
    print(f"Scraping category: {cat_name}...")
    current_url = base_cat_url
    
    while True:
        resp = requests.get(current_url)
        if resp.status_code != 200:
            break
            
        page_soup = BeautifulSoup(resp.content, 'html.parser')
        book_articles = page_soup.find_all('article', class_='product_pod')
        
        for book in book_articles:
            title_element = book.find('h3').find('a')
            title = title_element['title'] if title_element and 'title' in title_element.attrs else title_element.text
            
            price_element = book.find('p', class_='price_color')
            price_gbp_text = price_element.text if price_element else ""
            
            rating_element = book.find('p', class_='star-rating')
            star_rating_text = rating_element['class'][1] if rating_element and len(rating_element['class']) > 1 else ""
            
            availability_element = book.find('p', class_='instock availability')
            availability_text = availability_element.text.strip() if availability_element else ""
            
            all_books.append({
                'title': title,
                'price_gbp': price_gbp_text,
                'star_rating': star_rating_text,
                'availability': availability_text,
                'category': cat_name
            })
            
        # Check for next page
        next_btn = page_soup.find('li', class_='next')
        if next_btn:
            next_url_part = next_btn.find('a')['href']
            # current_url is like .../category/books/mystery_3/index.html
            # next_url_part is like page-2.html
            current_url = current_url.rsplit('/', 1)[0] + '/' + next_url_part
        else:
            break

print(f"Successfully scraped {len(all_books)} books.")

# ----------------- CLEANING -----------------
print("Cleaning data...")
df = pd.DataFrame(all_books)

# 1. Strip currency symbol and convert to float
def clean_price(p):
    try:
        return float(re.sub(r'[^\d.]', '', p))
    except:
        return None
        
df['price_gbp'] = df['price_gbp'].apply(clean_price)

# 2. Convert star rating text to integer
rating_map = {
    'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5
}
df['rating'] = df['star_rating'].map(rating_map)

# 3. Parse availability to boolean
df['in_stock'] = df['availability'].apply(lambda x: 'in stock' in x.lower()).astype(int)

# 4. Handle messy rows
initial_len = len(df)
df = df.dropna(subset=['price_gbp', 'rating'])
print(f"Dropped {initial_len - len(df)} messy rows with missing prices or ratings.")

# 5. Convert GBP to INR (Fixed rate: 1 GBP = 105.50 INR)
FIXED_RATE_INR = 105.50
df['price_inr'] = df['price_gbp'] * FIXED_RATE_INR

# ----------------- DATABASE -----------------
print("Building relational SQLite database...")
categories_df = df[['category']].drop_duplicates().reset_index(drop=True)
categories_df['category_id'] = categories_df.index + 1
categories_df = categories_df.rename(columns={'category': 'category_name'})

df = df.merge(categories_df, left_on='category', right_on='category_name', how='left')
df['book_id'] = df.index + 1
books_df = df[['book_id', 'title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']]

conn = sqlite3.connect('zepto.db')
categories_df.to_sql('categories', conn, if_exists='replace', index=False)
books_df.to_sql('books', conn, if_exists='replace', index=False)
print("Data successfully loaded into zepto.db (tables: categories, books).")

# ----------------- SQL QUERIES -----------------
print("\\n--- Executing Required SQL Queries ---")
q1 = "SELECT title, price_gbp FROM books WHERE rating = 5 LIMIT 5"
print("\\n1. 5-Star Books (SELECT/WHERE):")
print(pd.read_sql(q1, conn))

q2 = "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5"
print("\\n2. Top 5 Most Expensive Books (ORDER BY):")
print(pd.read_sql(q2, conn))

q3 = "SELECT DISTINCT category_name FROM categories LIMIT 5"
print("\\n3. Sample Categories (DISTINCT):")
print(pd.read_sql(q3, conn))

q4 = "SELECT title, rating FROM books WHERE rating IN (1, 2) LIMIT 5"
print("\\n4. Poorly Rated Books (IN):")
print(pd.read_sql(q4, conn))

q5 = """
SELECT c.category_name, b.title, b.rating, b.price_gbp
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.rating >= 4
ORDER BY b.rating DESC, b.title ASC
LIMIT 5
"""
print("\\n5. High Rated Books with Categories (JOIN):")
join_sql_df = pd.read_sql(q5, conn)
print(join_sql_df)

# ----------------- PANDAS VALIDATION -----------------
print("\\n--- Validating SQL Join vs Pandas Merge ---")
pandas_books = pd.read_sql("SELECT * FROM books", conn)
pandas_cats = pd.read_sql("SELECT * FROM categories", conn)

merged_df = pd.merge(pandas_books, pandas_cats, on='category_id')
filtered_merged = merged_df[merged_df['rating'] >= 4]
sorted_merged = filtered_merged.sort_values(by=['rating', 'title'], ascending=[False, True])
final_pandas_df = sorted_merged[['category_name', 'title', 'rating', 'price_gbp']].head(5).reset_index(drop=True)

print("\\nPandas Merge Result:")
print(final_pandas_df)
print("\\nDo the SQL and Pandas results match perfectly?")
print(join_sql_df.equals(final_pandas_df))

conn.close()
print("\\nPipeline execution complete!")
