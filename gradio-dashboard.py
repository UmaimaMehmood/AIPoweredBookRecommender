import os
import pandas as pd
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Load books data and inspect structure
csv_file = "books_with_emotions.csv"
if os.path.exists(csv_file):
    books = pd.read_csv(csv_file)
    print(f"Loaded {len(books)} books")
    print(f"Available columns: {list(books.columns)}")
else:
    print(f"CSV file {csv_file} not found!")
    exit()

# Try to identify the correct column names
title_col = None
author_col = None
description_col = None
emotions_col = None

# Common variations of column names
title_variations = ['title', 'Title', 'book_title', 'name', 'Name']
author_variations = ['author', 'Author', 'authors', 'writer', 'by']
description_variations = ['description', 'Description', 'summary', 'plot', 'synopsis', 'overview']
emotions_variations = ['emotions', 'emotion', 'feelings', 'mood', 'sentiment', 'tags', 'genres']

# Find matching columns
for col in books.columns:
    if col in title_variations:
        title_col = col
    elif col in author_variations:
        author_col = col
    elif col in description_variations:
        description_col = col
    elif col in emotions_variations:
        emotions_col = col

print(f"Detected columns - Title: {title_col}, Author: {author_col}, Description: {description_col}, Emotions: {emotions_col}")

# If we can't find standard columns, use the first few columns
if not title_col and len(books.columns) > 0:
    title_col = books.columns[0]
    print(f"Using first column as title: {title_col}")

if not description_col:
    # Look for the longest text column (likely description)
    text_lengths = {}
    for col in books.columns:
        if books[col].dtype == 'object':  # String columns
            avg_length = books[col].astype(str).str.len().mean()
            text_lengths[col] = avg_length
    
    if text_lengths:
        description_col = max(text_lengths, key=text_lengths.get)
        print(f"Using longest text column as description: {description_col}")

# Create safe accessors
def safe_get(row, column, default="N/A"):
    if column and column in books.columns:
        return str(row[column]) if pd.notna(row[column]) else default
    return default

print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all book data
print("Creating embeddings...")
book_texts = []
for _, book in books.iterrows():
    # Combine available text fields
    text_parts = []
    
    if title_col:
        text_parts.append(safe_get(book, title_col))
    
    if author_col:
        author_text = safe_get(book, author_col)
        if author_text != "N/A":
            text_parts.append(f"by {author_text}")
    
    if description_col:
        desc_text = safe_get(book, description_col)
        if desc_text != "N/A":
            text_parts.append(desc_text)
    
    if emotions_col:
        emotions_text = safe_get(book, emotions_col)
        if emotions_text != "N/A":
            text_parts.append(f"Themes: {emotions_text}")
    
    combined_text = ". ".join(text_parts)
    book_texts.append(combined_text)

# Generate embeddings
book_embeddings = model.encode(book_texts)
print(f"Generated embeddings for {len(book_embeddings)} books")

def get_book_recommendations(mood_query, num_recommendations=5):
    """
    Get book recommendations based on mood/emotion query using cosine similarity
    """
    try:
        if not mood_query.strip():
            return []
        
        # Encode the user's mood query
        query_embedding = model.encode([mood_query])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, book_embeddings)[0]
        
        # Get top recommendations
        top_indices = similarities.argsort()[-num_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            book = books.iloc[idx]
            recommendations.append({
                'title': safe_get(book, title_col),
                'author': safe_get(book, author_col),
                'description': safe_get(book, description_col),
                'emotions': safe_get(book, emotions_col),
                'similarity_score': round(similarities[idx], 3)
            })
        
        return recommendations
    
    except Exception as e:
        print(f"Error in recommendations: {e}")
        # Fallback: return first few books
        recommendations = []
        for _, book in books.head(num_recommendations).iterrows():
            recommendations.append({
                'title': safe_get(book, title_col),
                'author': safe_get(book, author_col),
                'description': safe_get(book, description_col),
                'emotions': safe_get(book, emotions_col),
                'similarity_score': 0.0
            })
        return recommendations

def format_recommendations(recommendations):
    """
    Format recommendations for display
    """
    if not recommendations:
        return "No recommendations found. Please try a different mood or emotion."
    
    formatted = "# Book Recommendations\n\n"
    for i, book in enumerate(recommendations, 1):
        formatted += f"## {i}. **{book['title']}**\n"
        if book['author'] != "N/A":
            formatted += f"*by {book['author']}*\n\n"
        if book['description'] != "N/A":
            formatted += f"**Description:** {book['description'][:300]}{'...' if len(book['description']) > 300 else ''}\n\n"
        if book['emotions'] != "N/A":
            formatted += f"**Themes/Emotions:** {book['emotions']}\n\n"
        if book['similarity_score'] > 0:
            formatted += f"**Match Score:** {book['similarity_score']:.1%}\n\n"
        formatted += "---\n\n"
    
    return formatted

def recommend_books_interface(mood, num_books):
    """
    Interface function for Gradio
    """
    if not mood.strip():
        return "Please enter a mood or emotion to get recommendations."
    
    try:
        recommendations = get_book_recommendations(mood, int(num_books))
        return format_recommendations(recommendations)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
print("Creating Gradio interface...")
iface = gr.Interface(
    fn=recommend_books_interface,
    inputs=[
        gr.Textbox(
            label="What mood are you in?", 
            placeholder="e.g., happy, sad, adventurous, romantic, mysterious, dystopian, magical...",
            lines=2
        ),
        gr.Slider(
            minimum=1, 
            maximum=10, 
            value=5, 
            step=1, 
            label="Number of recommendations"
        )
    ],
    outputs=gr.Markdown(label="Your Personalized Recommendations"),
    title="AI Powered Book Recommender",
    description=f"""
    Enter your current mood, desired emotions, or themes you want to explore in a book. 
    Our AI will analyze your input and recommend books from a database of {len(books)} books using semantic similarity!
    
    **Available data fields:** {', '.join([col for col in [title_col, author_col, description_col, emotions_col] if col])}
    """,
    theme="soft",
    examples=[
        ["I want something uplifting and hopeful", 5],
        ["Looking for a thrilling adventure", 3],
        ["I'm feeling nostalgic and romantic", 4],
        ["Need something dark and psychological", 3],
        ["Want to read about personal growth", 5]
    ]
)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Book Recommender System...")
    print(f"Loaded {len(books)} books in database")
    print("="*50)
    
    try:
        # Install scikit-learn if not available
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("Installing scikit-learn...")
            os.system("pip install scikit-learn")
            from sklearn.metrics.pairwise import cosine_similarity
        
        iface.launch(
            share=True, 
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}")