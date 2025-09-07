📚 AI Powered Book Recommender

An intelligent book recommendation system powered by AI and NLP techniques. This project combines data cleaning, semantic search, classification, sentiment analysis, and a simple web app to help users discover books based on meaning, mood, and categories rather than just keywords.

🚀 Features

✨ Text Data Cleaning

Preprocessed raw text data for better downstream tasks.

📒 Code: data-exploration.ipynb

🔎 Semantic (Vector) Search

Built a vector database to find books similar to a natural language query.

Example: "a book about a person seeking revenge" → returns similar recommendations.

📒 Code: vector-search.ipynb

📖 Text Classification (Fiction / Non-Fiction)

Classified books using zero-shot classification with LLMs.

Enables users to filter books by type.

📒 Code: text-classification.ipynb

😊 Sentiment & Emotion Analysis

Extracted emotions and tone (suspenseful, joyful, sad, etc.) from book descriptions.

Allows users to sort by mood.

📒 Code: sentiment-analysis.ipynb

🌐 Web Application with Gradio

Built an interactive web app for users to search & get recommendations.

📂 Code: gradio-dashboard.py

🛠️ Tech Stack

Programming Language: Python 3.11

Libraries & Frameworks:

🐼 Pandas

📊 Matplotlib

🎨 Seaborn

🔑 python-dotenv

🤖 Transformers

🌐 Gradio

🧠 LangChain Community

🗄️ Chroma

🐍 Jupyter Notebook

⚡ ipywidgets

📥 kagglehub


📊 Data

Book datasets are downloaded from Kaggle.

Processed datasets (cleaned & labeled versions) are included in this repo for quick use.

