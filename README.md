# Movies-Recommendation-Model
A Streamlit-based interactive web application that recommends movies based on content similarity. This project uses natural language processing (NLP) techniques to compare movie metadata (overview, genre, cast, keywords, director) and recommends top similar movies. It also includes insightful visualizations using Matplotlib and Seaborn.


🚀 Features

- 🔍 Movie search and selection
- 🎯 Top 5 similar movie recommendations
- 🧪 Model comparison: TF-IDF vs CountVectorizer
- 📊 Visualizations:
  - Top genres bar chart
  - Vote count vs vote average bubble chart
  - Runtime distribution histogram
  - Cosine similarity heatmap


📁 Dataset

- `tmdb_5000_movies.csv` — Movie details  
- `tmdb_5000_credits.csv` — Cast and crew data  
(Source: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata))


🛠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, CountVectorizer, Cosine Similarity)
- Matplotlib & Seaborn (for graphs)
- Streamlit (for the web interface)
- Natural Language Processing (NLP)


## 🌐 Live Demo

[Click here to try it on Streamlit](https://movies-recommendation-model-app.streamlit.app/)
