

import streamlit as st

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("A content-based movie recommender using TMDB 5000 dataset")

# Load data

#!/usr/bin/env python
# coding: utf-8

# # MOVIE RECOMMENDATION SYSTEM

# In[2]:


import pandas as pd


# In[5]:


movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")


# In[7]:


movies_df.head(1)


# In[8]:


credits_df.head(1)


# ## DATA PREPARATION: Merging and Cleaning

# ### Merging

# In[94]:


merged_df = movies_df.merge(credits_df, left_on='id',right_on='movie_id')


# In[10]:


merged_df.head()


# In[11]:


selected_columns = ['title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']
movies_cleaned_df = merged_df[selected_columns].copy()


# In[12]:


movies_cleaned_df.head(1)


# ### Cleaning

# In[13]:


movies_cleaned_df.rename(columns={'title_x': 'title'}, inplace=True)


# In[15]:


movies_cleaned_df.head(1)


# In[16]:


movies_cleaned_df.dropna(subset=['overview'], inplace=True)


# In[17]:


movies_cleaned_df.isnull().sum()


# In[18]:


print(movies_cleaned_df.info())


# In[22]:


movies_cleaned_df.head()


# ## DATA VISUALIZATION

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[27]:


def get_genre_names(text):
    return [i['name'] for i in ast.literal_eval(text)]
all_genres = []
for genres in movies_cleaned_df['genres']:
    all_genres.extend(get_genre_names(genres))


# In[28]:


genre_counts = Counter(all_genres).most_common(10)
genre_df = pd.DataFrame(genre_counts,columns=['Genre','Count'])


# ### What It Shows:

# #### 1. Most frequent genres

# In[29]:


fig1, ax1 = plt.subplots(figsize=(10,5))
sns.barplot(data=genre_df, x='Genre', y='Count', palette='coolwarm', ax=ax1)
ax1.set_title("Top 10 Most Common Genres")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
fig1.tight_layout()
st.pyplot(fig1)


# #### 2. Relationship between votes and ratings

# In[31]:


fig2, ax2 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=movies_df, x='vote_count',y='vote_average', size='popularity', hue='popularity', palette='viridis', sizes=(20,200), alpha=0.6, ax=ax2)
ax2.set_title("Vote Count vs Vote Average")
ax2.set_xlabel("Vote Count")
ax2.set_ylabel("Vote Average")
ax2.set_xscale('log')
fig2.tight_layout()
st.pyplot(fig2)


# #### 3. Length of movies

# In[33]:


fig3, ax3 = plt.subplots(figsize=(10,5))
sns.histplot(movies_df['runtime'].dropna(), bins=30, kde=True, color='skyblue', ax=ax3)
ax3.set_title("Distribution of Movie Runtimes")
ax3.set_xlabel("Runtime (minutes)")
ax3.set_ylabel("Number of Movies")
fig3.tight_layout()
st.pyplot(fig3)


# #### 4. Similarity between movie overview

# In[34]:


sample_df = movies_cleaned_df.head(10).copy()


# In[40]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(sample_df['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

fig4, ax4 = plt.subplots(figsize=(10,8))
sns.heatmap(cosine_sim, annot=True, cmap='coolwarm', xticklabels=sample_df['title'], yticklabels=sample_df['title'], ax=ax4)
ax4.set_title("Cosine Similarity Heatmap of Movies Overviews (Top 10 Movies)")
ax4.tick_params(axis='x', rotation=90)
ax4.tick_params(axis='y', rotation=0)
fig4.tight_layout()
st.pyplot(fig4)



# ## FEATURE ANALYSIS

# In[41]:


def extract_names(text):
    try:
        return [item['name'].replace(" ","") for item in ast.literal_eval(text)]
    except:
        return []


# In[42]:


def extract_cast(text):
    try:
        return [item['name'].replace(" ","") for item in ast.literal_eval(text)[:3]]
    except:
        return []


# In[43]:


def extract_director(text):
    try:
        crew_list = ast.literal_eval(text)
        for item in crew_list:
            if item['job'] == 'Director':
                return [item['name'].replace(" ","")]
        return []
    except:
        return []


# In[44]:


movies_cleaned_df['genres'] = movies_cleaned_df['genres'].apply(extract_names)
movies_cleaned_df['keywords'] = movies_cleaned_df['keywords'].apply(extract_names)
movies_cleaned_df['cast'] = movies_cleaned_df['cast'].apply(extract_cast)
movies_cleaned_df['crew'] = movies_cleaned_df['crew'].apply(extract_director)


# In[45]:


movies_cleaned_df['overview'] = movies_cleaned_df['overview'].apply(lambda x:x.split())


# In[47]:


movies_cleaned_df['tags'] = movies_cleaned_df['overview'] + movies_cleaned_df['genres'] + movies_cleaned_df['keywords'] + movies_cleaned_df['cast'] + movies_cleaned_df['crew']


# In[48]:


movies_cleaned_df['tags'] = movies_cleaned_df['tags'].apply(lambda x: " ".join(x))


# In[49]:


movies_cleaned_df.head()


# In[50]:


final_df = movies_cleaned_df[['title', 'tags']]


# In[51]:


final_df.head()


# ## MODEL BUILDING - Content-Based Filtering

# In[53]:


tfidf = TfidfVectorizer(stop_words='english',max_features=5000)
tfidf_matrix = tfidf.fit_transform(final_df['tags'])


# In[54]:


similarity = cosine_similarity(tfidf_matrix)


# In[57]:


def recommend(movie_title):
    if movie_title not in final_df['title'].values:
        print("Movies not found in the database.")
        return
    index = final_df[final_df['title'] == movie_title].index[0]

    distances = list(enumerate(similarity[index]))

    recommended_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    print(f"Top 5 movies similar to '{movie_title}':")
    for i in recommended_movies:
        print(final_df.iloc[i[0]]['title'])


# In[58]:


recommend('Batman Begins')


# ## HYPERPARAMETER TUNING AND MODEL COMPARISON

# In[77]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[87]:


tfidf = TfidfVectorizer(max_features = 5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(final_df['tags'])
cosine_sim_tfidf = cosine_similarity(tfidf_matrix)


# In[88]:


cv = CountVectorizer(max_features=5000, stop_words='english')
count_matrix = cv.fit_transform(final_df['tags'])
cosine_sim_cv = cosine_similarity(count_matrix)


# In[89]:


def compare_models(movie_title):
    if movie_title not in final_df['title'].values:
        print("Movie not found in database.")
        return

    index = final_df[final_df['title'] == movie_title].index[0]
    print(f"\n TF-IDF Recommendations for '{movie_title}':")
    distances_tfidf = list(enumerate(cosine_sim_tfidf[index]))
    top_tfidf = sorted(distances_tfidf, key=lambda x: x[1], reverse=True)[1:6]

    for i in top_tfidf:
        print(f"-> {final_df.iloc[i[0]].title}")

    print(f"\nCountVectorizer Recommendations for '{movie_title}':")
    distances_cv = list(enumerate(cosine_sim_cv[index]))
    top_cv = sorted(distances_cv, key=lambda x: x[1], reverse=True)[1:6]

    for i in top_cv:
        print(f"->{final_df.iloc[i[0]].title}")


# In[90]:


compare_models('Avatar')


# In[ ]:






st.header("Try the Recommender")

movie_list = final_df['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend (TF-IDF)"):
    index = final_df[final_df['title'] == selected_movie].index[0]
    distances = list(enumerate(similarity[index]))
    recommended_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    st.subheader(f"Top 5 movies similar to '{selected_movie}':")
    for i in recommended_movies:
        st.write(final_df.iloc[i[0]]['title'])

if st.button("Compare TF-IDF vs CountVectorizer"):
    index = final_df[final_df['title'] == selected_movie].index[0]

    st.subheader("TF-IDF Recommendations")
    distances_tfidf = list(enumerate(cosine_sim_tfidf[index]))
    top_tfidf = sorted(distances_tfidf, key=lambda x: x[1], reverse=True)[1:6]
    for i in top_tfidf:
        st.write(f"→ {final_df.iloc[i[0]].title}")

    st.subheader("CountVectorizer Recommendations")
    distances_cv = list(enumerate(cosine_sim_cv[index]))
    top_cv = sorted(distances_cv, key=lambda x: x[1], reverse=True)[1:6]
    for i in top_cv:
        st.write(f"→ {final_df.iloc[i[0]].title}")
