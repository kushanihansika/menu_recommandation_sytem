from fastapi import FastAPI
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import re
import scipy.sparse

app = FastAPI()

# Load the Excel file into a pandas DataFrame
df_recipes = pd.read_excel('files/Breakfast.xlsx')  # Adjust the path based on your project structure

# Preprocess columns with non-numeric characters
df_recipes['calories'] = df_recipes['calories'].apply(lambda x: re.sub('[^0-9.]', '', str(x)))
df_recipes['fat'] = df_recipes['fat'].apply(lambda x: re.sub('[^0-9.]', '', str(x)))
df_recipes['carbs'] = df_recipes['carbs'].apply(lambda x: re.sub('[^0-9.]', '', str(x)))
df_recipes['protein'] = df_recipes['protein'].apply(lambda x: re.sub('[^0-9.]', '', str(x)))

# Convert columns to numeric
df_recipes[['calories', 'fat', 'carbs', 'protein']] = df_recipes[['calories', 'fat', 'carbs', 'protein']].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with zeros
df_recipes.fillna(0, inplace=True)

# Create a CountVectorizer to convert ingredients and description into a matrix of token counts
vectorizer = CountVectorizer()
ingredients_matrix = vectorizer.fit_transform(df_recipes['ingredients'])
description_matrix = vectorizer.fit_transform(df_recipes['description'])

# Combine numerical features for similarity calculation
numerical_features = df_recipes[['calories', 'fat', 'carbs', 'protein']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Combine ingredients and description matrices with scaled numerical features
combined_matrix = scipy.sparse.hstack([ingredients_matrix, description_matrix, scaled_features])

# Convert sparse matrices to dense arrays
combined_matrix_dense = combined_matrix.toarray()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(combined_matrix_dense, combined_matrix_dense)


@app.get("/recommend/")
def recommend_by_type_and_veg(recipe_type: str, veg_type: str) -> list[Any] | dict[str, str]:
    try:
        # Filter recipes by both type and veg type
        filtered_recipes = df_recipes[(df_recipes['type'] == recipe_type) & (df_recipes['veg_or_non_veg'] == veg_type)]

        # Get the cosine similarity scores for all recipes of the given type and veg type
        sim_scores = cosine_similarity(combined_matrix_dense[filtered_recipes.index], combined_matrix_dense)

        # Create a DataFrame with cosine similarity scores for recipes of the given type and veg type
        sim_df = pd.DataFrame(sim_scores, index=filtered_recipes.index, columns=df_recipes.index)

        # Get the indices of recommended recipes
        recommended_indices = sim_df.mean(axis=0).sort_values(ascending=False).index[:15]

        # Use a set to keep track of recommended recipes
        recommended_recipes_set = set()

        # Get the recommended recipes from the DataFrame
        for idx in recommended_indices:
            recipe_title = df_recipes.loc[idx, 'title']
            recommended_recipes_set.add(recipe_title)

        # Convert the set to a list
        recommended_recipes = list(recommended_recipes_set)

        return recommended_recipes

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
