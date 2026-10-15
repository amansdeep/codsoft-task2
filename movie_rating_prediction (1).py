"""
Movie Rating Prediction with Python
====================================
Build a model that predicts the rating of a movie based on features
like genre, director, and actors, using regression techniques.

Dataset: IMDb India Movies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("movies.csv", encoding="latin-1")

# ----------------------------------------------------------------------
# 2. CLEAN DATA
# ----------------------------------------------------------------------
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)
df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
df["Votes"] = (
    df["Votes"].astype(str).str.replace(",", "", regex=False)
    .str.extract(r"(\d+)").astype(float)
)

# Drop rows with no rating (nothing to train/evaluate on)
df = df.dropna(subset=["Rating"]).copy()

# Fill remaining gaps
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"] = df["Votes"].fillna(0)
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")

df["Main_Genre"] = df["Genre"].apply(lambda x: x.split(",")[0].strip())

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING (Genre, Director, Actors -> numeric)
#    Encode each category by its historical average rating, smoothed
#    toward the overall mean to avoid overfitting on rare categories.
# ----------------------------------------------------------------------
global_mean = df["Rating"].mean()

def encode(col, smoothing=5):
    stats = df.groupby(col)["Rating"].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return df[col].map(smoothed).fillna(global_mean)

df["Genre_Encoded"] = encode("Main_Genre")
df["Director_Encoded"] = encode("Director")
df["Actor1_Encoded"] = encode("Actor 1")
df["Actor2_Encoded"] = encode("Actor 2")
df["Actor3_Encoded"] = encode("Actor 3")

# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
features = [
    "Genre_Encoded", "Director_Encoded",
    "Actor1_Encoded", "Actor2_Encoded", "Actor3_Encoded",
    "Year", "Duration", "Votes",
]
X = df[features]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------
# 5. TRAIN REGRESSION MODEL
# ----------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 6. EVALUATE
# ----------------------------------------------------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.3f}")

# ----------------------------------------------------------------------
# 7. EXAMPLE PREDICTION
# ----------------------------------------------------------------------
sample = X_test.iloc[[0]]
print("\nSample prediction:", model.predict(sample)[0], " | Actual:", y_test.iloc[0])
