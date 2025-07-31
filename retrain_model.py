import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the full datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.dropna(subset=["text"])             # Remove missing text
df = df[["text", "label"]]

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(f"Accuracy = {accuracy_score(y_test, y_pred):.2f}")

with open("fake_news_model.pkl", "wb") as m:
    pickle.dump(model, m)
with open("tfidf_vectorizer.pkl", "wb") as v:
    pickle.dump(vectorizer, v)

print("✅ Done retraining with full dataset!")
