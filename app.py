from flask import Flask, render_template, request
import joblib
import string

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        news_text = request.form["news"]
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        prediction = "🟢 Real News" if result == 1 else "🔴 Fake News"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)