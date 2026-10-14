from flask import Flask, render_template_string, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from PreProcessing import clean_doc

app = Flask(__name__)

# Load the trained model and saved preprocessing objects
model = load_model("sentiment_ann.h5")
vectorizer = joblib.load("sentiment_vectorizer.joblib")
label_encoder = joblib.load("sentiment_label_encoder.joblib")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roman Urdu Sentiment AI</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0; min-height: 100vh; font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #eef2ff, #fdf2f8);
    display: flex; align-items: center; justify-content: center;
    padding: 24px; color: #1f2937;
}
.card {
    width: 100%; max-width: 680px; background: white;
    border-radius: 22px; padding: 36px;
    box-shadow: 0 18px 55px rgba(40, 30, 90, .12);
}
.tag { color: #6d28d9; font-weight: bold; letter-spacing: 1px; }
h1 { font-size: 34px; margin-bottom: 10px; }
.subtitle { color: #6b7280; line-height: 1.6; }
label { display: block; font-weight: bold; margin: 24px 0 10px; }
textarea {
    width: 100%; min-height: 150px; padding: 16px;
    border: 1px solid #d1d5db; border-radius: 12px;
    font: inherit; resize: vertical;
}
textarea:focus { outline: 2px solid #8b5cf6; border-color: transparent; }
button {
    margin-top: 16px; width: 100%; padding: 15px;
    background: #6d28d9; color: white; border: 0;
    border-radius: 12px; font-size: 16px; font-weight: bold;
    cursor: pointer;
}
button:hover { background: #5b21b6; }
.result {
    margin-top: 24px; padding: 20px; background: #f5f3ff;
    border-radius: 14px;
}
.result h2 { margin-top: 0; }
.score { margin: 12px 0; }
.track { height: 8px; background: #e5e7eb; border-radius: 8px; overflow: hidden; }
.fill { height: 100%; background: #8b5cf6; }
.error { color: #b91c1c; margin-top: 16px; }
footer { margin-top: 26px; color: #9ca3af; font-size: 12px; text-align: center; }
</style>
</head>
<body>
<main class="card">
    <div class="tag">AI • NLP • DEEP LEARNING</div>
    <h1>Roman Urdu Sentiment AI</h1>
    <p class="subtitle">Discover the sentiment behind Roman Urdu reviews using an Artificial Neural Network.</p>

    <form method="POST">
        <label for="review">Enter your review</label>
        <textarea id="review" name="review" maxlength="5000"
          placeholder="Example: ye movie bohat achi hai, mujhe pasand ayi..."
          required>{{ review }}</textarea>
        <button type="submit">✨ Predict Sentiment</button>
    </form>

    {% if error %}
      <p class="error">{{ error }}</p>
    {% endif %}

    {% if prediction %}
    <section class="result">
        <h2>Prediction: {{ prediction }}</h2>
        <p>Model output scores (not calibrated probabilities):</p>
        {% for item in scores %}
        <div class="score">
            <div>{{ item.label }} — {{ "%.2f"|format(item.score * 100) }}%</div>
            <div class="track">
                <div class="fill" style="width: {{ item.score * 100 }}%"></div>
            </div>
        </div>
        {% endfor %}
    </section>
    {% endif %}

    <footer>Roman Urdu Sentiment Analysis • Powered by TensorFlow and Keras</footer>
</main>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    review = ""
    prediction = None
    scores = []
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a review."
        else:
            cleaned_review = " ".join(clean_doc(review))

            if not cleaned_review:
                error = "Please enter a review containing recognizable words."
            else:
                features = vectorizer.transform([cleaned_review]).toarray()
                output = model.predict(features, verbose=0)[0]
                index = int(np.argmax(output))
                prediction = str(label_encoder.inverse_transform([index])[0])

                scores = [
                    {"label": str(label), "score": float(output[i])}
                    for i, label in enumerate(label_encoder.classes_)
                ]
                scores.sort(key=lambda item: item["score"], reverse=True)

    return render_template_string(
        HTML, review=review, prediction=prediction,
        scores=scores, error=error
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
