from flask import Flask, request, render_template
import pickle
import numpy as np
from model import preprocess, vectorize_text

app = Flask(__name__)

with open("model.pkl", "rb") as model_file:
    trained_lgb_model, vectorizer, tfidf_transformer, label_encoder = pickle.load(model_file)

# Mapping of numeric labels to text labels
label_mapping = {
    1: "religion",
    2: "age",
    3: "gender",
    4: "ethnicity",
    5: "not_cyberbullying"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text_input = request.form.get("tweet_text")
        cleaned_input = preprocess(text_input)
        input_vectorized = vectorize_text(cleaned_input, vectorizer, tfidf_transformer)

        # Use predict_proba to get probabilities
        probabilities = trained_lgb_model.predict(input_vectorized, raw_score=True)
        
        # Find the index of the class with the highest probability
        predicted_label_index = np.argmax(probabilities)
        
        # Map the index back to the original label
        predicted_label_numeric = label_encoder.inverse_transform([predicted_label_index])[0]

        # Map the numeric label to the corresponding text label
        predicted_label_text = label_mapping.get(predicted_label_numeric, "Unknown")

        return render_template("index.html", prediction_text=f"The sentiment is: {predicted_label_text}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
