# import re
# import pickle
# import numpy as np
# from nltk.corpus import stopwords
# from googletrans import Translator
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from flask import Flask, render_template, request

# translator = Translator()
# sw = stopwords.words('bengali')
# wordnet_lemmatizer = WordNetLemmatizer()

# # Flask app initialization
# app = Flask(__name__)

# # Load the model and vectorizers from the saved pickle file
# with open("model.pkl", "rb") as model_file:
#     stacking_model, count_vectorizer, tf_transformer, y = pickle.load(model_file)

# # Mapping of numeric labels to text labels
# label_mapping = {
#     0: "Not Bully",
#     1: "Troll",
#     2: "Sexual",
#     3: "Religious",
#     4: "Threat"
# }

# # Preprocess the Bangla text
# def preprocess_bangla_text(text):
#     txt = re.sub(r'http\S+', '', text)
#     txt = re.sub(r'[!@#$%^&*?><,./\-+`~|:);(❤}{]', '', txt)

#     text_tokens = word_tokenize(txt)

#     remove_sw = [word for word in text_tokens if not word in sw]
#     un_items = np.unique(remove_sw)
#     r_sw = [wordnet_lemmatizer.lemmatize(w) for w in un_items]
#     bn_tokens = []

#     def tanslate_bengali(r_sw):
#         for i in range(len(r_sw)):
#             bn_tokens.append(translator.translate(r_sw[i], dest='bn').text)
#         return bn_tokens

#     bn_token = r_sw
#     preprocessed_text = ' '.join(bn_token)

#     return preprocessed_text

# # Vectorize the preprocessed Bangla text
# def vectorize_bangla_text(text):
#     counts = count_vectorizer.transform([text])
#     tfidf_vectorized_text = tf_transformer.transform(counts).toarray()
#     return tfidf_vectorized_text

# # Define the main route
# @app.route('/')
# def home():
#     return render_template('index.html')  # You can create an HTML file for the input form

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         bangla_text = request.form['text']
#         preprocessed_text = preprocess_bangla_text(bangla_text)
#         vectorized_text = vectorize_bangla_text(preprocessed_text)
#         vectorized_text = vectorized_text.reshape(1, -1)
#         predicted_class = stacking_model.predict(vectorized_text)
#         # Map the numeric label to the corresponding text label
#         predicted_label = label_mapping.get(predicted_class[0], "Unknown")
#         return render_template('index.html', prediction=predicted_label)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Ensure that the model module is in the same directory or update the import statement accordingly
from model import preprocess, vectorize_text

app = Flask(__name__)

# Load the first model and vectorizers from the saved pickle file
with open("model2.pkl", "rb") as model_file2:
    trained_lgb_model, vectorizer, tfidf_transformer2, label_encoder = pickle.load(model_file2)

# Load the second model and vectorizers from the saved pickle file
with open("model.pkl", "rb") as model_file:
    stacking_model, count_vectorizer, tf_transformer, y = pickle.load(model_file)

# Mapping of numeric labels to text labels for the first model
label_mapping1 = {
    1: "Religion",
    2: "Age",
    3: "Gender",
    4: "Ethnicity",
    5: "Not_cyberbullying"
}

# Mapping of numeric labels to text labels for the second model
label_mapping2 = {
    0: "Not Bully",
    1: "Troll",
    2: "Sexual",
    3: "Religious",
    4: "Threat"
}

translator = Translator()
sw = stopwords.words('bengali')
wordnet_lemmatizer = WordNetLemmatizer()

# Preprocess the Bangla text
def preprocess_bangla_text(text):
    txt = re.sub(r'http\S+', '', text)
    txt = re.sub(r'[!@#$%^&*?><,./\-+`~|:);(❤}{]', '', txt)

    text_tokens = word_tokenize(txt)

    remove_sw = [word for word in text_tokens if not word in sw]
    un_items = np.unique(remove_sw)
    r_sw = [wordnet_lemmatizer.lemmatize(w) for w in un_items]
    bn_tokens = []

    def translate_bengali(r_sw):
        for i in range(len(r_sw)):
            bn_tokens.append(translator.translate(r_sw[i], dest='bn').text)
        return bn_tokens

    bn_token = r_sw
    preprocessed_text = ' '.join(bn_token)

    return preprocessed_text

# Vectorize the preprocessed Bangla text
def vectorize_bangla_text(text):
    counts = count_vectorizer.transform([text])
    tfidf_vectorized_text = tf_transformer.transform(counts).toarray()
    return tfidf_vectorized_text

# Define the main route
@app.route('/')
def home():
    return render_template('index.html')  # You can create an HTML file for the input form

# Define the combined prediction route
@app.route('/predict', methods=['POST'])
def predict():
    predicted_label = "Unknown"  # Initialize the variable

    if request.method == 'POST':
        if 'tweet_text' in request.form:
            # For the first model
            text_input = request.form.get("tweet_text")
            cleaned_input = preprocess(text_input)
            input_vectorized = vectorize_text(cleaned_input, vectorizer, tfidf_transformer2)
            probabilities = trained_lgb_model.predict(input_vectorized, raw_score=True)
            predicted_label_index = np.argmax(probabilities)
            predicted_label_numeric = label_encoder.inverse_transform([predicted_label_index])[0]
            predicted_label = label_mapping1.get(predicted_label_numeric, "Unknown")
            return render_template('index.html', prediction_e=predicted_label)
        elif 'text' in request.form:
            # For the second model
            bangla_text = request.form['text']
            preprocessed_text = preprocess_bangla_text(bangla_text)
            vectorized_text = vectorize_bangla_text(preprocessed_text)
            vectorized_text = vectorized_text.reshape(1, -1)
            predicted_class = stacking_model.predict(vectorized_text)
            predicted_label = label_mapping2.get(predicted_class[0], "Unknown")
            return render_template('index.html', prediction_b=predicted_label)
    

if __name__ == '__main__':
    app.run(debug=True)
