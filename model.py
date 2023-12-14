import string
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.exceptions import NotFittedError

# Add this line to define stop_words
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    text = ' '.join(word for word in text.split() if len(word) < 14)
    return text

def train_lightgbm_model(X_train, y_train):
    params = {"objective": "multiclass",
              "num_class": 5,
              "metric": "multi_logloss",
              "boosting_type": "gbdt",
              "num_leaves": 31,
              "learning_rate": 0.05,
              "feature_fraction": 0.9,
              "bagging_fraction": 0.8,
              "bagging_freq": 5,
              "verbose": 0
              }

    train_data = lgb.Dataset(X_train, label=y_train)
    num_rounds = 100
    lgb_model = lgb.train(params, train_data, num_boost_round=num_rounds)
    return lgb_model

def vectorize_text(text, vectorizer, transformer):
    try:
        text_vectorized = vectorizer.transform([text])
        text_tfidf = transformer.transform(text_vectorized)
        return text_tfidf
    except NotFittedError:
        raise ValueError("Vocabulary not fitted or provided")

if __name__ == "__main__":
    df = pd.read_csv("cyberbullying_tweets.csv")

    df.drop(df[df['cyberbullying_type'] == 'other_cyberbullying'].index, inplace=True)
    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
    df["sentiment"].replace({"religion": 1, "age": 2, "gender": 3, "ethnicity": 4, "not_cyberbullying": 5}, inplace=True)

    sentiments = ["religion", "age", "gender", "ethnicity", "not bullying"]

    texts_cleaned = [preprocess(t) for t in df.text]

    df['text_clean'] = texts_cleaned

    df.drop_duplicates("text_clean", inplace=True)

    text_len = [len(text.split()) for text in df.text_clean]

    df['text_len'] = text_len

    df = df[(df['text_len'] > 3) & (df['text_len'] < 100)]

    X = df['text_clean']
    y = df['sentiment']

    # Use LabelEncoder to encode class labels starting from 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)


    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42)

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_vectorized)

    trained_lgb_model = train_lightgbm_model(X_train_tfidf, y_train)

    with open("model.pkl", "wb") as model_file:
        pickle.dump((trained_lgb_model, vectorizer, tfidf_transformer, label_encoder), model_file)
