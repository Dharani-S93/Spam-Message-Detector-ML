import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def load_data():
    try:
        df = pd.read_csv("spam.csv", encoding='latin-1')
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].copy()
            df.columns = ['label', 'text']
        return df
    except:
        print("Using backup dataset...")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        return pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def train_model():
    df = load_data()
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/spam_model.joblib')
    joblib.dump(tfidf, 'model/tfidf_vectorizer.joblib')
    
    return model, tfidf

def predict_spam(model, tfidf, text):
    cleaned = preprocess_text(text)
    features = tfidf.transform([cleaned])
    pred = model.predict(features)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"

def main():
    print("Loading spam detector...")
    
    try:
        model = joblib.load('model/spam_model.joblib')
        tfidf = joblib.load('model/tfidf_vectorizer.joblib')
    except:
        print("Training new model...")
        model, tfidf = train_model()
    
    print("\nSPAM DETECTOR READY")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter message: ").strip()
        if text.lower() == 'quit':
            break
            
        if not text:
            print("Please enter a message")
            continue
            
        result = predict_spam(model, tfidf, text)
        print(f"Result: {result}\n")

if __name__ == "__main__":
    main()