import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from preprocess import clean_text

df = pd.read_csv('../data/tweets.csv')

label_map = {"negative": 0, "positive": 1, "neutral": 2}
df['label'] = df['label'].map(label_map)

df['clean_tweet'] = df['tweet'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
