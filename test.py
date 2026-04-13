import pickle
from preprocess import clean_text

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Negative"
    else:
        return "Neutral"

print(predict_sentiment("I love this phone!"))
print(predict_sentiment("This is terrible"))
