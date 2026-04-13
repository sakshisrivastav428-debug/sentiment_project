from flask import Flask, request
import pickle
from src.preprocess import clean_text

app = Flask(__name__)

model = pickle.load(open('src/model.pkl', 'rb'))
vectorizer = pickle.load(open('src/vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        text = request.form['tweet']
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            prediction = "Positive 😊"
        elif result == 0:
            prediction = "Negative 😡"
        else:
            prediction = "Neutral 😐"

    return f"""
    <h2>Twitter Sentiment Analysis</h2>
    <form method='post'>
        <input name='tweet' style='width:300px'/>
        <button type='submit'>Analyze</button>
    </form>
    <h3>{prediction}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)
