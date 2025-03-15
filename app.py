from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the model and the text-classification pipeline
pipe = pipeline("text-classification", model="adanal/HelloWorld")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the list of sentences from the form
        sentences = request.form.get("sentences").splitlines()

        # Rank sentences based on sentiment (positive to negative)
        results = []
        for sentence in sentences:
            result = pipe(sentence)
            sentiment = result[0]['label']
            score = result[0]['score']
            results.append((sentence, sentiment, score))

        # Sort by the score (positive -> negative)
        ranked_results = sorted(results, key=lambda x: x[2], reverse=True)

        return render_template("index.html", sentences=ranked_results)
    
    return render_template("index.html", sentences=None)

if __name__ == "__main__":
    app.run(debug=True)
