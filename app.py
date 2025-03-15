from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME ="YOUR MODEL NAME"
# Load the model and the text-classification pipeline
pipe = pipeline("text-classification", model=MODEL_NAME)

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

        # Sort: Positive first, then negative, both by score
        ranked_results = sorted(results, key=lambda x: (x[1] != "POSITIVE", -x[2]))

        return render_template("index.html", sentences=ranked_results)
    
    return render_template("index.html", sentences=None)

if __name__ == "__main__":
    app.run(debug=True)
