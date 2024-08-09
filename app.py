from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import requests

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if text input was provided
        user_input = request.form.get("text")
        if user_input:
            try:
                # Make a POST request to the /predict route
                response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
                if response.status_code == 200:
                    prediction = response.json().get("prediction", "No prediction found.")
                    return render_template("mylanding.html", prediction=prediction)
                else:
                    return render_template("mylanding.html", error="Failed to get prediction.")
            except Exception as e:
                return render_template("mylanding.html", error=str(e))

        # Check if a file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file:
                try:
                    files = {'file': file}
                    response = requests.post("http://127.0.0.1:8000/predict", files=files)
                    if response.status_code == 200:
                        return send_file(BytesIO(response.content), mimetype="text/csv", as_attachment=True, download_name="Predictions.csv")
                    else:
                        return render_template("mylanding.html", error="Failed to process the file.")
                except Exception as e:
                    return render_template("mylanding.html", error=str(e))

    return render_template("mylanding.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Load the model, scaler, and vectorizer
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions_csv, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions_csv,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            return response

        elif request.is_json and "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=8000, debug=True)
