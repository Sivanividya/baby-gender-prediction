from flask import Flask, render_template, request
from app.custom_naive_bayes import GenderPredictor


app = Flask(__name__)
model = GenderPredictor("app/baby_gender_data.xlsx")  # ✅ updated path

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    if request.method == "POST":
        age = request.form["age"]
        placenta = request.form["placenta"]
        month = request.form["month"]
        belly = request.form["belly"]
        history = request.form["history"]
        lifestyle = request.form["lifestyle"]

        prediction, prob = model.predict(age, placenta, month, belly, history, lifestyle)
        probability = f"{prob * 100:.2f}%"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
