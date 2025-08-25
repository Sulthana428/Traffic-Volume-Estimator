from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
with open("IBM/flask/Traffic volume.pkl", "rb") as f:
    model = pickle.load(f)

with open("IBM/flask/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form inputs
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hours'])

        # Create input DataFrame
        input_df = pd.DataFrame([[holiday, temp, rain, snow, weather, year, month, day, hour]],
            columns=['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hour']
        )

        # Preprocess and predict
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]

        return render_template("chance.html", result=round(prediction, 2))

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template("noChance.html")

if __name__ == "__main__":
    app.run(debug=True)