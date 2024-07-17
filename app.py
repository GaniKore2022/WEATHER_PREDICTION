from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Set the path for the static directory
static_path = os.path.join(os.path.dirname(__file__), 'static')

# Load the saved RandomForestClassifier model
model = joblib.load('weather_forecast.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    date = request.form['date']
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    
    prediction = get_prediction(location, date, temperature, humidity, wind_speed)
    return render_template('prediction.html', location=location, date=date, weather=prediction)

def get_weather_features(location, date, temperature, humidity, wind_speed):
    # Add two dummy features to match the 5 features expected by the model
    feature_1 = temperature
    feature_2 = humidity
    feature_3 = wind_speed
    feature_4 = 0  # Dummy feature
    feature_5 = 0  # Dummy feature
    
    features = [feature_1, feature_2, feature_3, feature_4, feature_5]
    return features

def get_prediction(location, date, temperature, humidity, wind_speed):
    features = get_weather_features(location, date, temperature, humidity, wind_speed)
    prediction = model.predict([features])[0]
    return prediction

if __name__ == '__main__':
    # Add the static path to the Flask app
    app.static_folder = static_path
    app.run(debug=True)
