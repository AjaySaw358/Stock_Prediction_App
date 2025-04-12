from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError

custom_objects = {"mse": MeanSquaredError()}
model = tf.keras.models.load_model("stock_model.h5", custom_objects=custom_objects)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        close_values = []

        for i in range(10):
            open_val = float(request.form[f'open_{i}'])
            high_val = float(request.form[f'high_{i}'])
            low_val = float(request.form[f'low_{i}'])
            close_val = float(request.form[f'close_{i}'])

            input_data.append([open_val, high_val, low_val, close_val])
            close_values.append(close_val)

        input_array = np.array(input_data).reshape(1, 10, 4)
        prediction = model.predict(input_array).flatten().tolist()
        close_values.append(prediction[0])  # Add predicted value to chart

        return render_template("result.html", close_trend=close_values, prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
