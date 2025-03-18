from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError

# Register the custom loss function
custom_objects = {"mse": MeanSquaredError()}

# Load the trained model with the custom loss function
model = tf.keras.models.load_model("stock_model.h5", custom_objects=custom_objects)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['features'])  # Convert input to NumPy array

        # Ensure input has the correct shape (1, 10, 4)
        input_data = input_data.reshape(1, 10, 4)

        # Make prediction
        prediction = model.predict(input_data).tolist()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
