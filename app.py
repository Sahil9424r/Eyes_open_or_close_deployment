from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

# Load the trained model
model_path = 'eyes (1).keras'  # Update this with the correct model file path
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and preprocess the image using Pillow
        image = Image.open(BytesIO(file.read()))  # Open the image
        image = image.convert('RGB')  # Convert image to RGB to ensure 3 channels
        image = image.resize((256, 256))  # Resize to match the model's expected input size
        image_array = img_to_array(image)  # Convert image to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize the image data
    except Exception as e:
        return jsonify({'error': str(e)})

    # Make prediction
    prediction = model.predict(image_array)
    predicted_value = prediction[0][0]  # Access the first scalar value in the prediction

    # Apply the binary classification threshold
    output = 1 if predicted_value >= 0.5 else 0  # Binary classification threshold at 0.5

    labels = ['close eyes', 'open eyes']  # Use your model's labels
    prediction_text = 'Predicted Class: {}'.format(labels[output])

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
