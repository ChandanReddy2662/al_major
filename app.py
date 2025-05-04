from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the Keras model (.keras format)
model = load_model('./incident_detection_final_2.keras')  # Replace with your actual model file

# Define your class names
class_names = ['fire', 'road']  # Replace with your actual classes

# Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img_array = preprocess_image(file.read())
        predictions = model.predict(img_array)[0]

        response = {
            'predictions': [
                {'class': class_names[i], 'probability': float(pred)}
                for i, pred in enumerate(predictions)
            ],
            'predicted_class': class_names[int(np.argmax(predictions))]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
