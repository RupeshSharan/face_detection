# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # Import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all routes, which allows the frontend to communicate with the backend
CORS(app)

# --- Model and Prototxt File Paths ---
# Ensure these files are in the same directory as this app.py script
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# --- Load the Caffe Model from disk ---
try:
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Model loaded successfully.")
except cv2.error as e:
    print(f"[ERROR] Could not load model. Make sure '{prototxt_path}' and '{model_path}' are in the correct directory.")
    net = None

# --- Define the main route that renders the HTML page ---
@app.route('/')
def index():
    """Renders the main HTML page for the user interface."""
    return render_template('index.html')

# --- Define the API endpoint for face detection ---
@app.route('/detect', methods=['POST'])
def detect():
    """Receives an image and a mode, performs face detection, and returns the processed image and face count."""
    if net is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    # Get the JSON data from the request
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400
        
    image_data = data['image']
    # Get the detection mode ('box' or 'blur') from the request
    mode = data.get('mode', 'box') # Default to 'box' mode

    # --- Decode the Base64 Image ---
    try:
        # Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
        header, encoded = image_data.split(",", 1)
        decoded_image = base64.b64decode(encoded)
        # Convert the binary data to a PIL Image
        image = Image.open(io.BytesIO(decoded_image))
        # Convert the PIL image to an OpenCV image (NumPy array)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400

    (h, w) = frame.shape[:2]
    face_count = 0

    # --- Preprocess the Frame and Detect Faces ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # --- Loop Over the Detections ---
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box coordinates are within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            if mode == 'blur':
                # --- Apply Blurring ---
                face = frame[startY:endY, startX:endX]
                # **FIX:** Only apply blur if the face region is not empty
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.GaussianBlur(face, (99, 99), 30)
                    # Put the blurred face back into the frame
                    frame[startY:endY, startX:endX] = face
            else:
                # --- Draw Bounding Box (Default) ---
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # --- Encode the Processed Image back to Base64 ---
    processed_image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the processed image string and face count as a JSON response
    return jsonify({
        'image': 'data:image/jpeg;base64,' + img_str,
        'face_count': face_count
    })

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
