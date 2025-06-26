from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import hashlib
import sys
import platform

app = Flask(__name__)

MODEL_PATH = "model_quantized.tflite"

# Memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global interpreter variable
interpreter = None

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def verify_model_file(filepath):
    """Verify model file integrity"""
    if not os.path.exists(filepath):
        return False, "Model file does not exist"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "Model file is empty"

    if file_size < 1000:
        return False, f"Model file too small: {file_size} bytes"

    with open(filepath, 'rb') as f:
        header = f.read(8)
        if len(header) < 8:
            return False, "Invalid file header"

    return True, f"Model file appears valid ({file_size} bytes)"

def initialize_interpreter():
    """Initialize TensorFlow Lite interpreter with error handling"""
    global interpreter

    if interpreter is not None:
        return True

    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")

    try:
        try:
            from tflite_runtime.interpreter import Interpreter
            print("Using tflite_runtime")
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                print("Using tensorflow.lite")
            except ImportError:
                print("Neither tflite_runtime nor tensorflow available")
                return False

        is_valid, message = verify_model_file(MODEL_PATH)
        if not is_valid:
            print(f"Model file verification failed: {message}")
            return False

        print(f"Loading model from: {MODEL_PATH}")
        print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")

        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Model loaded successfully!")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")

        return True

    except Exception as e:
        print(f"Failed to initialize interpreter: {e}")
        print(f"Error type: {type(e).__name__}")
        interpreter = None
        return False

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(img_array):
    if interpreter is None:
        raise RuntimeError("Model not initialized")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def initialize_app():
    print("Initializing application...")

    is_valid, message = verify_model_file(MODEL_PATH)
    if not is_valid:
        print(f"Model file issue: {message}")
        return False

    if not initialize_interpreter():
        print("Failed to initialize model interpreter!")
        return False

    print("Application initialized successfully!")
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    error_message = None

    if request.method == "POST":
        if interpreter is None:
            error_message = "Model not available. Please try again later."
        else:
            file = request.files.get('file')
            if file and file.filename:
                try:
                    # Secure filename handling
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    img = Image.open(filepath)
                    img_array = preprocess_image(img)
                    pred = predict_image(img_array)

                    prediction = "Dog 🐶" if pred[0][0] < 0.5 else "Not a Dog ❌"

                    # Clean up uploaded file
                    try:
                        os.remove(filepath)
                    except:
                        pass

                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"
                    print(f"Prediction error: {e}")
            else:
                error_message = "Please select a file to upload."

    return render_template("index.html", 
                         prediction=prediction, 
                         filename=filename, 
                         error_message=error_message)

@app.route("/health")
def health_check():
    status = "healthy" if interpreter is not None else "unhealthy"
    return {"status": status, "model_loaded": interpreter is not None}

# Initialize the app when imported (for gunicorn)
initialize_app()

if __name__ == "__main__":
    if initialize_app():
        port = int(os.environ.get("PORT", 8080))
        app.run(debug=False, host="0.0.0.0", port=port)
    else:
        print("Failed to initialize application. Exiting.")
        sys.exit(1)