from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import os
import hashlib
import sys
import platform
import time
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

MODEL_PATH = "model_quantized.tflite"

# Memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global interpreter variable with lock for thread safety
interpreter = None
interpreter_lock = threading.Lock()

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_IMAGE_SIZE = (1024, 1024)  # Resize large images before processing
TARGET_SIZE = (224, 224)  # Model input size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Reduced to 5MB

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    with interpreter_lock:
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

            # Use 4 threads for better performance
            interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
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

def preprocess_image(img, target_size=TARGET_SIZE):
    """Optimized image preprocessing"""
    start_time = time.time()
    
    # Resize large images before processing to save memory
    if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
        img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert and resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS).convert('RGB')
    
    # Convert to numpy array efficiently
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    print(f"Image preprocessing took: {time.time() - start_time:.3f}s")
    return img_array

def predict_image(img_array):
    """Thread-safe prediction"""
    if interpreter is None:
        raise RuntimeError("Model not initialized")

    start_time = time.time()
    
    with interpreter_lock:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        result = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Model inference took: {time.time() - start_time:.3f}s")
    return result

def cleanup_old_files():
    """Clean up old uploaded files"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        current_time = time.time()
        
        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                # Remove files older than 1 hour
                if current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

def initialize_app():
    """Initialize application"""
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

@app.route("/", methods=["GET"])
def index():
    """Main page"""
    cleanup_old_files()  # Clean up old files on page load
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    start_time = time.time()
    
    try:
        if interpreter is None:
            return jsonify({"error": "Model not available. Please try again later."}), 503

        file = request.files.get('file')
        if not file or not file.filename:
            return render_template("index.html", 
                                 error_message="Please select a file to upload.")

        if not allowed_file(file.filename):
            return render_template("index.html", 
                                 error_message="Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP).")

        # Secure filename handling
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        print(f"File saved in: {time.time() - start_time:.3f}s")

        # Process image
        img = Image.open(filepath)
        img_array = preprocess_image(img)
        
        # Make prediction
        pred = predict_image(img_array)
        
        # Get confidence score
        confidence = float(pred[0][0])
        
        # Determine prediction
        if confidence < 0.5:
            prediction = "Dog 🐶"
            confidence_display = f"{(1-confidence)*100:.1f}%"
        else:
            prediction = "Not a Dog ❌"
            confidence_display = f"{confidence*100:.1f}%"

        total_time = time.time() - start_time
        print(f"Total prediction time: {total_time:.3f}s")

        return render_template("index.html", 
                             prediction=prediction,
                             confidence=confidence_display,
                             filename=filename,
                             processing_time=f"{total_time:.2f}s")

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Prediction error: {e}")
        
        # Clean up file if it exists
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
            
        return render_template("index.html", error_message=error_message)

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = "healthy" if interpreter is not None else "unhealthy"
    return jsonify({
        "status": status, 
        "model_loaded": interpreter is not None,
        "timestamp": time.time()
    })

# Initialize the app when imported (for gunicorn)
if not initialize_app():
    print("Failed to initialize application during import!")

if __name__ == "__main__":
    if initialize_app():
        port = int(os.environ.get("PORT", 8080))
        print(f"Starting server on port {port}")
        app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
    else:
        print("Failed to initialize application. Exiting.")
        sys.exit(1)