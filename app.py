import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings and info logs
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure directories for uploads and outputs exist
os.makedirs('./uploads', exist_ok=True)
os.makedirs('./outputs', exist_ok=True)

# Load the TF-Hub style transfer model
hub_handle = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
style_transfer_model = hub.KerasLayer(hub_handle)
# Preprocessing function to load and preprocess images
def preprocess_image(image_path, target_size):
    """Loads and preprocesses images for the style transfer model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)  # Resize image to the target size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Deprocessing function to convert tensor back to image
def deprocess_image(tensor):
    """Deprocess tensor to a PIL image."""
    tensor = tf.squeeze(tensor, axis=0)  # Remove batch dimension
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)  # Ensure values are in [0, 1]
    tensor = (tensor * 255).numpy().astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(tensor)

# Style Transfer Logic using TF-Hub
def style_transfer(content_path, style_path):
    """Applies style transfer using the pre-trained TF-Hub model."""
    # Load and preprocess the content and style images
    content_image = preprocess_image(content_path, (384, 384))  # Content image size can vary
    style_image = preprocess_image(style_path, (256, 256))  # Recommended size for style image

    # Use average pooling to smooth style image
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

    # Perform style transfer
    outputs = style_transfer_model(content_image, style_image)
    stylized_image = outputs[0]  # Extract the stylized image from the outputs
    return deprocess_image(stylized_image)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Retrieve content and style images
        content_image = request.files['contentImage']
        style_image = request.files['styleImage']

        # Secure filenames before saving
        content_filename = secure_filename(content_image.filename)
        style_filename = secure_filename(style_image.filename)

        # Save the uploaded images to the server
        content_image_path = os.path.join('./uploads', content_filename)
        style_image_path = os.path.join('./uploads', style_filename)
        content_image.save(content_image_path)
        style_image.save(style_image_path)

        # Perform style transfer
        stylized_image = style_transfer(content_image_path, style_image_path)

        # Save the styled image
        output_filename = f'styled_{content_filename}'
        output_path = os.path.join('./outputs', output_filename)
        stylized_image.save(output_path)  # Save as PIL Image

        # Return the path of the generated image
        return jsonify({'image_path': output_filename})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500  # Return error message in response

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory('./outputs', filename)

if __name__ == '__main__':
    app.run(debug=True)
