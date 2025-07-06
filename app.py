from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import io
import numpy as np
from torchvision import transforms
from mnist import MNISTClassifierSimple

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = MNISTClassifierSimple()
model.load_state_dict(torch.load("mnist_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

@app.route("/")
def index():
    return render_template("index.html")

# Update preprocessing in the classify endpoint to include normalization
@app.route("/classify", methods=["POST"])
def classify():
    try:
        # Get the image data from the request
        image_data = request.files["image"].read()
        image = Image.open(io.BytesIO(image_data)).convert("L")

        # Preprocess the image
        image = image.resize((28, 28))
        image = np.array(image, dtype=np.float32) / 255.0
        image = 1.0 - image  # Invert colors (MNIST expects white digit on black background)
        image = (image - 0.1307) / 0.3081  # Normalize using MNIST mean and std
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Predict the digit
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return jsonify({"digit": predicted.item()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
