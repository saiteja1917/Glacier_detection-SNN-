from flask import Flask, render_template, request
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy model for placeholder
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return (x.mean(dim=1, keepdim=True) > 0.5).float()

# Load your trained SNN model here
model = DummyModel().to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def generate_spike_image(image_tensor):
    spikes = (image_tensor > 0.5).float()
    spike_img = spikes.squeeze().numpy() * 255
    return Image.fromarray(spike_img.astype(np.uint8))

def compute_accuracy(pred, ground_truth):
    pred = np.array(pred).flatten() > 127
    gt = np.array(ground_truth).flatten() > 127
    return (pred == gt).sum() / len(gt)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('index.html', accuracy=None)

        # Save original image
        original_path = os.path.join(UPLOAD_FOLDER, 'original.png')
        file.save(original_path)

        original = Image.open(original_path).convert("RGB")
        grayscale = original.convert("L")
        gray_path = os.path.join(UPLOAD_FOLDER, 'grayscale.png')
        grayscale.save(gray_path)

        input_tensor = transform(original).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)

        output_image = output.squeeze().cpu().numpy() * 255
        predicted_img = Image.fromarray(output_image.astype(np.uint8))
        predicted_path = os.path.join(UPLOAD_FOLDER, 'predicted.png')
        predicted_img.save(predicted_path)

        # Spike image
        spike_img = generate_spike_image(transform(grayscale))
        spike_path = os.path.join(UPLOAD_FOLDER, 'spike.png')
        spike_img.save(spike_path)

        # Optional: load ground truth if available
        ground_truth_path = os.path.join(UPLOAD_FOLDER, 'ground_truth.png')  # Replace if your labels are elsewhere
        if os.path.exists(ground_truth_path):
            gt = Image.open(ground_truth_path).resize((128, 128)).convert('L')
            accuracy = compute_accuracy(predicted_img, gt)
        else:
            accuracy = None

        return render_template('index.html',
                               original='uploads/original.png',
                               grayscale='uploads/grayscale.png',
                               spike='uploads/spike.png',
                               predicted='uploads/predicted.png',
                               accuracy=accuracy)

    # Initial GET request
    return render_template('index.html',
                           original=None,
                           grayscale=None,
                           predicted=None,
                           spike=None,
                           accuracy=None)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
