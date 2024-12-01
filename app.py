import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

# Model loading function (from previous artifact)
def load_skin_type_model(model_path, num_classes=1000):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Remove unexpected keys from state_dict
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Initialize Flask app
app = Flask(__name__)

# Load model globally
MODEL_PATH = 'skin_type.pth'
model = load_skin_type_model(MODEL_PATH)

# Image preprocessing
def preprocess_image(image_bytes):
    """
    Preprocess the uploaded image
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transformations and add batch dimension
    return transform(image).unsqueeze(0)

# Home route to serve HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Inference route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for skin type classification
    """
    # Check if image is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Read image bytes
    img_bytes = file.read()
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(img_bytes)
        
        # Disable gradient computation
        with torch.no_grad():
            # Get model predictions
            outputs = model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_prob, top_classes = torch.topk(probabilities, 1)
            
            # Possible skin type labels (adjust as needed)
            skin_type_labels = ['dry', 'normal', 'oily']
            
            # Prepare response
            predictions = [
                {
                    'class': skin_type_labels[cls.item()],
                    'probability': prob.item()
                } 
                for cls, prob in zip(top_classes[0], top_prob[0])
            ]
            
            return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)