import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Models
from torchvision.models import inception_v3, densenet201, DenseNet201_Weights, Inception_V3_Weights
import timm

# Preprocessing
import cv2
from PIL import Image
import numpy as np

# Grad-CAM
from captum.attr import LayerGradCam
from captum.attr._utils.visualization import visualize_image_attr

# Flask
from flask import Flask, request, render_template, jsonify, url_for
import io
import base64

# ==============================
# ✅ 1. App Setup
# ==============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ==============================
# ✅ 2. Device and Classes
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hardcoding classes based on BRISC challenge
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for i, cls in enumerate(classes)}
print(f"Using device: {device}")
print(f"Classes: {classes}")

# ==============================
# ✅ 3. Custom Preprocessing (MODIFIED)
# ==============================
class CustomPreprocessing:
    """
    Modified to return both the tensor and the preprocessed
    CV2 image for Grad-CAM overlay.
    """
    def __init__(self, image_size=(299, 299)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __call__(self, img_pil):
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)

        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            img_cv = img_cv[y:y+h, x:x+w]

        img_resized_cv = cv2.resize(img_cv, self.image_size)
        
        # Convert resized CV2 image back to PIL for ToTensor
        img_pil_resized = Image.fromarray(cv2.cvtColor(img_resized_cv, cv2.COLOR_BGR2RGB))
        
        tensor = self.transform(img_pil_resized)
        
        # Return tensor for model and cv2 image for overlay
        return tensor, img_resized_cv

# ==============================
# ✅ 4. Model Definitions (Copied from your script)
# ==============================

# 4.1. Base Models
def get_base_models(num_classes):
    # InceptionV3
    model_incep = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    in_features_incep = model_incep.fc.in_features
    model_incep.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features_incep, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    # DenseNet201
    model_dense = densenet201(weights=DenseNet201_Weights.DEFAULT)
    in_features_dense = model_dense.classifier.in_features
    model_dense.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features_dense, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    # Xception
    model_xcep = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    in_features_xcep = model_xcep.num_features
    model_xcep.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features_xcep, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model_incep, model_dense, model_xcep

# 4.2. Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3, num_classes):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        # We need to run a dummy forward pass to get in_features
        # Run this on the *device*
        dummy_input = torch.randn(1, 3, 299, 299).to(device)
        self.model1.eval(), self.model2.eval(), self.model3.eval()
        with torch.no_grad():
            out1 = self.model1(dummy_input)
            if hasattr(out1, "logits"):
                out1 = out1.logits
            out2 = self.model2(dummy_input)
            out3 = self.model3(dummy_input)
        in_features = out1.shape[1] + out2.shape[1] + out3.shape[1]
        print(f"Meta-learner input features: {in_features}")

        self.meta = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out1 = self.model1(x)
        if hasattr(out1, "logits"):
            out1 = out1.logits
        out2 = self.model2(x)
        out3 = self.model3(x)
        combined = torch.cat((out1, out2, out3), dim=1)
        return self.meta(combined)

# ==============================
# ✅ 5. Load Model and Transform
# ==============================
def load_model():
    # Instantiate base models and move to device *before* passing to Ensemble
    model_incep, model_dense, model_xcep = get_base_models(len(classes))
    model_incep, model_dense, model_xcep = model_incep.to(device), model_dense.to(device), model_xcep.to(device)
    
    # Instantiate ensemble
    model = EnsembleModel(model_incep, model_dense, model_xcep, num_classes=len(classes)).to(device)
    
    # Load state dict
    model_path = "brisc_ensemble_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully")
    return model

model = load_model()
transform = CustomPreprocessing(image_size=(299, 299))

# ==============================
# ✅ 6. Grad-CAM Helper Function
# ==============================
def generate_gradcam_overlay(model, input_tensor, preprocessed_cv, target_class_idx):
    """
    Generates a Grad-CAM overlay and returns it as a base64 encoded string.
    """
    # Target the features of the DenseNet201 model (model2)
    layer_gc = LayerGradCam(model, model.model2.features)
    
    # Compute attribution
    attribution = layer_gc.attribute(input_tensor, target=target_class_idx)
    
    # Process heatmap
    heatmap = attribution.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=2)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) # Normalize
    
    # Apply heatmap to the *preprocessed* CV2 image
    heatmap = cv2.resize(heatmap, (preprocessed_cv.shape[1], preprocessed_cv.shape[0]))
    heatmap_uint = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    
    # Create superimposed image
    superimposed_img = cv2.addWeighted(preprocessed_cv, 0.6, heatmap_color, 0.4, 0)
    
    # Convert to base64 string to send to frontend
    _, buffer = cv2.imencode('.png', superimposed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

# ==============================
# ✅ 7. Flask Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    try:
        # 1. Read and save uploaded image
        img_pil = Image.open(file.stream).convert('RGB')
        
        # Also convert to base64 to send back to UI
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        original_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        original_img_src = f"data:image/png;base64,{original_img_base64}"

        # 2. Preprocess the image
        # Our modified transform returns both
        tensor, preprocessed_cv = transform(img_pil)
        input_tensor = tensor.unsqueeze(0).to(device)

        # 3. Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        pred_class = idx_to_class[pred.item()]
        confidence = conf.item()
        
        # 4. Get all probabilities for the chart
        all_probs = probs.cpu().numpy().flatten()
        probabilities = [{"class": idx_to_class[i], "prob": float(p)} for i, p in enumerate(all_probs)]
        probabilities.sort(key=lambda x: x['prob'], reverse=True)


        # 5. Generate Grad-CAM
        gradcam_img_src = generate_gradcam_overlay(
            model=model,
            input_tensor=input_tensor,
            preprocessed_cv=preprocessed_cv,
            target_class_idx=pred.item()
        )

        # 6. Send results
        return jsonify({
            'prediction': pred_class,
            'confidence': f"{confidence*100:.2f}",
            'original_image': original_img_src,
            'gradcam_image': gradcam_img_src,
            'probabilities': probabilities
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)