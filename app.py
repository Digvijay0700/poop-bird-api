from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import timm  # MobileViT

app = Flask(__name__)

# ------------------ CONFIG ------------------
DEVICE = torch.device('cpu')  # change to 'cuda' if running on GPU
POOP_MODEL_PATH = "poop_mobilevit_best.pth"  # path to your new MobileViT .pth checkpoint
BIRD_MODEL_PATH = "bird_count_model.pth"     # existing ResNet50 count model

INPUT_SIZE = 256  # input size used during training
# --------------------------------------------

# --- Poop scan model ---
ckpt = torch.load(POOP_MODEL_PATH, map_location=DEVICE)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

num_classes = len(class_to_idx)
poop_model = timm.create_model("mobilevit_xs", pretrained=False, num_classes=num_classes)
poop_model.load_state_dict(ckpt["model_state_dict"])
poop_model.to(DEVICE)
poop_model.eval()

# --- Bird count model ---
class ResNet50Count(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

resnet_model = ResNet50Count()
resnet_model.load_state_dict(torch.load(BIRD_MODEL_PATH, map_location=DEVICE))
resnet_model.to(DEVICE)
resnet_model.eval()

# --- Transform ---
preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------ ENDPOINTS ------------------

# Poop disease prediction (without accuracy)
@app.route("/predict_pooping", methods=["POST"])
def predict_pooping():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    img = Image.open(request.files['file'].stream).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = poop_model(img_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_class = idx_to_class[predicted_idx]

    return jsonify({"disease": predicted_class})

# Bird counting prediction
@app.route("/predict_birds", methods=["POST"])
def predict_birds():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    img = Image.open(request.files['file'].stream).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = resnet_model(img_tensor)
        predicted_count = output.item()

    return jsonify({"predicted_count": round(predicted_count)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
