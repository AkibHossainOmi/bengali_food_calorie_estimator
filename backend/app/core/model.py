from torchvision.datasets import ImageFolder
from src.predict import load_model
from app.core.config import MODEL_PATH, DATA_DIR, DEVICE

# Load dataset class names once
dataset = ImageFolder(str(DATA_DIR))
class_names = dataset.classes
num_classes = len(class_names)

# Load the model once
model = load_model(str(MODEL_PATH), num_classes, DEVICE)
