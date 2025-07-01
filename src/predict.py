# src/predict.py

import torch
from torchvision import transforms
from PIL import Image
from src.calorie_map import calorie_map

def load_model(model_path, num_classes, device):
    from src.model import get_model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, class_names, device, img_size=224):
    """
    Predict the class of a single image and estimate calories.

    Args:
        image_path (str): Path to image.
        model: Trained PyTorch model.
        class_names (list): List of class names indexed by model output.
        device: 'cpu' or 'cuda'.
        img_size (int): Resize dimension.

    Returns:
        predicted_class (str), estimated_calories (int or str)
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        estimated_calories = calorie_map.get(predicted_class, "Unknown")

    return predicted_class, estimated_calories
