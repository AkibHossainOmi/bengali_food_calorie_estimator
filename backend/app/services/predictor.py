from PIL import Image
from io import BytesIO
from src.predict import predict_image
from app.core.model import model, class_names, DEVICE

def predict_image_from_bytes(image_bytes: BytesIO):
    image = Image.open(image_bytes).convert("RGB")
    predicted_class, calories, confidence = predict_image(
        image_path=image_bytes,
        model=model,
        class_names=class_names,
        device=DEVICE
    )
    return predicted_class, calories, confidence
