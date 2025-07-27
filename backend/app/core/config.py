from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "outputs/models/bengali_food_model.pth"
DATA_DIR = BASE_DIR / "data/Bengali_Food"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
