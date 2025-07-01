import argparse
import torch
from torchvision.datasets import ImageFolder
from src.data_loader import get_data_loaders
from src.model import get_model, train
from src.predict import load_model, predict_image

def train_model(args):
    print("Starting training...")

    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset = ImageFolder(args.data_dir)
    num_classes = len(dataset.classes)

    model = get_model(num_classes, pretrained=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass save_path so train() saves best model to the right location
    trained_model = train(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
        save_path=args.model_path
    )

    print(f"Training complete. Best model saved to {args.model_path}")

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(args.data_dir)
    class_names = dataset.classes
    num_classes = len(class_names)

    model = load_model(args.model_path, num_classes, device)
    predicted_class, calories = predict_image(args.image_path, model, class_names, device)

    print(f"Predicted Food: {predicted_class}")
    print(f"Estimated Calories: {calories} kcal")

def main():
    parser = argparse.ArgumentParser(description="Bengali Food Calorie Estimator")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, default="data/Bengali_Food", help="Dataset directory")
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--img_size", type=int, default=224)
    train_parser.add_argument("--val_split", type=float, default=0.2)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--model_path", type=str, default="outputs/models/bengali_food_model.pth")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict food and calories from image")
    predict_parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    predict_parser.add_argument("--model_path", type=str, default="outputs/models/bengali_food_model.pth")
    predict_parser.add_argument("--data_dir", type=str, default="data/Bengali_Food", help="Dataset directory for classes")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)

if __name__ == "__main__":
    main()
