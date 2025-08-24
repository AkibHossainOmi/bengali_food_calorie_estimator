# main.py

import argparse
import torch
from torchvision.datasets import ImageFolder
from src.data_loader import get_data_loaders
from src.model import get_model, train
import torch.nn as nn
from src.predict import load_model, predict_image

def train_model(args):
    print("Starting training...")

    # Load data
    train_loader, val_loader, class_weights = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset = ImageFolder(args.data_dir)
    num_classes = len(dataset.classes)

    # Load model with frozen backbone
    model = get_model(num_classes, pretrained=True, freeze_backbone=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Weighted loss for imbalanced dataset
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Separate learning rates for backbone and classifier
    backbone_params = [p for n, p in model.named_parameters() if "features" in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': classifier_params, 'lr': args.classifier_lr}
    ])

    # Train the model
    trained_model = train(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
        save_path=args.model_path,
        patience=args.patience,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        progressive_unfreeze_epoch=args.unfreeze_epoch  # unfreeze backbone if specified
    )

    print(f"Training complete. Best model saved to {args.model_path}")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(args.data_dir)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Load trained model
    model = load_model(args.model_path, num_classes, device)

    # Predict
    predicted_class, calories, confidence = predict_image(
        args.image_path, model, class_names, device, threshold=args.threshold
    )

    print(f"Predicted Food: {predicted_class}")
    print(f"Estimated Calories: {calories} kcal")
    print(f"Confidence: {confidence * 100:.2f}%")


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
    train_parser.add_argument("--lr", type=float, default=1e-4)  # fallback
    train_parser.add_argument("--backbone_lr", type=float, default=1e-5)
    train_parser.add_argument("--classifier_lr", type=float, default=1e-4)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--unfreeze_epoch", type=int, default=None,
                              help="Epoch to start unfreezing backbone progressively")
    train_parser.add_argument("--model_path", type=str, default="outputs/models/bengali_food_model.pth")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    train_parser.add_argument("--checkpoint_path", type=str, default="outputs/models/checkpoint.pth")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict food and calories from image")
    predict_parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    predict_parser.add_argument("--model_path", type=str, default="outputs/models/bengali_food_model.pth")
    predict_parser.add_argument("--data_dir", type=str, default="data/Bengali_Food", help="Dataset directory for classes")
    predict_parser.add_argument("--threshold", type=float, default=0.6,
                                help="Confidence threshold below which prediction is 'Unknown Food'")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
