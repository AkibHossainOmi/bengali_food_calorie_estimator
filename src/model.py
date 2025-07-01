# src/model.py

import copy
import time
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_model(num_classes, pretrained=True):
    """
    Load a pretrained MobileNetV2 model and replace classifier head.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use pretrained weights.

    Returns:
        model (torch.nn.Module): Modified MobileNetV2 model.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    # Replace the classifier with a new one for num_classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def train(model, dataloaders, criterion, optimizer, device, num_epochs=10, save_path="best_model.pth", patience=3):
    """
    Train the model and save the best weights based on validation accuracy.
    Stop early if no improvement for 'patience' epochs.

    Args:
        model: PyTorch model
        dataloaders: dict with 'train' and 'val' DataLoaders
        criterion: loss function
        optimizer: optimizer
        device: 'cpu' or 'cuda'
        num_epochs: number of epochs
        save_path: path to save best model
        patience: number of epochs to wait for val acc improvement

    Returns:
        model: best trained model
    """
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Check for improvement and save best model on validation phase
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved new best model at epoch {epoch+1} with Val Acc: {best_acc:.4f}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s)")

        print(f"Epoch time: {(time.time() - start_time):.2f} sec\n")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_acc:.4f}")
            break

    model.load_state_dict(best_model_wts)
    return model
