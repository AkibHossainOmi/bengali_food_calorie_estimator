import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def get_data_loaders(data_dir, batch_size=16, img_size=224, val_split=0.2, num_workers=4):
    """
    Creates PyTorch DataLoaders with weighted sampling for tiny/imbalanced datasets.
    """

    # Stronger augmentations for tiny dataset
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Split train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transforms

    # Compute class weights for weighted sampling
    targets = [s[1] for s in train_dataset.dataset.samples]
    class_sample_count = np.array([targets.count(i) for i in range(len(full_dataset.classes))])
    class_weights = 1. / class_sample_count
    samples_weights = np.array([class_weights[targets[i]] for i in train_dataset.indices])
    samples_weights = torch.from_numpy(samples_weights).float()
    train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_weights
