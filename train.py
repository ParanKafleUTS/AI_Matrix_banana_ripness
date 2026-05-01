"""
train.py – Train MobileNetV3, EfficientNetB0, and ResNet50 on the banana
ripeness dataset and save the best weights for each model.

Usage:
    python train.py [--data-dir DATA_DIR] [--epochs N] [--batch-size N]

The script expects the dataset to be organized as:
    DATA_DIR/
      train/  <class_name>/  *.jpg
      valid/  <class_name>/  *.jpg
      test/   <class_name>/  *.jpg
"""

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "Dataset/banana-ripeness-dataset-original"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 10
NUM_CLASSES = 6
INPUT_SIZE = 224
OUTPUT_DIR = "saved_models"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((-90, -90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((0, 0)),
        ]),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    since = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}\n" + "-" * 10)

        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"  {phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.6f}")
    model.load_state_dict(best_weights)
    return model


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str) -> nn.Module:
    if name == "MobileNetV3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train banana ripeness classifiers")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf),
        "valid": datasets.ImageFolder(os.path.join(args.data_dir, "valid"), val_tf),
        "test":  datasets.ImageFolder(os.path.join(args.data_dir, "test"),  val_tf),
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
        )
        for split in ["train", "valid", "test"]
    }

    dataset_sizes = {s: len(image_datasets[s]) for s in ["train", "valid", "test"]}
    class_names = image_datasets["train"].classes
    print(f"Classes ({len(class_names)}): {class_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trained = {}
    for model_name in ["MobileNetV3", "EfficientNetB0", "ResNet50"]:
        print(f"\n{'=' * 40}\nTraining {model_name}\n{'=' * 40}")
        model = build_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trained[model_name] = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, device, args.epochs
        )
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_banana_ripeness.pth")
        torch.save(trained[model_name].state_dict(), save_path)
        print(f"Saved {model_name} → {save_path}")

    # Evaluate on test set
    print("\nFinal Evaluation on Test Set:")
    for model_name, model in trained.items():
        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                corrects += torch.sum(preds == labels.data)
        acc = corrects.double() / dataset_sizes["test"]
        print(f"  {model_name} Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
