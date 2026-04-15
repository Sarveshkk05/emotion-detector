import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from collections import Counter

from model.model import CNN
from utils.preprocessing import train_transform, val_transform
from config import *

def train_model():

    # Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    num_class = len(class_names)

    # Class imbalance handling 🔥
    labels = [label for _, label in full_dataset.samples]
    count = Counter(labels)
    weights = [1.0 / count[i] for i in range(len(count))]
    weights = torch.tensor(weights).to(device)

    # Split
    val_count = int(len(full_dataset) * VAL_SPLIT)
    train_count = len(full_dataset) - val_count

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    train_set, _ = random_split(train_dataset, [train_count, val_count])
    _, val_set   = random_split(val_dataset, [train_count, val_count])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Model
    model = CNN(num_classes=num_class).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4   # 🔥 overfitting fix
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3
    )

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            preds = output.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                preds = output.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE
            }, MODEL_PATH)

    print("Training Done. Best Val Acc:", best_acc)