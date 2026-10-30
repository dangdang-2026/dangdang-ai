import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

MEAN = [0.485, 0.456, 0.406]  # ImageNet 평균 — 서버 전처리도 이 숫자
STD = [0.229, 0.224, 0.225]  # ImageNet 표준편차


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False  # 이미 잘 보는 눈(앞부분)은 얼린다
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    top1 = top3 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            k = min(3, model.classifier[1].out_features)
            pred = model(images).topk(k, dim=1).indices
            top1 += (pred[:, 0] == labels).sum().item()
            top3 += (pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return top1 / total, top3 / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)  # train/ val/ 이 있는 폴더
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_tf, eval_tf = build_transforms()
    train_set = datasets.ImageFolder(args.data / "train", transform=train_tf)
    val_set = datasets.ImageFolder(args.data / "val", transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)
    print(f"device={device} 종={len(train_set.classes)} train={len(train_set)} val={len(val_set)}")

    model = build_model(len(train_set.classes)).to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    args.out.mkdir(parents=True, exist_ok=True)
    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        top1, top3 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: loss {running_loss / len(train_set):.3f} "
            f"val top1 {top1:.1%} top3 {top3:.1%} ({time.time() - started:.0f}s)"
        )
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), args.out / "food.pt")

    labels_path = args.out / "labels.json"
    labels_path.write_text(json.dumps(train_set.classes, ensure_ascii=False, indent=2), "utf-8")
    print(f"가장 좋은 val top1 {best_top1:.1%} → {args.out / 'food.pt'}, {labels_path}")


if __name__ == "__main__":
    main()