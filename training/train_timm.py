import argparse
import json
import time
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from training.train import build_transforms, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="timm/convnext_nano.in12k_ft_in1k")  # Hugging Face 이름
    parser.add_argument("--data", type=Path, required=True)  # train/ val/ 이 있는 폴더
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)  # 새 판단 층
    parser.add_argument("--backbone-lr", type=float, default=1e-4)  # 앞부분 전체
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_tf, eval_tf = build_transforms()
    train_set = datasets.ImageFolder(args.data / "train", transform=train_tf)
    val_set = datasets.ImageFolder(args.data / "val", transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)
    print(f"device={device} 종={len(train_set.classes)} train={len(train_set)} val={len(val_set)}")

    num_classes = len(train_set.classes)
    model = timm.create_model(f"hf-hub:{args.model}", pretrained=True, num_classes=num_classes)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    model = model.to(device)
    print(f"{args.model} 숫자 {sum(p.numel() for p in model.parameters()):,}개")

    optimizer = torch.optim.AdamW(
        [{"params": head, "lr": args.lr}, {"params": body, "lr": args.backbone_lr}]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        scheduler.step()
        top1, top3 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: loss {running_loss / len(train_set):.3f} "
            f"val top1 {top1:.1%} top3 {top3:.1%} ({time.time() - started:.0f}s)",
            flush=True,
        )
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), args.out / "food.pt")

    labels_path = args.out / "labels.json"
    labels_path.write_text(json.dumps(train_set.classes, ensure_ascii=False, indent=2), "utf-8")
    print(f"가장 좋은 val top1 {best_top1:.1%}")

    model.load_state_dict(torch.load(args.out / "food.pt", map_location="cpu"))
    model.eval().cpu()
    onnx_path = args.out / "food.onnx"
    torch.onnx.export(
        model,
        (torch.randn(1, 3, 224, 224),),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}},
        dynamo=False,
    )
    print(f"서버용 {onnx_path} {onnx_path.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
