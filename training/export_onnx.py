"""학습한 food.pt 를 서버용 food.onnx 로 바꾸고, 같은 사진에서 두 결과가 같은지 확인."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from train import build_model, build_transforms

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def export(model: torch.nn.Module, onnx_path: Path) -> None:
    dummy = torch.randn(1, 3, 224, 224)  # 사진 1장 모양: (장수, 색 3, 세로, 가로)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},  # 장수는 바뀔 수 있음.
        external_data=False,  # 파일 1개로 저장
    )


def pick_photos(folder: Path) -> list[Path]:
    photos = []
    for food_dir in sorted(folder.iterdir()):  # 음식 폴더마다
        if not food_dir.is_dir():
            continue
        for photo in sorted(food_dir.iterdir()):  # 첫 사진 1장만
            if photo.suffix.lower() in PHOTO_SUFFIXES:
                photos.append(photo)
                break
    return photos[:5]


def check(model: torch.nn.Module, onnx_path: Path, photos: list[Path], labels: list[str]) -> bool:
    _, eval_tf = build_transforms()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_same = True
    for photo in photos:
        tensor = eval_tf(Image.open(photo).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            torch_out = model(tensor).numpy()
        onnx_out = session.run(None, {"image": tensor.numpy()})[0]
        torch_top3 = np.argsort(-torch_out[0])[:3].tolist()
        onnx_top3 = np.argsort(-onnx_out[0])[:3].tolist()
        same = torch_top3 == onnx_top3
        all_same = all_same and same
        names = [labels[i] for i in onnx_top3]
        diff = float(np.abs(torch_out - onnx_out).max())
        print(f"{'같음' if same else '다름'} {photo.name}: {names} (차이 {diff:.6f})")
    return all_same


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))  # food.pt 가 있는 폴더
    parser.add_argument("--photos", type=Path, required=True)  # 음식별 폴더가 든 사진 폴더
    args = parser.parse_args()

    labels = json.loads((args.models / "labels.json").read_text("utf-8"))
    model = build_model(len(labels))
    model.load_state_dict(torch.load(args.models / "food.pt", map_location="cpu"))
    model.eval()

    onnx_path = args.models / "food.onnx"
    export(model, onnx_path)
    print(f"저장: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f}MB)")

    photos = pick_photos(args.photos)
    if check(model, onnx_path, photos, labels):
        print(f"{len(photos)}장 모두 후보 순서가 같다 → 서버에 써도 된다")
    else:
        print("후보 순서가 다른 사진이 있다 → 변환을 다시 확인")


if __name__ == "__main__":
    main()
