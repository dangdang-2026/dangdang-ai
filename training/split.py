import argparse
import random
from pathlib import Path

from PIL import Image

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}  # ImageFolder 가 읽는 것만


def is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)  # 예: aihub-food/kfood5
    parser.add_argument("--dst", type=Path, required=True)  # 예: aihub-food/split5
    parser.add_argument("--val", type=float, default=0.15)  # val 비율
    parser.add_argument("--seed", type=int, default=42)  # 같은 숫자면 매번 같은 나누기
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for food_dir in sorted(p for p in args.src.glob("*/*") if p.is_dir()):
        photos = sorted(
            p for p in food_dir.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES and is_readable(p)
        )
        rng.shuffle(photos)
        n_val = int(len(photos) * args.val)
        for i, photo in enumerate(photos):
            part = "val" if i < n_val else "train"
            link = args.dst / part / food_dir.name / photo.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(photo.resolve())
        print(f"{food_dir.name}: train {len(photos) - n_val} / val {n_val}")


if __name__ == "__main__":
    main()