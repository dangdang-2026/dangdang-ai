import argparse
import unicodedata
from collections import Counter
from pathlib import Path

from app.core.config import settings
from app.food.predictor import Candidate, FoodPredictor, NotAnImageError

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
OTHER_PREFIX = "기타_"  # 150종 밖 음식 폴더
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Result = tuple[str, list[Candidate]]  # (정답 폴더 이름, 후보 3개)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)  # 맥에서 풀어 쓴 한글을 모아 쓴 글자로


def predict_all(predictor: FoodPredictor, root: Path) -> list[Result]:
    results = []
    for food_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        truth = nfc(food_dir.name)
        for photo in sorted(food_dir.iterdir()):
            if photo.suffix.lower() == ".heic":
                print(f"건너뜀(HEIC는 JPG로 바꾸기): {photo.name}")
            if photo.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            try:
                results.append((truth, predictor.predict(photo.read_bytes())))
            except NotAnImageError:
                print(f"건너뜀(열 수 없는 파일): {photo.name}")
    return results


def percent(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "-"


def report_accuracy(known: list[Result]) -> None:
    top1 = sum(truth == candidates[0].name for truth, candidates in known)
    top3 = sum(truth in [c.name for c in candidates] for truth, candidates in known)
    print(f"Top-1 {percent(top1, len(known))} / Top-3 {percent(top3, len(known))} (목표 85%)")


def report_confusions(known: list[Result]) -> None:
    pairs = Counter(
        (truth, candidates[0].name) for truth, candidates in known if truth != candidates[0].name
    )
    print("많이 틀리는 짝 (정답 → 모델 1등):")
    for (truth, predicted), count in pairs.most_common(10):
        print(f"  {truth} → {predicted}: {count}장")


def report_thresholds(known: list[Result], others: list[Result]) -> None:
    print("기준값 | 150종 통과 | 통과 중 Top-3 정답 | 기타 통과")
    for threshold in THRESHOLDS:
        passed = [(t, c) for t, c in known if c[0].confidence >= threshold]
        correct = sum(truth in [c.name for c in candidates] for truth, candidates in passed)
        other_passed = sum(c[0].confidence >= threshold for _, c in others)
        print(
            f"{threshold:.1f} | {percent(len(passed), len(known))} | "
            f"{percent(correct, len(passed))} | {percent(other_passed, len(others))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)  # 음식별 폴더가 든 test 폴더
    args = parser.parse_args()

    predictor = FoodPredictor(Path(settings.food_model_path), Path(settings.food_labels_path))
    labels = {nfc(label) for label in predictor.labels}
    results = predict_all(predictor, args.photos)

    known = [r for r in results if r[0] in labels]
    others = [r for r in results if r[0].startswith(OTHER_PREFIX)]
    typos = {r[0] for r in results if r[0] not in labels and not r[0].startswith(OTHER_PREFIX)}
    if typos:
        print(f"⚠️ 목록에 없는 폴더 이름(점수에서 뺌): {sorted(typos)}")

    print(f"150종 안 {len(known)}장 / 기타 {len(others)}장")
    report_accuracy(known)
    report_confusions(known)
    report_thresholds(known, others)


if __name__ == "__main__":
    main()
