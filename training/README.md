# training — 음식 사진 분류 모델 학습

서버(`app/`)와 따로 둔다. 학습에 쓰는 PyTorch는 `train` 의존성 그룹이라 **서버 Docker 이미지·CI에는 설치되지 않는다.**

## 준비

```bash
uv sync --group train      # torch · torchvision · onnx (처음 한 번)
```

사진은 레포 **밖** `~/04_Dev/dangdang/aihub-food/`에 둔다. 깃·구글 드라이브·Colab에 올리지 않는다 (AI Hub 데이터는 국내에서만, 재배포 금지).

## 실행

```bash
# 1) train / val 나누기 (사진은 복사하지 않고 링크만)
uv run --group train python training/split.py \
  --src ~/04_Dev/dangdang/aihub-food/kfood5 \
  --dst ~/04_Dev/dangdang/aihub-food/split5

# 2) 학습 → models/food.pt · models/labels.json (깃 제외)
uv run --group train python training/train.py \
  --data ~/04_Dev/dangdang/aihub-food/split5 --epochs 3

# 3) 서버용으로 바꾸기 → models/food.onnx (깃 제외), 사진 5장으로 PyTorch와 후보 순서 비교
uv run --group train python training/export_onnx.py \
  --photos ~/04_Dev/dangdang/aihub-food/split5/val

# 4) 내 사진으로 시험 (폴더 이름 = 정답, 150종 밖은 기타_○○)
uv run python -m training.evaluate_my_photos --photos ~/04_Dev/dangdang/my-food/test
```

서버는 `models/food.onnx`·`models/labels.json` 두 파일만 쓴다.

## 실험 기록

| 날짜 | 데이터 | 종 | train / val | epoch | val Top-1 | val Top-3 | 메모 |
|---|---|---|---|---|-------|-------|---|
| 2026-10-30 | AI Hub 5종 (김치볶음밥·비빔밥·김밥·짜장면·떡볶이) | 5 | 4243 / 745 | 3 | 90.3% | 98.8% | 맥 mps, 앞부분 얼림, lr 1e-3 |
| 2026-11-10 | AI Hub 150종 (종당 300장) | 150 | 38250 / 6750 | 5 | 56.7% | 76.3% | 맥 mps, 앞부분 얼림, lr 1e-3, 내 사진+위키미디어 156장 시험 Top-1 49.6% · Top-3 69.6%, 확신도 기준 0.3 |

## 데이터 출처

음식 사진 인식 모델은 과학기술정보통신부·한국지능정보사회진흥원의 AI 허브(aihub.or.kr) 「한국 이미지(음식)」 데이터로 학습했습니다.