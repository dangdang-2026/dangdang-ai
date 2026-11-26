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

# 2-1) 이어서 학습 (fine-tuning) — 지난 food.pt 에서 시작, 앞부분 끝 3묶음 풀기
uv run --group train python training/train.py \
  --data ~/04_Dev/dangdang/aihub-food/split150 --epochs 5 --init 백업폴더/food.pt \
  --unfreeze 3 --lr 3e-4 --backbone-lr 1e-4 --cosine

# 2-2) Hugging Face(timm) 모델로 학습 → food.pt · labels.json · food.onnx 까지 (3)은 건너뜀)
uv run --group train python -m training.train_timm \
  --data ~/04_Dev/dangdang/aihub-food/split150 --epochs 3

# 3) 서버용으로 바꾸기 → models/food.onnx (깃 제외), 사진 5장으로 PyTorch와 후보 순서 비교
uv run --group train python training/export_onnx.py \
  --photos ~/04_Dev/dangdang/aihub-food/split5/val

# 4) 내 사진으로 시험 (폴더 이름 = 정답, 150종 밖은 기타_○○)
uv run python -m training.evaluate_my_photos --photos ~/04_Dev/dangdang/my-food/test
```

서버는 `models/food.onnx`·`models/labels.json` 두 파일만 쓴다.

## 한상 사진: 그릇 찾기 모델

한상 사진에서 그릇 자리를 먼저 찾고, 자리마다 잘라서 위의 150종 모델로 이름 후보를 낸다. 그릇 찾기는 학습하지 않고 OWL-ViT(`google/owlvit-base-patch32`, Apache-2.0)를 「음식이 담긴 그릇」 같은 말로 고정해서 쓴다. 빈 그릇, 컵, 병, 사람 점수가 더 높은 자리는 버린다.

```bash
# models/dish.onnx (int8, 약 150MB) · models/dish.json (깃 제외)
uv run --group train python -m training.export_dish_detector
```

서버는 `models/dish.onnx`·`models/dish.json`이 있으면 그릇을 찾고, 없으면 예전처럼 사진 전체를 음식 하나로 본다. 서버에 올릴 때는 `food.onnx`와 같은 폴더에 두 파일을 함께 둔다.

| 날짜 | 시험 | 결과 | 메모 |
|---|---|---|---|
| 2026-11-26 | 내 한상 사진 23장 | 그릇 자리는 대부분 맞게 찾음, 사진 1장 약 1초 (맥 CPU, 검출과 분류 합계) | 빈 그릇과 컵이 가끔 남아 「식혜」로 나옴. 잘라 낸 그릇의 이름은 사진 전체 분류보다 덜 맞음 (150종 모델이 그릇 하나만 찍은 사진으로 배워서) |

## 실험 기록

| 날짜         | 데이터 | 종 | train / val | epoch | val Top-1 | val Top-3 | 메모 |
|------------|---|---|---|---|-------|-------|---|
| 2026-10-30 | AI Hub 5종 (김치볶음밥·비빔밥·김밥·짜장면·떡볶이) | 5 | 4243 / 745 | 3 | 90.3% | 98.8% | 맥 mps, 앞부분 얼림, lr 1e-3 |
| 2026-11-10 | AI Hub 150종 (종당 300장) | 150 | 38250 / 6750 | 5 | 56.7% | 76.3% | 맥 mps, 앞부분 얼림, lr 1e-3, 내 사진+위키미디어 156장 시험 Top-1 49.6% · Top-3 69.6%, 확신도 기준 0.3 |
| 2026-11-11 | AI Hub 150종 (종당 300장) | 150 | 38250 / 6750 | 5 | 73.9% | 89.4% | 맥 mps, 11/10 모델에서 이어서, 앞부분 끝 3묶음 풀기(앞 1e-4 · 뒤 3e-4), cosine, epoch당 약 800초, 같은 156장 시험 Top-1 68.0% · Top-3 84.0%, 확신도 기준 0.5 (채택 — 서버 모델 교체) |
| 2026-11-14 | AI Hub 150종 (종당 300장) | 150 | 38250 / 6750 | 3 | 82.2% | 93.8% | 맥 mps, `timm/convnext_nano.in12k_ft_in1k`(Hugging Face, ImageNet-12k, Apache-2.0), 앞부분 전체 1e-4 · 판단 층 1e-3, cosine, epoch당 약 4,200초, 같은 156장 시험 Top-1 75.2% · Top-3 91.2%, 확신도 기준 0.7, 모델 60MB · CPU 1장 118ms (채택 — 서버 모델 교체) |

## 데이터 출처

음식 사진 인식 모델은 과학기술정보통신부·한국지능정보사회진흥원의 AI 허브(aihub.or.kr) 「한국 이미지(음식)」 데이터로 학습했습니다.
