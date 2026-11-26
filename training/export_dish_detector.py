"""그릇 찾기 모델을 서버용 dish.onnx · dish.json 으로 내보낸다.

OWL-ViT(google/owlvit-base-patch32, Apache-2.0)는 글로 찾을 것을 알려 주면
사진에서 그 자리를 찾는 모델이다.
찾을 말을 아래 QUERIES 로 고정해서 내보내므로, 서버는 사진만 넣으면 된다.
앞쪽 POSITIVE 개는 「음식이 담긴 그릇」, 나머지는 빈 그릇이나 컵처럼 음식이 아닌 것이다.
뒤쪽 점수가 더 높은 자리는 버린다.
학습은 하지 않는다. 받은 모델을 그대로 쓰고, int8 로 줄여 CPU 에서 빠르게 돈다 (600MB → 150MB).

uv run --group train python -m training.export_dish_detector
"""

import argparse
import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import OwlViTForObjectDetection, OwlViTProcessor

MODEL = "google/owlvit-base-patch32"
QUERIES = [
    "a bowl of food",
    "a plate of food",
    "a bowl of soup",
    "a bowl of rice",
    "a dish of food",
    "a pan of food",
    "an empty bowl",
    "an empty plate",
    "a cup",
    "a glass",
    "a bottle",
    "a can",
    "a person",
    "a phone",
]
POSITIVE = 6
SIZE = 768  # OWL-ViT 입력 크기


class FixedQueries(torch.nn.Module):
    """찾을 말을 넣어 둔 모델: 사진 → (자리 cx·cy·w·h 0~1, 말마다 점수 0~1)"""

    def __init__(self, model: OwlViTForObjectDetection, text: dict) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_ids", text["input_ids"])
        self.register_buffer("attention_mask", text["attention_mask"])

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            input_ids=self.input_ids, attention_mask=self.attention_mask, pixel_values=image
        )
        return out.pred_boxes, out.logits.sigmoid()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    processor = OwlViTProcessor.from_pretrained(MODEL)
    model = OwlViTForObjectDetection.from_pretrained(MODEL).eval()
    wrapped = FixedQueries(model, processor(text=[QUERIES], return_tensors="pt")).eval()

    full = args.out / "dish-fp32.onnx"
    torch.onnx.export(
        wrapped,
        (torch.zeros(1, 3, SIZE, SIZE),),
        full,
        input_names=["image"],
        output_names=["boxes", "scores"],
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(full, args.out / "dish.onnx", weight_type=QuantType.QInt8)
    full.unlink()

    config = {
        "model": MODEL,
        "queries": QUERIES,
        "positive": POSITIVE,
        "size": SIZE,
        "mean": processor.image_processor.image_mean,
        "std": processor.image_processor.image_std,
    }
    (args.out / "dish.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), "utf-8")
    print(f"→ {args.out / 'dish.onnx'}, {args.out / 'dish.json'}")


if __name__ == "__main__":
    main()
