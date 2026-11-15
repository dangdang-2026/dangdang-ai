"""2단계: 규칙에 걸린 '약·용량' 문장만 GPT에 물어 진짜 약 용량 이야기인지 확인한다.

- 글 전체·작성자 정보는 보내지 않는다. 걸린 문장 몇 개만 보낸다
- 실패(키 없음·시간 초과·사용량 초과·이상한 답)하면 None -> 부르는 쪽은 규칙 결과를 그대로 쓴다
- 숫자·판단을 GPT가 만들지 않는다: 문장마다 true/false만 받는다
"""

import json
import logging

from openai import OpenAI, OpenAIError

log = logging.getLogger(__name__)

PROMPT_VERSION = (
    "dose-check-v1"  # 프롬프트를 고치면 숫자를 올린다 (로그로 어떤 버전이 답했는지 안다)
)

SYSTEM = """너는 임신성 당뇨 산모 커뮤니티의 글 검사 도우미다.
사용자가 준 문장마다, 약(인슐린·먹는 약 등)의 구체적인 용량이나 복용량(숫자)을
말하거나 묻거나 권하는 문장이면 true, 아니면 false로 판단한다.
혈당 수치(예: 140mg/dL), 음식 양(예: 밥 1공기, 계란 2알), 시간은 약 용량이 아니다.
JSON으로만 답한다: {"results": [true, false, ...]} (문장 개수와 같은 길이)"""


class OpenAIDoseChecker:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def confirm_dose(self, sentences: list[str]) -> list[bool] | None:
        """문장마다 진짜 약·용량 이야기인지. 물어보지 못하면 None."""
        if not sentences:
            return []
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": numbered},
                ],
                response_format={"type": "json_object"},
            )
            results = json.loads(completion.choices[0].message.content or "")["results"]
        except (OpenAIError, KeyError, IndexError, TypeError, ValueError) as e:
            log.warning(
                "[GPT] 확인 실패 - 규칙 결과로 진행 (%s, %s)", PROMPT_VERSION, type(e).__name__
            )
            return None
        if len(results) != len(sentences) or not all(isinstance(r, bool) for r in results):
            log.warning("[GPT] 답 모양이 다름 - 규칙 결과로 진행 (%s)", PROMPT_VERSION)
            return None
        return results
