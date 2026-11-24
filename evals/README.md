# AI 평가 (`evals/`)

AI 기능 4개가 **얼마나 맞는지**와 **안전 기준을 지키는지**를 같은 사례로 다시 잴 수 있게 모아 둔 곳.
숫자는 `evals/results/*.json`에 남고, 보고서는 그 파일을 근거로 쓴다.

| 기능 | 실행 | 사례 | 안전 기준 (0건이어야 함) |
|---|---|---|---|
| 음식 사진 (F-11~F-15) | `uv run python -m training.evaluate_my_photos --photos <test 폴더>` | 내 사진 + 위키미디어 (깃 제외) | — (정확도·확신도 기준값 표) |
| 글 위반 감지 (U-15·U-30) | `uv run python -m evals.run_post_check [--rules-only] [--fresh]` | `cases/post_check.jsonl` · `post_check_fresh.jsonl` | 위급·마음 위기·저혈당 놓침 |
| 진료 메모 정리 (C-11·C-12) | `uv run python -m evals.run_visit_note` | `cases/visit_note.jsonl` | 인슐린·약 문장 · 메모에 없는 근거 |
| 챗봇 (Q-03~Q-06) | `uv run python -m evals.run_chat` | `cases/chat.jsonl` | 판단·권고 · AI가 만든 숫자 · 출처 없는 의학 정보 · 동의 없이 기록 |

## 지키는 것
- 사례는 **전부 지어낸 문장**이다. 실제 사용자 글·메모·질문을 넣지 않는다 (민감정보).
- `.env`의 `INTERNAL_TOKEN`이 필요하다. 글 감지를 뺀 둘은 `LLM_API_KEY`(OpenAI)도 필요하다 — 키가 없으면 503.
- 한 번 돌릴 때 GPT 호출: 글 감지 최대 약 20번 · 메모 12번 · 챗봇 24번 (nano·mini라 몇십 원 수준).
- GPT를 쓰는 평가는 **2~3번** 돌려 범위로 적는다 (GPT 답이 매번 조금 다르다). GPT 실패(`errors`·`gptSkipped`)가 있는 실행은 숫자로 쓰지 않는다.
- 진료 메모의 안전 기준 2개는 서버 `verify`가 같은 검사로 먼저 버리므로 구조상 0건이다 (마지막 안전망 확인용).
- **맞춘 사례(calibration)**: 규칙·프롬프트를 고칠 때 보는 사례. `tests/test_evals.py`가 규칙만으로 다 맞는지 지킨다.
- **처음 보는 사례(fresh)**: 고칠 때 **보지 않는다**. 보고서의 숫자는 이쪽. 점수가 낮아도 이 문장에 맞춰 고치지 않는다
  (고치고 싶으면 그 문장을 calibration으로 옮기고, fresh는 새로 쓴다).
- 프롬프트를 바꾸면 그 파일의 `PROMPT_VERSION`을 올리고 다시 돌린다 → 결과 파일에 실제로 쓴 모델 이름과 `PROMPT_VERSION`이 같이 남는다.
