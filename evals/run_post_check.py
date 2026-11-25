"""커뮤니티 글 위반 감지 평가 (U-15 · U-30).

    uv run python -m evals.run_post_check              # 규칙 + GPT (키가 있으면)
    uv run python -m evals.run_post_check --rules-only # 규칙만 (GPT가 얼마나 거르는지 비교용)

안전 기준: 위급(URGENT_MEDICAL)·마음 위기(MIND)·폭력(SAFETY)·저혈당(LOW_GLUCOSE) **놓침 0건**.
약·용량(DOSE)은 잘못 거는 것(오탐)도 본다.
"""

import argparse

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.moderation.dose_check import PROMPT_VERSION
from app.moderation.router import get_dose_checker
from app.moderation.rules import find_hits
from evals.common import load_cases, make_client, percent, save

KINDS = ("DOSE", "URGENT_MEDICAL", "MIND", "SAFETY", "LOW_GLUCOSE")


def evaluate(client: TestClient, cases: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for case in cases:
        title = case.get("title", "")
        res = client.post("/internal/post-check", json={"title": title, "content": case["content"]})
        res.raise_for_status()  # 글 감지는 GPT가 실패해도 규칙 결과로 200이 온다
        body = res.json()
        got = sorted({issue["kind"] for issue in body["issues"]})
        rule_dose = any(h.kind == "DOSE" for h in find_hits(title, case["content"]))
        rows.append(
            {
                "id": case["id"],
                "expect": sorted(case["expect"]),
                "got": got,
                "llmChecked": body["llmChecked"],
                "gptSkipped": rule_dose and not body["llmChecked"],  # GPT에 물어야 했는데 못 물음
                "ok": got == sorted(case["expect"]),
            }
        )

    summary: dict = {"cases": len(rows), "exact": sum(r["ok"] for r in rows)}
    for kind in KINDS:
        should = [r for r in rows if kind in r["expect"]]
        flagged = [r for r in rows if kind in r["got"]]
        hit = [r for r in should if kind in r["got"]]
        summary[kind] = {
            "should": len(should),  # 걸려야 하는 글
            "caught": len(hit),  # 그중 걸린 글
            "missed": [r["id"] for r in should if kind not in r["got"]],  # 놓친 글
            "falseAlarm": [r["id"] for r in flagged if kind not in r["expect"]],  # 잘못 건 글
        }
    return summary, rows


def print_report(summary: dict) -> None:
    print(f"사례 {summary['cases']}개 · 완전히 맞힘 {percent(summary['exact'], summary['cases'])}")
    print("종류 | 걸려야 함 | 잡음(재현율) | 놓친 사례 | 잘못 건 사례")
    for kind in KINDS:
        k = summary[kind]
        print(
            f"{kind} | {k['should']} | {k['caught']} ({percent(k['caught'], k['should'])}) | "
            f"{', '.join(k['missed']) or '-'} | {', '.join(k['falseAlarm']) or '-'}"
        )
    danger = (
        summary["URGENT_MEDICAL"]["missed"]
        + summary["MIND"]["missed"]
        + summary["SAFETY"]["missed"]
        + summary["LOW_GLUCOSE"]["missed"]
    )
    print(
        "✅ 위급·마음 위기·폭력·저혈당 놓침 0건"
        if not danger
        else f"❌ 위급·마음 위기·폭력·저혈당 놓침 {len(danger)}건"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")  # 처음 보는 문장으로 (보고서 숫자)
    args = parser.parse_args()

    client = make_client()
    rules_only = args.rules_only or not settings.llm_api_key  # 키가 없으면 실제로는 규칙만이다
    if rules_only:
        app.dependency_overrides[get_dose_checker] = lambda: None
    cases = "post_check_fresh.jsonl" if args.fresh else "post_check.jsonl"
    summary, rows = evaluate(client, load_cases(cases))
    summary["mode"] = "rules-only" if rules_only else "rules+gpt"
    summary["set"] = "fresh" if args.fresh else "calibration"
    summary["llmCheckedCases"] = sum(r["llmChecked"] for r in rows)
    summary["gptSkipped"] = [] if rules_only else [r["id"] for r in rows if r["gptSkipped"]]
    print_report(summary)
    print(f"GPT가 확인한 사례 {summary['llmCheckedCases']}개 (모드: {summary['mode']})")
    if summary["gptSkipped"]:
        print(f"⚠️ GPT 확인 실패 {len(summary['gptSkipped'])}건 → 이 실행 숫자는 쓰지 말고 다시")
    model = None if rules_only else settings.llm_model
    name = f"post_check-{summary['set']}-{summary['mode']}"
    print("저장:", save(name, summary, rows, model, None if rules_only else PROMPT_VERSION))


if __name__ == "__main__":
    main()
