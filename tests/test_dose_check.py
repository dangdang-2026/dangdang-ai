from types import SimpleNamespace

from openai import OpenAIError

from app.moderation.dose_check import OpenAIDoseChecker


class FakeOpenAI:
    """openai.OpenAI 와 같은 모양(client.chat.completions.create)만 흉내 낸다."""

    def __init__(self, answer_text=None, error=None):
        self.answer_text = answer_text
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.answer_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_문장마다_true_false를_돌려준다():
    fake = FakeOpenAI('{"results": [true, false]}')
    result = OpenAIDoseChecker(fake, "gpt-5.4-nano").confirm_dose(["인슐린 8단위", "혈당 140mg"])
    assert result == [True, False]
    call = fake.calls[0]
    assert call["model"] == "gpt-5.4-nano"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][1]["content"] == "1. 인슐린 8단위\n2. 혈당 140mg"


def test_답_개수가_다르면_None():
    fake = FakeOpenAI('{"results": [true]}')
    assert OpenAIDoseChecker(fake, "m").confirm_dose(["a", "b"]) is None


def test_JSON이_아니면_None():
    fake = FakeOpenAI("네 맞아요")
    assert OpenAIDoseChecker(fake, "m").confirm_dose(["인슐린 8단위"]) is None


def test_OpenAI_오류면_None():  # 시간 초과·사용량 초과·키 틀림 모두 OpenAIError 계열
    fake = FakeOpenAI(error=OpenAIError("Request timed out."))
    assert OpenAIDoseChecker(fake, "m").confirm_dose(["인슐린 8단위"]) is None


def test_문장이_없으면_부르지_않는다():
    fake = FakeOpenAI('{"results": []}')
    assert OpenAIDoseChecker(fake, "m").confirm_dose([]) == []
    assert fake.calls == []
