"""Ollama 로컬 LLM으로 장면을 이해하고 쇼츠 구성과 나레이션 대본을 만든다."""
from __future__ import annotations

import json
import re

import ollama
from pydantic import BaseModel, ValidationError

from . import config
from .korean_numbers import number_values

# 한자·가나가 섞인 대본은 나레이션 음성이 읽지 못하므로 실패로 보고 다시 요청한다
FOREIGN_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")

FRAME_PROMPT = """이 이미지는 영상의 한 장면입니다. 무엇이 보이는지 한국어 한두 문장으로 객관적으로 설명하세요.
- 인물, 사물, 장소, 행동 위주로 쓰고, 시간대·색감·분위기도 짧게 덧붙이세요.
- 화면에 글자가 있으면 그대로 따옴표 안에 적으세요.
- 추측은 하지 말고, 설명만 쓰세요."""

STYLES = {
    "정보 전달 (차분한 설명)": "차분하고 명확한 정보 전달형. 핵심을 짚어 주는 다큐멘터리 해설처럼.",
    "유쾌한 리뷰": "친근하고 유쾌한 리뷰어 말투. 가벼운 감탄과 리액션을 섞어서.",
    "감성 스토리": "잔잔하고 감성적인 이야기체. 장면의 분위기를 살려서.",
}

SELECT_PROMPT = """당신은 유튜브 쇼츠·인스타그램 릴스 편집자입니다.
긴 영상의 타임라인(장면 설명과 대사)을 보고, 핵심만 골라 {target}초 안팎의 쇼츠에 쓸 구간을 고르세요.

규칙:
- key_points: 먼저 영상 전체의 핵심 내용을 시간 순서대로 3~6개 적으세요.
  단계·팁·목록을 차례로 설명하는 영상이면 항목 하나가 key_point 하나입니다.
- clips: 원본에서 쓸 구간 약 {count}개. 시간 순서대로, 서로 겹치지 않게. 각 구간은 3~15초.
  - covers: 그 구간이 보여 주는 key_points 번호(0부터). 모든 key_point가 어느 구간엔가 들어가야 합니다.
- 구간 길이(end - start)의 합은 {target}초에 가깝게 (±20%). 대사 한 문장만 담은 짧은 구간보다
  장면 하나를 충분히 보여 주는 구간이 좋습니다.
- 원본 대사가 있는 구간은 문장 중간에서 끊기지 않게 대사 시작·끝 시간에 맞추세요.
- title: 화면 위쪽에 계속 보일 쇼츠 제목. 한글 15자 이내."""

NARRATE_PROMPT = """당신은 쇼츠 나레이션 작가입니다. 쇼츠는 아래 구간들을 순서대로 이어 붙인 영상입니다.
각 구간(id)마다 그 구간 화면 위에 깔릴 한국어 나레이션을 쓰세요.

규칙:
- 각 구간의 나레이션은 반드시 그 구간의 excerpt(그 시간의 화면 설명과 대사)에 나온 내용만 다루세요.
  다른 구간의 내용을 미리 말하거나 뒤늦게 말하면 화면과 맞지 않습니다.
- max_chars(공백 포함 글자 수)를 넘지 마세요.
- 첫 구간은 시청자를 붙잡는 한 문장으로 시작하고, 마지막 구간은 짧게 마무리하세요.
- 구간이 이어지는 하나의 이야기처럼 자연스럽게 연결하세요.
- 대사를 그대로 번역해 반복하지 말고, 무슨 일이 일어나는지 설명하거나 해설하세요.
- excerpt에 없는 사실(이름, 숫자, 장소)은 지어내지 마세요. 숫자는 excerpt에 나온 값 그대로 아라비아 숫자로 쓰세요.
- 한글로만 쓰세요. 한자, 일본어, 영어 문장은 쓰지 마세요. 짧은 영어 고유명사는 한글로 적으세요.
- 말투: {style}"""


class FrameNote(BaseModel):
    description: str


class ClipChoice(BaseModel):
    start: float
    end: float
    covers: list[int]


class Selection(BaseModel):
    title: str
    key_points: list[str]
    clips: list[ClipChoice]


class NarrationLine(BaseModel):
    id: int
    narration: str


class NarrationSet(BaseModel):
    lines: list[NarrationLine]


def list_models() -> list[str]:
    return sorted(m.model for m in ollama.list().models)


def _chat(model: str, messages: list[dict], schema: type[BaseModel], keep_alive: str, temperature: float):
    kwargs = dict(
        model=model,
        messages=messages,
        format=schema.model_json_schema(),
        options={"temperature": temperature},
        keep_alive=keep_alive,
    )
    try:
        resp = ollama.chat(**kwargs, think=False)
    except ollama.ResponseError:  # 생각(thinking) 옵션을 모르는 모델
        resp = ollama.chat(**kwargs)
    try:
        return schema.model_validate_json(resp.message.content)
    except ValidationError:
        return None


def describe_frame(model: str, image_path) -> str:
    note = _chat(
        model,
        [{"role": "user", "content": FRAME_PROMPT, "images": [str(image_path)]}],
        FrameNote,
        keep_alive="5m",  # 장면 설명 → 구성까지 계속 쓰므로 유지
        temperature=0.2,
    )
    return note.description.strip() if note else ""


def _ask_until_ok(model, messages, schema, check, attempts: int):
    """응답을 check로 검사해서 문제가 있으면 문제를 알려 주고 다시 쓰게 한다. 문제가 가장 적은 응답을 돌려준다."""
    best, best_problems = None, None
    for _ in range(attempts):
        answer = _chat(model, messages, schema, keep_alive="1m", temperature=0.5)
        if answer is None:
            continue
        problems = check(answer)
        if best is None or len(problems) < len(best_problems):
            best, best_problems = answer, problems
        if not problems:
            break
        messages = messages + [
            {"role": "assistant", "content": answer.model_dump_json()},
            {"role": "user", "content": "다음 문제를 고쳐서 처음부터 다시 작성하세요.\n- " + "\n- ".join(problems)},
        ]
    return best


def select_clips(model: str, timeline: str, duration: float, target_sec: float, attempts: int = 3) -> Selection:
    count = max(2, min(6, round(target_sec / 8)))
    messages = [
        {"role": "system", "content": SELECT_PROMPT.format(target=int(target_sec), count=count)},
        {"role": "user", "content": f"원본 영상 길이: {duration:.1f}초\n\n타임라인:\n{timeline}"},
    ]

    def check(sel: Selection) -> list[str]:
        # LLM은 길이 합계를 잘 맞추지 못하므로, 무엇이 틀렸는지 숫자로 알려 준다
        if not sel.clips:
            return ["clips가 비어 있습니다."]
        total = sum(max(0.0, min(c.end, duration) - max(c.start, 0.0)) for c in sel.clips)
        lengths = ", ".join(f"{c.end - c.start:.1f}" for c in sel.clips)
        problems = []
        if total < 0.75 * target_sec:
            problems.append(f"구간 길이 합계가 {total:.0f}초({lengths})로 목표 {target_sec:.0f}초보다 너무 짧습니다. 구간을 늘리거나 추가하세요.")
        elif total > 1.3 * target_sec:
            problems.append(f"구간 길이 합계가 {total:.0f}초({lengths})로 목표 {target_sec:.0f}초보다 너무 깁니다. 구간을 줄이거나 빼세요.")
        # 목록형 영상에서 "첫째, 셋째"처럼 항목을 빠뜨리는 일이 있어서 핵심 내용을 먼저 적게 하고 확인한다
        covered = {i for c in sel.clips for i in c.covers}
        missing = [f"{i}번 '{p}'" for i, p in enumerate(sel.key_points) if i not in covered]
        if missing:
            problems.append(f"핵심 내용 {', '.join(missing)}이(가) 어느 구간에도 없습니다. 그 내용이 나오는 구간을 넣으세요.")
        if FOREIGN_SCRIPT.search(sel.title):
            problems.append("제목에 한자나 일본어 문자가 있습니다. 한글로만 쓰세요.")
        return problems

    sel = _ask_until_ok(model, messages, Selection, check, attempts)
    if sel is None or not sel.clips:
        raise RuntimeError("쇼츠 구간을 고르지 못했습니다. 다른 모델을 고르거나 다시 시도하세요.")
    sel.title = FOREIGN_SCRIPT.sub("", sel.title).strip()
    return sel


def write_narrations(model: str, items: list[dict], style: str, attempts: int = 3) -> list[str]:
    """items: [{start, end, excerpt}] → 구간별 나레이션. excerpt는 그 구간의 장면 설명과 대사."""
    payload = [
        {
            "id": i,
            "seconds": round(it["end"] - it["start"], 1),
            "max_chars": max(8, int((it["end"] - it["start"]) * config.NARRATION_CHARS_PER_SECOND)),
            "excerpt": it["excerpt"] or "(설명 없음)",
        }
        for i, it in enumerate(items)
    ]
    messages = [
        {"role": "system", "content": NARRATE_PROMPT.format(style=style)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)},
    ]

    def check(result: NarrationSet) -> list[str]:
        got = {ln.id: ln.narration.strip() for ln in result.lines}
        problems = []
        for item in payload:
            text = got.get(item["id"], "")
            if not text:
                problems.append(f"id {item['id']}의 나레이션이 없습니다.")
                continue
            if FOREIGN_SCRIPT.search(text):
                problems.append(f"id {item['id']}에 한자나 일본어 문자가 있습니다. 한글로만 쓰세요.")
            if len(text) > item["max_chars"] * 1.3:
                problems.append(f"id {item['id']}가 {len(text)}자로 max_chars {item['max_chars']}자보다 깁니다.")
            # "한 번", "한 시간"처럼 1은 말버릇으로 흔히 쓰므로 허용
            invented = number_values(text) - number_values(item["excerpt"]) - {1}
            if invented:
                nums = ", ".join(str(n) for n in sorted(invented))
                problems.append(f"id {item['id']}의 숫자 {nums}는 그 구간 excerpt에 없습니다. 지어내지 마세요.")
        return problems

    result = _ask_until_ok(model, messages, NarrationSet, check, attempts)
    got = {ln.id: FOREIGN_SCRIPT.sub("", ln.narration).strip() for ln in result.lines} if result else {}
    return [got.get(i, "") for i in range(len(items))]


def timeline_text(frames: list[dict], segments: list[dict], start: float = 0.0, end: float = float("inf")) -> str:
    """장면 설명과 대사를 시간 순서로 합친 타임라인 (LLM 입력용). start~end와 겹치는 것만."""
    events = [
        (f["t"], f"[{f['t']:.1f}초] 화면: {f['caption']}") for f in frames if f["caption"] and start <= f["t"] <= end
    ]
    events += [
        (s["start"], f"[{s['start']:.1f}~{s['end']:.1f}초] 대사: {s['text']}")
        for s in segments
        if s["end"] > start and s["start"] < end
    ]
    return "\n".join(text for _, text in sorted(events, key=lambda e: e[0]))

