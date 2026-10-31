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
{list_rules}- max_chars(공백 포함 글자 수)를 넘지 마세요.
- 첫 구간은 시청자를 붙잡는 한 문장으로 시작하고, 마지막 구간은 짧게 마무리하세요.
- 구간이 이어지는 하나의 이야기처럼 자연스럽게 연결하세요.
- 대사를 그대로 번역해 반복하지 말고, 무슨 일이 일어나는지 설명하거나 해설하세요.
- excerpt에 없는 사실(이름, 숫자, 장소)은 지어내지 마세요. 숫자는 excerpt에 나온 값 그대로 아라비아 숫자로 쓰세요.
- 한글로만 쓰세요. 한자, 일본어, 영어 문장은 쓰지 마세요. 짧은 영어 고유명사는 한글로 적으세요.
- 말투: {style}"""


# 번호 붙은 목록 영상일 때만 넣는 규칙. 늘 넣으면 목록이 아닌 영상에도 "둘째, 셋째"를 붙였다.
LIST_RULES = """- 이 영상은 번호 붙은 목록(팁·단계)입니다. excerpt에 항목이 여러 개 나오면 하나도 빠뜨리지 말고 모두 짧게 담으세요.
- 번호 붙은 항목(화면 "TIP 2", 대사 "Second")이 나오는 구간은 "둘째,"처럼 그 번호의 순서 표현으로 시작해 그 항목 내용을 쓰세요.
"""


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


def tidy(text: str) -> str:
    """LLM 글 다듬기: 한자·가나 제거, '5 시에'·'18 도'처럼 숫자와 단위 사이 띄어쓰기 붙이기."""
    text = FOREIGN_SCRIPT.sub("", text)
    return re.sub(r"(\d)\s+(?=[가-힣])", r"\1", text).strip()


MAX_ANSWER_TOKENS = 4096  # 가장 긴 정상 응답(대본·나레이션 묶음)도 2000토큰 안쪽
DEVICE = "gpu"  # 계산 장치 설정. "cpu"면 Ollama 모델을 GPU에 올리지 않는다 (gpu·hybrid는 Ollama가 알아서 나눔)


def list_models() -> list[str]:
    return sorted(m.model for m in ollama.list().models)


def pull_model(name: str):
    """Ollama 모델을 내려받는다. 진행 상황을 (상태, 받은 바이트, 전체 바이트)로 차례로 돌려준다."""
    for part in ollama.pull(name, stream=True):
        yield part.status or "", part.completed or 0, part.total or 0


def _chat(model: str, messages: list[dict], schema: type[BaseModel], keep_alive: str, temperature: float):
    kwargs = dict(
        model=model,
        messages=messages,
        format=schema.model_json_schema(),
        # num_predict: 같은 내용을 끝없이 되풀이하는 응답을 끊는다. 상한이 없을 때 쇼츠 여러 편 나누기 응답 하나가
        # 문맥 한도(131072 토큰)까지 26분 동안 이어진 적이 있다. 잘린 응답은 JSON이 깨져 None → 다시 요청한다.
        options={"temperature": temperature, "num_predict": MAX_ANSWER_TOKENS,
                 **({"num_gpu": 0} if DEVICE == "cpu" else {})},
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


SCRIPT_PROMPT = """당신은 유튜브 쇼츠·인스타그램 릴스 작가입니다. 원본 영상 없이, 주제만으로 {target}초 안팎의 쇼츠를 기획하세요.
화면은 장면마다 AI가 그림(또는 짧은 영상)으로 만들고, 그 위에 한국어 나레이션과 자막이 나옵니다.

규칙:
- scenes: {count}개 안팎(3~8개). 장면 길이 seconds는 3~8초, 합계는 {target}초에 가깝게 (±20%).
- narration: 그 장면 위에 깔릴 한국어 나레이션. seconds × {cps}자를 넘지 않게.
  - 첫 장면은 시청자를 붙잡는 한 문장, 마지막 장면은 짧은 마무리.
  - 장면이 이어지는 하나의 이야기처럼.
  - 주제에 없는 구체적인 숫자·통계·사람 이름·날짜는 지어내지 마세요. 확실하지 않은 사실은 쓰지 마세요.
  - 한글로만 쓰세요. 한자, 일본어, 영어 문장은 쓰지 마세요.
- subject: 이 쇼츠의 주인공 피사체를 영어 명사구 하나로 (예: commercial airliner, kimchi fried rice, tabby cat,
  autumn mountain lake). 주제에서 화면에 꼭 보여야 하는 것입니다.
- visual_ko: 그 장면에 보일 그림을 한국어 한 문장으로 (사람이 읽고 고칠 설명). 주인공 피사체가 보이게.
- visual: 같은 그림을 이미지 생성 모델용 영어 프롬프트로 (피사체, 장소, 행동, 빛, 구도, 40단어 이내).
  - 생성 모델은 장면마다 따로 그려서 주제를 모릅니다. 모든 visual에 subject의 낱말을 그대로 넣고,
    주인공 피사체가 화면에 크게, 분명히 보이게 쓰세요. 사람 얼굴·손만, 빈 풍경만, 컴퓨터 화면만 나오는 장면은 쓰지 마세요.
  - 뜻이 여러 가지인 낱말 대신 구체적인 이름을 쓰세요 (예: "engine block" 대신 "turbofan jet engine mounted on the airliner wing").
  - 그림 안에 글자·로고·워터마크가 나오지 않게. 장면마다 그림이 서로 달라야 합니다.
- title: 화면 위쪽 제목. 한글 15자 이내.
- 말투: {style}"""


class Scene(BaseModel):
    narration: str
    visual_ko: str
    visual: str
    seconds: float


class Script(BaseModel):
    title: str
    subject: str = ""
    scenes: list[Scene]


SUBJECT_STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "for", "to", "its", "their"}


def subject_words(subject: str) -> list[str]:
    """주인공 피사체 명사구에서 프롬프트에 있는지 볼 낱말 (관사·전치사 제외, 3글자 이상)."""
    return [w for w in re.findall(r"[a-z]+", subject.lower()) if len(w) >= 3 and w not in SUBJECT_STOP]


def has_subject(prompt: str, subject: str) -> bool:
    """영어 프롬프트에 주인공 피사체가 들어 있는지 (명사구의 핵심 낱말 = 마지막 낱말, 또는 낱말 절반 이상)."""
    words = subject_words(subject)
    if not words:
        return True
    found = [w for w in words if re.search(rf"\b{w}", prompt.lower())]
    return words[-1] in found or len(found) * 2 >= len(words)


def anchor_subject(prompt: str, subject: str) -> str:
    """주인공 피사체가 빠진 프롬프트 앞에 피사체를 붙인다 (생성 모델은 장면마다 따로 그려 주제를 모름)."""
    prompt = prompt.strip()
    if not subject.strip() or has_subject(prompt, subject):
        return prompt
    return f"{subject.strip()}, {prompt}"


def write_script(model: str, topic: str, target_sec: float, style: str, attempts: int = 3) -> Script:
    """주제만으로 쇼츠 대본: 장면마다 나레이션, 그림 설명(한글), 생성용 영어 프롬프트, 길이."""
    count = max(3, min(8, round(target_sec / 5)))
    cps = config.NARRATION_CHARS_PER_SECOND
    messages = [
        {"role": "system", "content": SCRIPT_PROMPT.format(target=int(target_sec), count=count, cps=cps, style=style)},
        {"role": "user", "content": f"주제: {topic.strip()}"},
    ]

    def check(script: Script) -> list[str]:
        problems = []
        scenes = script.scenes
        if not 3 <= len(scenes) <= 8:
            problems.append(f"scenes가 {len(scenes)}개입니다. 3~8개로 쓰세요.")
        total = sum(s.seconds for s in scenes)
        if scenes and not 0.75 * target_sec <= total <= 1.3 * target_sec:
            problems.append(f"장면 길이 합계가 {total:.0f}초입니다. {target_sec:.0f}초에 가깝게 맞추세요.")
        for i, s in enumerate(scenes):
            if not 2.5 <= s.seconds <= 8.5:
                problems.append(f"{i}번 장면 길이 {s.seconds}초를 3~8초로 바꾸세요.")
            if not s.narration.strip() or FOREIGN_SCRIPT.search(s.narration):
                problems.append(f"{i}번 장면 나레이션을 한글로 쓰세요.")
            if len(s.narration) > s.seconds * cps * 1.3:
                problems.append(f"{i}번 장면 나레이션이 {len(s.narration)}자로 너무 깁니다 ({int(s.seconds * cps)}자 이내).")
            if re.search(r"[가-힣]", s.visual) or not s.visual.strip():
                problems.append(f"{i}번 장면 visual을 영어로만 쓰세요.")
        if not subject_words(script.subject) or re.search(r"[가-힣]", script.subject):
            problems.append("subject에 주인공 피사체를 영어 명사구로 쓰세요 (예: commercial airliner).")
        else:
            missing = [i for i, s in enumerate(scenes) if not has_subject(s.visual, script.subject)]
            if missing:
                problems.append(f"{', '.join(map(str, missing))}번 장면 visual에 주인공 피사체 '{script.subject}'가 없습니다. "
                                "그 낱말을 넣고 피사체가 화면에 분명히 보이게 다시 쓰세요.")
        if len(re.findall(r"[가-힣]", script.title)) < 4 or FOREIGN_SCRIPT.search(script.title):
            problems.append("title을 내용을 담은 한글 제목(4~15자)으로 쓰세요.")
        # 주제에 없는 숫자("5분", "3가지")를 지어내지 않게 ("한 번"의 1은 허용)
        written = " ".join([script.title] + [s.narration for s in scenes])
        invented = number_values(written) - number_values(topic) - {1}
        if invented:
            problems.append(f"주제에 없는 숫자 {', '.join(map(str, sorted(invented)))}를 쓰지 마세요.")
        return problems

    script = _ask_until_ok(model, messages, Script, check, attempts)
    if script is None or not script.scenes:
        raise RuntimeError("대본을 만들지 못했습니다. 주제를 조금 더 구체적으로 쓰거나 다시 시도하세요.")
    script.title = tidy(script.title)
    script.subject = script.subject.strip() if not re.search(r"[가-힣]", script.subject) else ""
    for s in script.scenes:
        s.narration, s.seconds = tidy(s.narration), float(min(8.0, max(3.0, s.seconds)))
        s.visual = anchor_subject(s.visual, script.subject)  # 다시 받아도 빠졌으면 앞에 붙임
    return script


UPLOAD_PROMPT = """당신은 유튜브 쇼츠·인스타그램 릴스 운영자입니다. 완성된 쇼츠의 제목과 나레이션을 보고, 올릴 때 쓸 글을 쓰세요.

규칙:
- titles: 업로드 제목 후보 3개. 각각 한글 위주 40자 이내, 서로 다른 느낌(궁금증 / 정보 요약 / 감성 등). 낚시성 과장 금지.
- description: 설명란 글. 2~3문장, 200자 이내. 영상 내용을 요약하고 마지막에 가벼운 질문이나 권유 한 문장.
- hashtags: 해시태그 5~8개. 각각 #으로 시작하고 띄어쓰기 없이. 첫 번째는 #Shorts, 나머지는 내용에 맞는 한국어 위주.
- 나레이션과 제목에 없는 숫자·통계·사람 이름·장소 이름·날짜는 지어내지 마세요."""


class UploadText(BaseModel):
    titles: list[str]
    description: str
    hashtags: list[str]


def write_upload_text(model: str, title: str, script: str, attempts: int = 3) -> UploadText:
    """완성된 쇼츠의 업로드 제목 후보 3개, 설명, 해시태그."""
    messages = [
        {"role": "system", "content": UPLOAD_PROMPT},
        {"role": "user", "content": f"화면 제목: {title.strip() or '(없음)'}\n\n나레이션:\n{script.strip()}"},
    ]
    source = f"{title} {script}"

    def check(answer: UploadText) -> list[str]:
        problems = []
        if len(answer.titles) != 3:
            problems.append(f"titles가 {len(answer.titles)}개입니다. 정확히 3개로 쓰세요.")
        for t in answer.titles:
            if len(t) > 45 or len(re.findall(r"[가-힣]", t)) < 4 or FOREIGN_SCRIPT.search(t):
                problems.append(f"제목 「{t}」을 한글 위주 40자 이내로 고치세요.")
        if not answer.description.strip() or len(answer.description) > 260 or FOREIGN_SCRIPT.search(answer.description):
            problems.append(f"description이 {len(answer.description)}자입니다. 한글로 200자 이내로 쓰세요.")
        tags = [h.strip() for h in answer.hashtags]
        if not 5 <= len(tags) <= 8:
            problems.append(f"hashtags가 {len(tags)}개입니다. 5~8개로 쓰세요.")
        bad = [h for h in tags if not re.fullmatch(r"#[^\s#]{1,30}", h)]
        if bad:
            problems.append(f"해시태그 {', '.join(bad)}를 #으로 시작하고 띄어쓰기 없이 쓰세요.")
        written = " ".join(answer.titles + [answer.description] + tags)
        invented = number_values(written) - number_values(source) - {1}
        if invented:
            problems.append(f"나레이션에 없는 숫자 {', '.join(map(str, sorted(invented)))}를 쓰지 마세요.")
        return problems

    answer = _ask_until_ok(model, messages, UploadText, check, attempts)
    if answer is None:
        raise RuntimeError("업로드 문구를 만들지 못했습니다. 다시 시도하세요.")
    tags = []
    for h in ["#Shorts"] + [h.strip() for h in answer.hashtags]:
        h = "#" + re.sub(r"[\s#]+", "", h)
        if len(h) > 1 and h.lower() not in {t.lower() for t in tags}:
            tags.append(h)
    return UploadText(titles=[tidy(t) for t in answer.titles[:3]], description=answer.description.strip(),
                      hashtags=tags[:8])


TOPIC_PROMPT = """당신은 긴 영상을 여러 편의 쇼츠로 나누는 편집자입니다.
타임라인(장면 설명과 대사)을 보고, 서로 다른 주제의 쇼츠 {count}편을 만들 수 있게 원본을 {count}부분으로 나누세요.

규칙:
- topics: 정확히 {count}개. 시간 순서대로, 서로 겹치지 않게.
- 각 부분(start~end)은 그 쇼츠에 쓸 장면이 모두 들어 있는 원본 범위입니다. 최소 {min_len}초.
- 각 부분은 혼자 봐도 이해되는 하나의 주제여야 합니다. 비슷한 주제를 두 번 고르지 마세요.
- title: 그 쇼츠의 한글 제목 (15자 이내), angle: 그 쇼츠가 다룰 내용 한 줄."""


class Topic(BaseModel):
    title: str
    start: float
    end: float
    angle: str


class TopicSet(BaseModel):
    topics: list[Topic]


def suggest_topics(model: str, timeline: str, duration: float, count: int, target_sec: float,
                   attempts: int = 3) -> list[Topic]:
    """긴 영상을 서로 다른 주제의 쇼츠 count편으로 나눌 원본 범위들."""
    min_len = int(min(target_sec, duration / count) * 0.6)
    messages = [
        {"role": "system", "content": TOPIC_PROMPT.format(count=count, min_len=min_len)},
        {"role": "user", "content": f"원본 영상 길이: {duration:.1f}초\n\n타임라인:\n{timeline}"},
    ]

    def check(result: TopicSet) -> list[str]:
        topics = sorted(result.topics, key=lambda t: t.start)
        problems = []
        if len(topics) != count:
            problems.append(f"topics가 {len(topics)}개입니다. 정확히 {count}개로 나누세요.")
        for a, b in zip(topics, topics[1:]):
            if b.start < a.end - 0.5:
                problems.append(f"'{a.title}'({a.start:.0f}~{a.end:.0f}초)와 '{b.title}'({b.start:.0f}~{b.end:.0f}초)가 겹칩니다.")
        for t in topics:
            if t.end - t.start < min_len:
                problems.append(f"'{t.title}'({t.start:.0f}~{t.end:.0f}초)가 {min_len}초보다 짧습니다.")
            if t.start < -0.5 or t.end > duration + 0.5:
                problems.append(f"'{t.title}'이(가) 원본 길이 {duration:.0f}초를 벗어납니다.")
        return problems

    result = _ask_until_ok(model, messages, TopicSet, check, attempts)
    if result is None or not result.topics:
        raise RuntimeError("쇼츠 주제를 나누지 못했습니다. 쇼츠 개수를 줄이거나 다시 시도하세요.")
    topics = sorted(result.topics, key=lambda t: t.start)[:count]
    for t in topics:
        t.start, t.end, t.title = max(0.0, t.start), min(duration, t.end), tidy(t.title)
    return [t for t in topics if t.end - t.start >= 3]


def select_clips(model: str, timeline: str, duration: float, target_sec: float, attempts: int = 3,
                 span: tuple[float, float] | None = None, focus: str = "") -> Selection:
    """span을 주면 원본의 그 범위 안에서만 고른다 (여러 편으로 나눈 쇼츠 한 편). focus는 그 편의 주제."""
    count = max(2, min(6, round(target_sec / 8)))
    lo, hi = span or (0.0, duration)
    where = f"원본 영상 길이: {duration:.1f}초"
    if span:
        where += f"\n이 쇼츠는 원본 {lo:.1f}~{hi:.1f}초 부분에서만 구간을 고르세요."
    if focus:
        where += f"\n이 쇼츠의 주제: {focus}"
    messages = [
        {"role": "system", "content": SELECT_PROMPT.format(target=int(target_sec), count=count)},
        {"role": "user", "content": f"{where}\n\n타임라인:\n{timeline}"},
    ]

    def check(sel: Selection) -> list[str]:
        # LLM은 길이 합계를 잘 맞추지 못하므로, 무엇이 틀렸는지 숫자로 알려 준다
        if not sel.clips:
            return ["clips가 비어 있습니다."]
        outside = [f"{c.start:.1f}~{c.end:.1f}초" for c in sel.clips if c.start < lo - 1 or c.end > hi + 1]
        if outside:
            return [f"구간 {', '.join(outside)}이(가) 이 쇼츠의 범위({lo:.1f}~{hi:.1f}초) 밖입니다."]
        total = sum(max(0.0, min(c.end, duration) - max(c.start, 0.0)) for c in sel.clips)
        lengths = ", ".join(f"{c.end - c.start:.1f}" for c in sel.clips)
        problems = []
        if total < 0.75 * target_sec:
            problems.append(f"구간 길이 합계가 {total:.0f}초({lengths})로 목표 {target_sec:.0f}초보다 너무 짧습니다. 구간을 늘리거나 추가하세요.")
        elif total > 1.3 * target_sec:
            problems.append(f"구간 길이 합계가 {total:.0f}초({lengths})로 목표 {target_sec:.0f}초보다 너무 깁니다. 구간을 줄이거나 빼세요.")
        # 긴 구간 하나에 항목 여러 개가 들어가면 나레이션이 다른 화면의 항목을 말하기 쉬워서 나누게 한다
        too_long = [f"{c.start:.1f}~{c.end:.1f}초" for c in sel.clips if c.end - c.start > 15.5]
        if too_long:
            problems.append(f"구간 {', '.join(too_long)}이(가) 15초보다 깁니다. 항목·장면별로 나누세요.")
        # 목록형 영상에서 "첫째, 셋째"처럼 항목을 빠뜨리는 일이 있어서 핵심 내용을 먼저 적게 하고 확인한다
        covered = {i for c in sel.clips for i in c.covers}
        missing = [f"{i}번 '{p}'" for i, p in enumerate(sel.key_points) if i not in covered]
        if missing:
            problems.append(f"핵심 내용 {', '.join(missing)}이(가) 어느 구간에도 없습니다. 그 내용이 나오는 구간을 넣으세요.")
        if FOREIGN_SCRIPT.search(sel.title):
            problems.append("제목에 한자나 일본어 문자가 있습니다. 한글로만 쓰세요.")
        if len(re.findall(r"[가-힣]", sel.title)) < 4:  # 화면 확인 중 제목이 "3" 하나만 나온 적이 있음
            problems.append(f"제목 '{sel.title}'이(가) 너무 짧습니다. 영상 내용을 담은 한글 제목(4~15자)을 쓰세요.")
        return problems

    sel = _ask_until_ok(model, messages, Selection, check, attempts)
    if sel is None or not sel.clips:
        raise RuntimeError("쇼츠 구간을 고르지 못했습니다. 다른 모델을 고르거나 다시 시도하세요.")
    sel.title = tidy(sel.title)
    return sel


def write_narrations(model: str, items: list[dict], style: str, attempts: int = 3) -> list[str]:
    """items: [{start, end, excerpt}] → 구간별 나레이션. excerpt는 그 구간의 장면 설명과 대사.

    모든 구간을 한 번에 쓰게 한 뒤 검사하고, 그래도 문제가 남은 구간은 그 구간만 따로 다시 쓰게 한다
    (한 번에 쓰면 LLM이 다른 구간의 내용을 옮겨 쓰는 일이 있어서, 한 구간씩 고치면 더 잘 맞는다).
    """
    payload = [
        {
            "id": i,
            "seconds": round(it["end"] - it["start"], 1),
            "max_chars": max(8, int((it["end"] - it["start"]) * config.NARRATION_CHARS_PER_SECOND)),
            "excerpt": it["excerpt"] or "(설명 없음)",
        }
        for i, it in enumerate(items)
    ]
    is_list = any(excerpt_ordinals(item["excerpt"]) for item in payload)
    system = NARRATE_PROMPT.format(style=style, list_rules=LIST_RULES if is_list else "")

    def item_problems(item: dict, text: str) -> list[str]:
        i = item["id"]
        if not text:
            return [f"id {i}의 나레이션이 없습니다."]
        problems = []
        if FOREIGN_SCRIPT.search(text):
            problems.append(f"id {i}에 한자나 일본어 문자가 있습니다. 한글로만 쓰세요.")
        if len(text) > item["max_chars"] * 1.3:
            problems.append(f"id {i}가 {len(text)}자로 max_chars {item['max_chars']}자보다 깁니다.")
        # "한 번", "한 시간"처럼 1은 말버릇으로 흔히 쓰므로 허용
        invented = number_values(text) - number_values(item["excerpt"]) - {1}
        if invented:
            problems.append(f"id {i}의 숫자 {', '.join(map(str, sorted(invented)))}는 그 구간 excerpt에 없습니다. 지어내지 마세요.")
        told, shown = ordinals(text), excerpt_ordinals(item["excerpt"])
        if not is_list:
            if told:
                problems.append(f"id {i}: 이 영상은 번호 붙은 목록이 아닙니다. '둘째', '셋째' 같은 순서 표현을 쓰지 마세요.")
            return problems
        # "둘째, 스마트폰…"을 셋째 팁 화면에 쓰는 일이 있었다. 구간 excerpt에 번호가 있으면 그 번호만 쓰게 한다
        wrong = sorted(told - shown) if shown else []
        if wrong:
            problems.append(f"id {i}의 '{', '.join(f'{n}번째' for n in wrong)}' 항목은 이 구간 화면·대사에 없습니다 "
                            f"(이 구간: {', '.join(f'{n}번째' for n in sorted(shown))}). 그 항목이 나오는 구간에 쓰세요.")
        # 반대로 구간에 번호 붙은 항목이 있는데 나레이션이 말하지 않으면 빠진 것 ("소개합니다"만 하고 첫째 팁을 뺌)
        untold = sorted(shown - told)
        if untold:
            problems.append(f"id {i} 구간에는 {', '.join(f'{n}번째' for n in untold)} 항목이 나오는데 나레이션에 없습니다. "
                            f"'첫째', '둘째'처럼 순서 표현과 함께 그 내용을 쓰세요.")
        return problems

    def check(result: NarrationSet) -> list[str]:
        got = {ln.id: ln.narration.strip() for ln in result.lines}
        problems = [p for item in payload for p in item_problems(item, got.get(item["id"], ""))]
        if is_list:  # "세 가지 팁을 소개합니다 → 두 번째로…"처럼 순서 표현이 건너뛰면 그 항목이 빠진 것
            orders = sorted({n for text in got.values() for n in ordinals(text)})
            skipped = [n for n in range(1, max(orders, default=0) + 1) if n not in orders]
            if skipped:
                problems.append(f"순서 표현에서 {', '.join(f'{n}번째' for n in skipped)} 항목이 빠졌습니다. "
                                "그 내용이 나오는 구간 나레이션에 넣으세요.")
        return problems

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}]
    result = _ask_until_ok(model, messages, NarrationSet, check, attempts)
    got = {ln.id: ln.narration.strip() for ln in result.lines} if result else {}

    for item in payload:  # 문제가 남은 구간만 한 구간씩 다시
        before = item_problems(item, got.get(item["id"], ""))
        if not before:
            continue
        context = {
            "앞 구간 나레이션": got.get(item["id"] - 1, ""),
            "뒤 구간 나레이션": got.get(item["id"] + 1, ""),
            "고칠 구간": item,
            "지금 나레이션의 문제": before,
        }
        single = [{"role": "system", "content": system},
                  {"role": "user", "content": "아래 '고칠 구간' 하나의 나레이션만 다시 쓰세요. lines에는 그 id 하나만 넣으세요.\n"
                                              + json.dumps(context, ensure_ascii=False, indent=1)}]

        def check_one(res: NarrationSet, item=item) -> list[str]:
            text = next((ln.narration.strip() for ln in res.lines if ln.id == item["id"]), "")
            return item_problems(item, text)

        fixed = _ask_until_ok(model, single, NarrationSet, check_one, 2)
        text = next((ln.narration.strip() for ln in fixed.lines if ln.id == item["id"]), "") if fixed else ""
        if text and len(item_problems(item, text)) < len(before):
            got[item["id"]] = text
    return [tidy(got.get(i, "")) for i in range(len(items))]


class PromptText(BaseModel):
    prompt: str


VISUAL_PROMPT = """You write prompts for an image/video generation model.
Turn the user's scene description (often Korean) into one English prompt: subject, setting, action, lighting, camera.
Keep it under 40 words. No text, letters, logos or watermarks in the scene. Output only the prompt."""

MUSIC_PROMPT = """You write prompts for a music generation model (MusicGen).
Turn the user's request (often Korean) into one short English prompt: genre, mood, instruments, tempo.
Instrumental only, no vocals. Under 25 words. Output only the prompt."""


def english_prompt(model: str, text: str, kind: str = "visual", subject: str = "") -> str:
    """장면·음악 설명을 생성 모델용 영어 프롬프트로. 이미 영어면 그대로 쓴다.
    subject(쇼츠의 주인공 피사체, 영어)를 주면 장면 프롬프트에 꼭 들어가게 한다 (사람이 장면 설명을 고친 경우)."""
    text = text.strip()
    if not text or not re.search(r"[가-힣]", text):
        return anchor_subject(text, subject) if kind == "visual" else text
    system = VISUAL_PROMPT if kind == "visual" else MUSIC_PROMPT
    if kind == "visual" and subject.strip():
        system += f"\nThe main subject of the whole video is: {subject.strip()}. It must be clearly visible; name it in the prompt."

    def check(r: PromptText) -> list[str]:
        if re.search(r"[가-힣]", r.prompt) or not r.prompt.strip():
            return ["영어로만 쓰세요."]
        if kind == "visual" and not has_subject(r.prompt, subject):
            return [f"Include the main subject '{subject}' in the prompt."]
        return []

    result = _ask_until_ok(
        model, [{"role": "system", "content": system}, {"role": "user", "content": text}], PromptText, check, 3,
    )
    prompt = result.prompt.strip() if result and not re.search(r"[가-힣]", result.prompt) else text
    return anchor_subject(prompt, subject) if kind == "visual" else prompt


ORDINAL_WORDS = {"첫": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7}
ORDINAL = re.compile(r"(첫|둘|셋|넷|다섯|여섯|일곱)째|(첫|두|세|네|다섯|여섯|일곱)\s?번째|(첫)\s?(?:번째|단계|팁)")


FIRST_WORDS = re.compile(r"(?<![가-힣])(먼저|우선|처음으로)(?=[,\s])")  # "먼저 … 두 번째로 …"의 "먼저"는 첫 번째


def ordinals(text: str) -> set[int]:
    """'첫째', '두 번째', '세번째' 같은 순서 표현의 번호. 두 번째 이상이 있을 때는 '먼저'도 첫 번째로 센다."""
    found = {ORDINAL_WORDS[next(g for g in m.groups() if g)] for m in ORDINAL.finditer(text)}
    if found and FIRST_WORDS.search(text):
        found.add(1)
    return found


EXCERPT_ORDINAL = re.compile(r"\b(?:tip|step|part|no\.?|#)\s*(\d{1,2})\b", re.I)
ENGLISH_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def excerpt_ordinals(text: str) -> set[int]:
    """구간 원본(화면 글자·대사)에 나오는 항목 번호: 'TIP 2', 'Second,', '두 번째' 등."""
    found = ordinals(text) | {int(m.group(1)) for m in EXCERPT_ORDINAL.finditer(text)}
    found |= {n for word, n in ENGLISH_ORDINALS.items() if re.search(rf"\b{word}\b", text, re.I)}
    return found


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

