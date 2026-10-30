"""쇼츠 스타일 한글 자막(ASS) 생성."""
from __future__ import annotations

import re
from pathlib import Path

# libass는 컬러 이모지를 그리지 못하고 흑백 윤곽이나 빈 상자로 그린다. 자막·제목에서는 뺀다.
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def clean_text(text: str) -> str:
    return re.sub(r"\s{2,}", " ", EMOJI.sub("", text)).strip()


# (폰트 파일, libass에서 찾을 폰트 이름) — 위에서부터 먼저 찾은 것을 사용
FONT_CANDIDATES = [
    (Path("C:/Windows/Fonts/malgunbd.ttf"), "Malgun Gothic"),
    (Path("C:/Windows/Fonts/NotoSansKR-VF.ttf"), "Noto Sans KR"),
    (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"), "Noto Sans CJK KR"),
    (Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"), "NanumGothic"),
    (Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"), "Apple SD Gothic Neo"),
]


def find_korean_font() -> tuple[Path, str]:
    for path, family in FONT_CANDIDATES:
        if path.exists():
            return path, family
    raise RuntimeError("한글 폰트를 찾을 수 없습니다. 맑은 고딕, Noto Sans KR, 나눔고딕 중 하나를 설치하세요.")


def chars_per_line(font_size: int, width: int, side_margin: int = 60) -> int:
    return max(6, int((width - 2 * side_margin) / font_size))


def wrap(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:  # 띄어쓰기 없이 긴 단어는 강제로 나눔
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _chars(text: str) -> str:
    return re.sub(r"[\s\W_]", "", text)


def word_times(text: str, start: float, duration: float, engine_words: list[dict] | None = None,
               lead: float = 0.0) -> list[tuple[float, float]]:
    """문장의 띄어쓰기 단위 낱말마다 (시작, 끝) 시간.

    엔진이 단어 시간을 주면(Edge TTS) 그 시간을 쓰고 (음성 앞 무음을 잘라 낸 만큼 lead를 빼서 맞춤),
    못 주거나(Chatterbox) 낱말과 맞지 않으면 글자 수 비율로 나눈다.
    """
    tokens = text.split()
    if not tokens:
        return []
    if engine_words:
        times, k = [], 0
        for token in tokens:
            need, got, first = _chars(token), "", None
            while k < len(engine_words) and len(got) < len(need):
                first = engine_words[k] if first is None else first
                got += _chars(engine_words[k]["text"])
                k += 1
            if first is None or got != need:  # "18도"를 "십팔도"로 돌려주는 등 맞지 않으면 비율로
                break
            times.append((start + first["start"] - lead, start + engine_words[k - 1]["end"] - lead))
        else:
            if all(start - 0.2 <= a <= b <= start + duration + 0.3 for a, b in times):
                return [(max(start, a), min(start + duration, b)) for a, b in times]
    weights = [max(1, len(_chars(t))) for t in tokens]
    total, t, times = sum(weights), start, []
    for wgt in weights:
        share = duration * wgt / total
        times.append((t, t + share))
        t += share
    return times


Cue = tuple[float, float, list[str], list[list[tuple[float, float]]] | None]


def build_cues(rows: list[dict], max_chars: int, max_lines: int = 2) -> list[Cue]:
    """한 문장의 자막이 길면 여러 화면으로 나눈다. 단어 시간이 있으면 화면마다 실제 읽는 시간에 맞추고
    (start, end, 줄 목록, 줄마다 낱말 시간)을, 없으면 글자 수 비율로 나눈 시간을 돌려준다."""
    cues: list[Cue] = []
    for row in rows:
        text = clean_text(row["text"])
        if not text:
            continue
        tokens = row["text"].split()
        times = row.get("words") or []
        kept = [(clean_text(tok), tm) for tok, tm in zip(tokens, times) if clean_text(tok)]
        lines = wrap(text, max_chars)
        groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
        line_tokens = [ln.split() for ln in lines]
        aligned = len(times) == len(tokens) and [t for lt in line_tokens for t in lt] == [t for t, _ in kept]
        if aligned:
            k = 0
            for g, group in enumerate(groups):
                per_line = []
                for ln in line_tokens[g * max_lines : g * max_lines + len(group)]:
                    per_line.append([tm for _, tm in kept[k : k + len(ln)]])
                    k += len(ln)
                flat = [tm for line in per_line for tm in line]
                end = kept[k][1][0] if g + 1 < len(groups) else row["end"]  # 다음 화면의 첫 낱말 전까지
                cues.append((flat[0][0] if g else row["start"], max(end, flat[-1][1]), group, per_line))
            continue
        total = sum(len("".join(g)) for g in groups)
        t = row["start"]
        for group in groups:
            share = (row["end"] - row["start"]) * len("".join(group)) / total
            cues.append((t, t + share, group, None))
            t += share
    return cues


# 제목 모양. 화면 문구는 app.py, 설정 이름은 presets.CHOICES["title_style"]
TITLE_STYLES = ("bold", "box", "band")

# 쇼츠 화면에서 앱 UI가 가리는 곳 (1080x1920 기준 비율, 참고용). 공식 수치가 없어 여러 안내의 넉넉한 쪽을 썼다:
# 위 상태 표시줄·검색 (약 100~200px), 아래 제목·채널·설명 (약 180px 이상, 설명이 길면 더), 오른쪽 좋아요·댓글 버튼 열 (약 80~120px).
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT = 0.10, 0.20, 0.11


def title_keyword(tokens: list[str]) -> int:
    """제목에서 색으로 강조할 낱말: 숫자가 든 첫 낱말, 없으면 마지막 낱말."""
    for i, tok in enumerate(tokens):
        if re.search(r"\d", tok):
            return i
    return len(tokens) - 1


def _ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(lines: list[str]) -> str:
    return r"\N".join(ln.replace("{", "(").replace("}", ")") for ln in lines)


def _ass_rgb(hex_color: str) -> str:
    """\\c 덮어쓰기용 색: &HBBGGRR&"""
    r, g, b = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    return f"&H{b}{g}{r}&".upper()


def _ass_color(hex_color: str) -> str:
    r, g, b = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    return f"&H00{b}{g}{r}".upper()


def write_ass(
    cues: list[tuple[float, float, list[str]]],
    path: Path,
    font_family: str,
    font_size: int,
    margin_v: int,
    color: str,
    size: tuple[int, int],
    title: str = "",
    title_end: float = 0.0,
    highlight: str | None = None,
    title_style: str = "box",
    title_color: str = "#FFE600",
) -> None:
    """title이 있으면 0초부터 title_end까지 화면 위쪽에 계속 보여 준다. title_style:
    - "bold": 크고 굵은 두 줄 제목, 굵은 외곽선, 숫자가 든 낱말(없으면 마지막 낱말)을 title_color로 강조
    - "box": 반투명 상자 안에 한 줄 (예전 모양)
    - "band": 화면 위쪽 전체 폭의 어두운 띠 위에 흰 글씨
    highlight(색)를 주면 단어 강조 자막: 지금 읽는 낱말만 그 색으로 조금 크게 보인다."""
    title = clean_text(title)
    w, h = size
    outline = max(3, font_size // 12)
    top = int(h * SAFE_TOP)  # 위쪽 앱 UI 아래부터
    if title_style == "bold":  # 두 줄까지 크게 (자막 크기의 135%, 두 줄에 안 들어가면 85%까지 줄임)
        title_size = int(font_size * 1.35)
        while title_size > font_size * 0.85 and len(wrap(title, chars_per_line(title_size, w, side_margin=70))) > 2:
            title_size -= 2
    elif title_style == "band":
        title_size = int(font_size * 1.1)
        while title_size > font_size * 0.8 and len(wrap(title, chars_per_line(title_size, w, side_margin=50))) > 1:
            title_size -= 2
    else:  # 제목은 되도록 한 줄로: 길면 글자 크기를 줄인다 (자막 크기의 70%까지)
        title_size = int(font_size * 1.05)
        title_size = max(int(font_size * 0.7), min(title_size, (w - 80) // max(1, len(title.strip()))))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{_ass_color(color)},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1
Style: Title,{font_family},{title_size},&H00FFFFFF,&H000000FF,&H60000000,&H00000000,-1,0,0,0,100,100,0,0,3,18,0,8,60,60,{int(h * 0.08)},1
Style: TitleBold,{font_family},{title_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,-1,0,1,{max(6, title_size // 9)},4,8,70,70,{top + 20},1
Style: TitleBand,{font_family},{title_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,8,50,50,{top + 36},1
Style: Band,{font_family},20,&H30000000,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    when = f"{_ass_time(0)},{_ass_time(title_end)}"
    if title.strip() and title_style == "bold":
        lines = wrap(title.strip(), chars_per_line(title_size, w, side_margin=70))[:2]
        key = title_keyword([t for ln in lines for t in ln.split()])
        k, parts = 0, []
        for ln in lines:
            words = []
            for tok in ln.split():
                tok = tok.replace("{", "(").replace("}", ")")
                words.append(f"{{\\c{_ass_rgb(title_color)}}}{tok}{{\\c&HFFFFFF&}}" if k == key else tok)
                k += 1
            parts.append(" ".join(words))
        events.append(f"Dialogue: 1,{when},TitleBold,,0,0,0,," + r"\N".join(parts))
    elif title.strip() and title_style == "band":
        lines = wrap(title.strip(), chars_per_line(title_size, w, side_margin=50))[:2]
        band_h = top + 36 + int(title_size * 1.25) * len(lines) + 36
        # 띠는 ASS 그림(\p1)으로: 위 끝부터 제목 아래까지 전체 폭 사각형
        events.append(f"Dialogue: 0,{when},Band,,0,0,0,,{{\\pos(0,0)\\p1}}m 0 0 l {w} 0 l {w} {band_h} l 0 {band_h}{{\\p0}}")
        events.append(f"Dialogue: 1,{when},TitleBand,,0,0,0,,{_ass_text(lines)}")
    elif title.strip():
        lines = wrap(title.strip(), chars_per_line(title_size, w, side_margin=40))[:2]
        events.append(f"Dialogue: 1,{when},Title,,0,0,0,,{_ass_text(lines)}")
    for start, end, lines, words in cues:
        if not (highlight and words):
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_ass_text(lines)}")
            continue
        # 낱말마다 이벤트 하나: 그 낱말을 읽기 시작할 때부터 다음 낱말 전까지 그 낱말만 강조
        flat = [(li, wi) for li, line in enumerate(words) for wi in range(len(line))]
        for n, (li, wi) in enumerate(flat):
            a = start if n == 0 else words[li][wi][0]
            b = words[flat[n + 1][0]][flat[n + 1][1]][0] if n + 1 < len(flat) else end
            if b <= a:
                continue
            events.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,"
                          f"{_highlight_text(lines, li, wi, highlight, color)}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _highlight_text(lines: list[str], line_i: int, word_i: int, highlight: str, base: str) -> str:
    out = []
    for li, line in enumerate(lines):
        tokens = [t.replace("{", "(").replace("}", ")") for t in line.split()]
        if li == line_i:
            tokens[word_i] = (f"{{\\c{_ass_rgb(highlight)}\\fscx108\\fscy108}}{tokens[word_i]}"
                              f"{{\\c{_ass_rgb(base)}\\fscx100\\fscy100}}")
        out.append(" ".join(tokens))
    return r"\N".join(out)
