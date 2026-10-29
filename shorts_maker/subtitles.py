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


def build_cues(rows: list[dict], max_chars: int, max_lines: int = 2) -> list[tuple[float, float, list[str]]]:
    """한 구간의 자막이 길면 여러 화면으로 나누고, 글자 수 비율로 시간을 나눠 준다."""
    cues = []
    for row in rows:
        text = clean_text(row["text"])
        if not text:
            continue
        lines = wrap(text, max_chars)
        groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
        total = sum(len("".join(g)) for g in groups)
        t = row["start"]
        for group in groups:
            share = (row["end"] - row["start"]) * len("".join(group)) / total
            cues.append((t, t + share, group))
            t += share
    return cues


def _ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_text(lines: list[str]) -> str:
    return r"\N".join(ln.replace("{", "(").replace("}", ")") for ln in lines)


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
) -> None:
    """title이 있으면 0초부터 title_end까지 화면 위쪽에 반투명 상자로 계속 보여 준다."""
    title = clean_text(title)
    w, h = size
    outline = max(3, font_size // 12)
    # 제목은 되도록 한 줄로: 길면 글자 크기를 줄인다 (자막 크기의 70%까지)
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

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if title.strip():
        lines = wrap(title.strip(), chars_per_line(title_size, w, side_margin=40))[:2]
        events.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(title_end)},Title,,0,0,0,,{_ass_text(lines)}")
    for start, end, lines in cues:
        events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_ass_text(lines)}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
