"""화면·자막 설정 미리보기: 실제 쇼츠와 같은 방법(ASS 자막 + ffmpeg)으로 세로 장면 한 장에 제목·자막을 그린다.

쇼츠 앱 UI가 가리는 곳(subtitles.SAFE_*)을 반투명하게 겹쳐 보여 줄 수 있다.
"""
from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config, fonts, media, subtitles

SAMPLE_TITLE = "잠 잘 자는 3가지 방법"
SAMPLE_TEXT = "첫째, 매일 같은 시간에 자고 일어나세요. 몸의 리듬은 규칙성을 좋아합니다."


def _background(size: tuple[int, int]) -> Image.Image:
    """원본 장면이 없을 때 쓰는 배경 (위아래로 어두운 색 그러데이션)."""
    w, h = size
    top, bottom = (38, 52, 92), (120, 70, 110)
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(h):
        k = y / (h - 1)
        draw.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * k) for a, b in zip(top, bottom)))
    return image


def _guide(image: Image.Image, font_file: Path) -> Image.Image:
    """앱 UI가 가리는 곳을 빨갛게 겹친다."""
    w, h = image.size
    top, bottom, right = int(h * subtitles.SAFE_TOP), int(h * subtitles.SAFE_BOTTOM), int(w * subtitles.SAFE_RIGHT)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = (255, 60, 60, 90)
    draw.rectangle([0, 0, w, top], fill=color)
    draw.rectangle([0, h - bottom, w, h], fill=color)
    draw.rectangle([w - right, top, w, h - bottom], fill=color)
    font = ImageFont.truetype(str(font_file), 34)
    draw.text((24, 24), "앱 UI (상태 표시줄·검색)", font=font, fill=(255, 255, 255, 230))
    draw.text((24, h - bottom + 24), "앱 UI (제목·채널·설명)", font=font, fill=(255, 255, 255, 230))
    draw.text((w - right + 8, top + 24), "버튼", font=font, fill=(255, 255, 255, 230))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def render(background: bytes | None, title: str, text: str, font_size: int, margin_percent: int, color: str,
           subtitle_style: str, highlight_color: str, title_style: str, show_title: bool, guide: bool,
           width: int = 360, font: str = "") -> bytes:
    """미리보기 PNG. background는 세로 장면 RGB 바이트(1080x1920, media.layout_frame) 또는 None."""
    w, h = config.OUTPUT_SIZE
    font_file, family = fonts.resolve(font)
    key = hashlib.sha1(repr((background is not None and hashlib.sha1(background).hexdigest(), str(font_file), title, text, font_size,
                             margin_percent, color, subtitle_style, highlight_color, title_style, show_title)).encode())
    work = config.WORK_DIR / "preview" / key.hexdigest()[:12]
    out = work / "frame.png"
    if not out.exists():
        (work / "fonts").mkdir(parents=True, exist_ok=True)
        if not (work / "fonts" / font_file.name).exists():
            shutil.copy(font_file, work / "fonts" / font_file.name)
        bg = Image.frombytes("RGB", (w, h), background[: w * h * 3]) if background else _background((w, h))
        bg.save(work / "bg.png")
        lines = subtitles.wrap(subtitles.clean_text(text), subtitles.chars_per_line(font_size, w))[:2]
        n = sum(len(ln.split()) for ln in lines)
        # 낱말마다 0.5초씩: 1.1초 장면을 뽑으면 단어 강조 자막은 세 번째 낱말이 강조된 모습이 된다
        words, k = [], 0
        for ln in lines:
            words.append([(0.5 * (k + j) + 0.01, 0.5 * (k + j) + 0.5) for j in range(len(ln.split()))])
            k += len(ln.split())
        cues = [(0.0, 0.5 * n + 1, lines, words if subtitle_style == "karaoke" else None)]
        subtitles.write_ass(cues, work / "p.ass", family, font_size, int(h * margin_percent / 100), color, (w, h),
                            title=title if show_title else "", title_end=0.5 * n + 1,
                            highlight=highlight_color if subtitle_style == "karaoke" else None,
                            title_style=title_style, title_color=highlight_color)
        media.run_ffmpeg(["-loop", "1", "-framerate", "10", "-t", "2", "-i", "bg.png",
                          "-vf", "ass=p.ass:fontsdir=fonts,trim=start_frame=11", "-frames:v", "1", "frame.png"], cwd=work)
        # 결과 그림만 남긴다 (글꼴 복사본·배경이 미리보기마다 수 MB씩 쌓였음)
        shutil.rmtree(work / "fonts", ignore_errors=True)
        (work / "bg.png").unlink(missing_ok=True)
    image = Image.open(out).convert("RGB")
    if guide:
        image = _guide(image, font_file)
    image = image.resize((width, int(width * h / w)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()
