"""쇼츠와 함께 내보내는 파일: SRT 자막, 썸네일 이미지."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import subtitles


def _srt_time(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(rows: list[dict], path: Path, max_chars: int = 18) -> None:
    """문장마다 자막 하나 (두 줄까지 나눠 씀). 업로드할 때 자막 파일로 쓸 수 있다."""
    blocks = []
    for row in rows:
        text = subtitles.clean_text(row["text"])
        if not text or row["end"] <= row["start"]:
            continue
        lines = subtitles.wrap(text, max_chars)
        blocks.append(f"{len(blocks) + 1}\n{_srt_time(row['start'])} --> {_srt_time(row['end'])}\n" + "\n".join(lines))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _score(img: Image.Image) -> float:
    """선명하고(라플라시안 분산) 얼굴이 있는 장면일수록 높은 점수."""
    import cv2

    gray = np.asarray(img.convert("L").resize((270, 480)))
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(20, 20))
    return sharp * (3.0 if len(faces) else 1.0) + float(gray.std())


def make_thumbnail(frames: list[Image.Image], title: str, font_path: Path, out: Path) -> None:
    """후보 장면(자막을 입히기 전 세로 화면) 중 가장 좋은 장면에 아래쪽 어둡게 + 큰 제목을 얹은 JPG를 만든다."""
    best = max(frames, key=_score)
    size = best.size
    w, h = size

    # 아래쪽 45%를 점점 어둡게
    shade = Image.new("L", (1, h), 0)
    for y in range(h):
        shade.putpixel((0, y), int(220 * max(0.0, (y - h * 0.55) / (h * 0.45)) ** 1.4))
    dark = Image.new("RGB", size, (0, 0, 0))
    img = Image.composite(dark, best, shade.resize(size))

    title = subtitles.clean_text(title)
    if title:
        draw = ImageDraw.Draw(img)
        font_size = 140 if len(title) <= 8 else 116
        font = ImageFont.truetype(str(font_path), font_size)
        lines = subtitles.wrap(title, max(4, int((w - 120) / font_size)))[:3]
        y = int(h * 0.80) - (len(lines) - 1) * int(font_size * 1.2)
        for line in lines:
            tw = draw.textlength(line, font=font)
            draw.text(((w - tw) / 2, y), line, font=font, fill=(255, 230, 0),
                      stroke_width=max(4, font_size // 14), stroke_fill=(0, 0, 0))
            y += int(font_size * 1.2)
    img.filter(ImageFilter.SHARPEN).save(out, "JPEG", quality=90)
