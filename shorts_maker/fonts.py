"""자막·제목 글꼴: 이 컴퓨터에 있는 한글 글꼴 + 올린 글꼴 보관함(workdir/fonts).

설정값(subtitle_font)은 글꼴 ID 문자열이다. "" = 자동(맑은 고딕 등 처음 찾은 것), "sys_<이름>" = 시스템 글꼴,
"up_<해시>" = 올린 글꼴. 다른 컴퓨터에 없는 글꼴이면 자동으로 돌아간다 (프리셋을 나눠 써도 깨지지 않게).
libass에 넘길 글꼴 이름(family)은 글꼴 파일에서 읽는다.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from . import config, subtitles

FONT_TYPES = ["ttf", "otf", "ttc"]
WIN = Path("C:/Windows/Fonts")

# (ID, 화면 이름, 후보 파일들) — 굵은 것을 먼저
SYSTEM_FONTS = [
    ("sys_malgun", "맑은 고딕", [WIN / "malgunbd.ttf"]),
    ("sys_noto_sans", "본고딕 (Noto Sans KR)", [WIN / "NotoSansKR-VF.ttf", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")]),
    ("sys_noto_serif", "본명조 (Noto Serif KR)", [WIN / "NotoSerifKR-VF.ttf", Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")]),
    ("sys_hcr_dotum", "함초롬돋움", [WIN / "HANDotumB.ttf"]),
    ("sys_hcr_batang", "함초롬바탕", [WIN / "HANBatangB.ttf"]),
    ("sys_santteut", "한컴 산뜻돋움", [WIN / "HanSantteutDotum-Bold.ttf"]),
    ("sys_headline", "HY헤드라인", [WIN / "HMKMRHD.TTF"]),
    ("sys_nanum", "나눔고딕", [Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")]),
    ("sys_apple", "Apple SD 산돌고딕 Neo", [Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")]),
]


def library_dir() -> Path:
    return config.WORK_DIR / "fonts"


@lru_cache(maxsize=64)
def family_of(path: str) -> str:
    """글꼴 파일의 family 이름 (libass가 fontsdir에서 이 이름으로 찾음)."""
    return ImageFont.truetype(path, 20).getname()[0]


def has_hangul(path: Path) -> bool:
    """'가'를 그릴 수 있는지 ('가'가 없는 글꼴은 빈 상자(.notdef)를 그려서 영어 'A'와 모양이 같아지지 않음을 이용)."""
    font = ImageFont.truetype(str(path), 40)
    notdef = font.getmask("\uffff").getbbox()
    return font.getmask("가").getbbox() not in (None, notdef) and font.getlength("가") > 0


def list_fonts() -> list[dict]:
    """[{id, name, path, family, source}] 시스템 글꼴(이 컴퓨터에 있는 것) + 올린 글꼴."""
    fonts = []
    for font_id, name, candidates in SYSTEM_FONTS:
        path = next((p for p in candidates if p.exists()), None)
        if path:
            try:
                fonts.append({"id": font_id, "name": name, "path": path, "family": family_of(str(path)), "source": "system"})
            except OSError:
                pass
    folder = library_dir()
    for meta in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            path = folder / info["file"]
            if path.exists():
                fonts.append({"id": meta.stem, "name": info["name"], "path": path, "family": info["family"], "source": "upload"})
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return fonts


BOLD_WEIGHT = 700


def static_bold(path: Path) -> Path:
    """가변 글꼴이면 굵기 700으로 고정한 글꼴 파일을 만들어 돌려준다 (그대로 쓰면 libass가 기본 굵기를 써서
    Noto Sans KR VF는 Thin, Noto Serif KR VF는 ExtraLight로 가늘게 나왔음). 가변 글꼴이 아니면 그대로."""
    if path.suffix.lower() not in (".ttf", ".otf"):
        return path
    from fontTools.ttLib import TTFont

    try:
        font = TTFont(str(path), lazy=True)
        if "fvar" not in font:
            return path
        digest = hashlib.sha1(f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime}".encode()).hexdigest()[:8]
        out = library_dir() / "static" / f"{path.stem}_{digest}_w{BOLD_WEIGHT}{path.suffix.lower()}"
        if out.exists():
            return out
        from fontTools.varLib import instancer

        axes = {a.axisTag: (min(max(BOLD_WEIGHT, a.minValue), a.maxValue) if a.axisTag == "wght" else a.defaultValue)
                for a in font["fvar"].axes}
        static = instancer.instantiateVariableFont(TTFont(str(path)), axes)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        static.save(str(tmp))
        tmp.replace(out)
        return out
    except Exception:  # 고정하지 못하면 원래 글꼴 (가늘어도 그려지기는 함)
        return path


def resolve(font_id: str) -> tuple[Path, str]:
    """글꼴 ID → (파일, family). 없거나 ""면 자동 (subtitles.find_korean_font). 가변 글꼴은 굵게 고정한 파일."""
    path, family = subtitles.find_korean_font()
    if font_id:
        font = next((f for f in list_fonts() if f["id"] == font_id), None)
        if font:
            path, family = font["path"], font["family"]
    return static_bold(path), family


def add_upload(filename: str, data: bytes) -> str:
    """올린 글꼴 파일을 보관함에 넣고 ID를 돌려준다 (같은 파일이면 재사용). 한글이 없으면 ValueError."""
    font_id = "up_" + hashlib.sha1(data).hexdigest()[:10]
    folder = library_dir()
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix.lstrip(".") not in FONT_TYPES:
        raise ValueError("TTF, OTF, TTC 글꼴 파일만 올릴 수 있습니다.")
    path = folder / f"{font_id}{suffix}"
    path.write_bytes(data)
    try:
        family = family_of(str(path))
        if not has_hangul(path):
            raise ValueError("한글 글자가 없는 글꼴입니다. 한글 글꼴을 올려 주세요.")
    except OSError as e:
        path.unlink(missing_ok=True)
        raise ValueError(f"글꼴 파일을 읽지 못했습니다: {e}") from e
    except ValueError:
        path.unlink(missing_ok=True)
        raise
    meta = {"name": Path(filename).stem[:30], "file": path.name, "family": family,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / f"{font_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return font_id


def delete(font_id: str) -> None:
    folder = library_dir()
    meta = folder / f"{font_id}.json"
    if font_id.startswith("up_") and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("file"):
            (folder / info["file"]).unlink(missing_ok=True)
        meta.unlink()
