"""실제 브라우저(Microsoft Edge)에서 화면을 눌러 가며 확인한다.

사용법 (프로젝트 루트에서, requirements-dev.txt 설치 후):
    .venv\\Scripts\\python tests\\test_ui_browser.py            # 드래그 업로드 + 1~3단계 전체 (GPU, Ollama, 인터넷 필요, 약 2분)
    .venv\\Scripts\\python tests\\test_ui_browser.py --quick    # 드래그 업로드만 (몇십 초)

테스트용 앱 서버를 따로 띄우고(포트 8599), 작업 폴더는 임시 폴더를 쓴다.
AppTest는 파일 업로드를 지원하지 않아서, 업로드 뒤 화면(2단계 표 등)은 이 테스트로만 확인할 수 있다.
실제로 2단계 표가 pyarrow/NumPy 버전 충돌로 깨진 것을 이 테스트로 처음 찾았다.
"""
import asyncio
import base64
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import edge_tts
from playwright.sync_api import sync_playwright

from shorts_maker import media

QUICK = "--quick" in sys.argv
PORT = 8599
URL = f"http://localhost:{PORT}"
ZONE = '.st-key-video_drop [data-testid="stFileUploaderDropzone"]'
WALL = Path("C:/Windows/Web/Wallpaper")
T0 = time.time()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{time.time() - T0:5.0f}s] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def make_video(folder: Path, name: str, image: Path, seconds: float, speech: str = "") -> Path:
    """정지 이미지 + (선택) 한국어 음성으로 짧은 테스트 영상을 만든다."""
    audio = ["-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=44100:cl=stereo"]
    if speech:
        mp3 = folder / f"{name}.mp3"
        asyncio.run(edge_tts.Communicate(speech, "ko-KR-SunHiNeural").save(str(mp3)))
        audio = ["-i", str(mp3)]
    out = folder / f"{name}.mp4"
    media.run_ffmpeg(["-loop", "1", "-t", str(seconds), "-i", str(image), *audio,
                      "-vf", "scale=1280:720,format=yuv420p", "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
                      "-c:a", "aac", "-t", str(seconds), str(out)])
    return out


# 드래그 흉내: 놓을 위치의 요소에 dragenter/dragover → 잠시 뒤 drop (실제 마우스 드래그와 같은 순서)
DRAG_JS = """
async ([x, y, b64, name, doDrop]) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const dt = new DataTransfer();
  dt.items.add(new File([bytes], name, { type: 'video/mp4' }));
  const el = document.elementFromPoint(x, y);
  const fire = (type) => {
    const ev = new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt, clientX: x, clientY: y });
    el.dispatchEvent(ev);
    return ev.defaultPrevented;
  };
  fire('dragenter');
  const dragoverPrevented = fire('dragover');
  if (!doDrop) return { dragoverPrevented };
  await new Promise(r => setTimeout(r, 300));
  fire('dragover');
  return { dragoverPrevented, dropPrevented: fire('drop') };
}
"""


def drag(page, x, y, video: Path, drop=True):
    return page.evaluate(DRAG_JS, [x, y, base64.b64encode(video.read_bytes()).decode(), video.name, drop])


def uploaded_name(page):
    chip = page.query_selector('.st-key-video_drop [data-testid="stFileChipName"]')
    return chip.get_attribute("title") if chip else None


def open_app(page):
    page.goto(URL)
    page.wait_for_selector(ZONE, timeout=60000)
    page.wait_for_function("window.__shortsDropInstalled === true", timeout=30000)
    page.wait_for_timeout(500)


def open_settings(page, label: str):
    """사이드바의 설정 분류 버튼을 눌러 팝업을 열고, 팝업 영역을 돌려준다."""
    page.locator('[data-testid="stSidebar"]').get_by_role("button", name=label).click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout=30000)
    page.wait_for_timeout(800)
    return dialog


def wait_idle(page, timeout: int = 60000) -> None:
    """앱이 다시 그리는 중(오른쪽 위 Running 표시)이 끝날 때까지 기다린다. 그리는 중에 누른 버튼은 무시될 수 있다."""
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout)


def wait_queue_idle(page, timeout: int = 900000) -> None:
    """공용 작업 큐가 다 끝날 때까지 기다린다 (화면 위쪽 「🚦 작업 큐」 제목이 '쉬는 중 · 대기 0')."""
    page.wait_for_function(
        "() => [...document.querySelectorAll('summary')].some(s => s.innerText.includes('🚦 작업 큐 · 쉬는 중 · 대기 0'))",
        timeout=timeout)
    wait_idle(page)
    page.wait_for_timeout(1500)


def open_queue(page) -> None:
    """화면 위쪽 「🚦 작업 큐」를 펼친다 (내 작업이 없으면 접혀 있음)."""
    if not page.get_by_role("button", name="내 완료 비우기").is_visible():
        page.locator("summary").filter(has_text="🚦 작업 큐").first.click()
        page.get_by_role("button", name="내 완료 비우기").wait_for(timeout=30000)
        page.wait_for_timeout(500)


def close_settings(page) -> None:
    """팝업의 「완료」로 닫는다 (바뀐 설정이 화면 전체에 반영됨). 다시 그리는 중이라 눌림이 무시되면 한 번 더 누른다."""
    for _ in range(3):
        wait_idle(page)
        page.get_by_role("dialog").get_by_role("button", name="완료").click()
        try:
            page.get_by_role("dialog").wait_for(state="detached", timeout=15000)
            break
        except Exception:
            continue
    page.wait_for_timeout(1000)


def test_drag_drop(page, lake: Path, art: Path):
    open_app(page)
    box = page.query_selector(ZONE).bounding_box()
    check("업로드 칸이 크게 표시됨", box["height"] >= 190, f"{box['height']:.0f}px")

    r = drag(page, 700, 140, lake, drop=False)  # 제목 근처 = 업로드 칸 밖
    check("끌고 오면 화면 전체 안내", page.is_visible("#shorts-drop-overlay"))
    check("칸 밖에서도 놓을 수 있음", r["dragoverPrevented"])
    page.evaluate("document.dispatchEvent(new DragEvent('drop', {bubbles: true}))")

    r = drag(page, 700, 140, lake)
    check("칸 밖에 놓아도 브라우저가 파일을 열지 않음", r["dropPrevented"])
    page.get_by_role("button", name="1단계 · 영상 분석 + 쇼츠 구성").wait_for(timeout=60000)
    check("칸 밖에 놓으면 업로드", uploaded_name(page) == lake.name, uploaded_name(page))
    check("안내 사라짐", not page.is_visible("#shorts-drop-overlay"))
    check("업로드 칸이 '끄는 중' 상태로 남지 않음", "Drag and drop" not in page.inner_text(ZONE))
    check("올린 뒤 업로드 칸이 작아짐", page.query_selector(ZONE).bounding_box()["height"] < 120)

    vbox = page.query_selector("video").bounding_box()
    drag(page, vbox["x"] + vbox["width"] / 2, vbox["y"] + 20, art)  # 영상 미리보기 위에 다른 영상
    page.wait_for_function(
        f"document.querySelector('.st-key-video_drop [data-testid=\"stFileChipName\"]')?.title === '{art.name}'",
        timeout=60000)
    check("다른 영상을 놓으면 바뀜", uploaded_name(page) == art.name)

    open_app(page)
    b = page.query_selector(ZONE).bounding_box()
    drag(page, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, lake)  # 칸 안에 직접
    page.get_by_role("button", name="1단계 · 영상 분석 + 쇼츠 구성").wait_for(timeout=60000)
    check("칸 안에 놓아도 한 번만 업로드", page.locator('.st-key-video_drop [data-testid="stFileChip"]').count() == 1)
    check("칸 안에 놓은 뒤에도 '끄는 중' 상태로 남지 않음", "Drag and drop" not in page.inner_text(ZONE))


def card_texts(page):
    return [t.input_value() for t in page.locator('[class*="st-key-cardbox_"] textarea').all()]


def test_cards(page):
    """구간 카드: 구간 재생, 나레이션 고치기, 추가, 순서 바꾸기, 삭제."""
    cards = page.locator('[class*="st-key-cardbox_"]')
    check("카드마다 구간 재생", cards.first.locator("video").count() == 1)
    n = cards.count()
    steps = page.locator(".sm-step")
    check("단계 표시줄 4단계, 지금은 카드 편집", steps.count() == 4 and "now" in (steps.nth(1).get_attribute("class") or ""),
          [s.get_attribute("class") for s in steps.all()])
    blocks = page.locator(".sm-tl .seq .blk")
    check("타임라인: 카드 수만큼 순서 블록", blocks.count() == n, f"{blocks.count()} / {n}")
    check("타임라인: 원본 위치 표시", page.locator(".sm-tl .pos").count() == n)

    page.get_by_text("카드를 세로 화면 미리보기로 보기").first.click()
    page.wait_for_timeout(1500)
    wait_idle(page)
    images = cards.first.locator("img")
    check("세로 미리보기: 영상 대신 그림", images.count() == 1 and cards.first.locator("video").count() == 0)
    size = images.first.evaluate("i => [i.naturalWidth, i.naturalHeight]") if images.count() else [0, 0]
    check("세로 미리보기 9:16", size[0] > 0 and abs(size[1] / size[0] - 16 / 9) < 0.02, size)
    page.get_by_text("카드를 세로 화면 미리보기로 보기").first.click()
    page.wait_for_timeout(1500)
    wait_idle(page)
    box = cards.first.locator("textarea")
    box.fill("카드에서 고친 나레이션입니다.")
    box.press("Control+Enter")
    page.wait_for_timeout(1500)
    check("나레이션 글자 수 표시", f"{len('카드에서 고친 나레이션입니다.')}자 / 권장" in cards.first.inner_text())

    page.get_by_role("button", name="➕ 구간 추가").click()
    page.wait_for_timeout(1500)
    check("구간 추가", cards.count() == n + 1, cards.count())
    check("구간 추가 → 타임라인에도 블록", page.locator(".sm-tl .seq .blk").count() == n + 1)
    last = cards.nth(n)
    last.locator("textarea").fill("추가한 구간")
    last.locator("textarea").press("Control+Enter")
    page.wait_for_timeout(1500)

    cards.first.get_by_role("button", name="⬇").click()  # 첫 카드를 뒤로
    page.wait_for_timeout(1500)
    texts = card_texts(page)
    check("순서 바꾸기 (고친 내용 유지)", len(texts) > 1 and texts[1] == "카드에서 고친 나레이션입니다.", texts[:2])

    added = cards.filter(has=page.locator("textarea", has_text="추가한 구간"))
    added.get_by_role("button", name="🗑").click()  # 추가한 카드 삭제
    page.wait_for_timeout(1500)
    check("구간 삭제", cards.count() == n and "추가한 구간" not in card_texts(page), card_texts(page))

    undo = page.get_by_role("button", name=re.compile("↩️ 되돌리기"))
    undo.first.click()  # 삭제 되돌리기
    page.wait_for_timeout(1500)
    check("되돌리기: 지운 카드가 돌아옴", cards.count() == n + 1 and "추가한 구간" in card_texts(page), card_texts(page))
    undo.first.click()  # 순서 바꾸기 되돌리기
    page.wait_for_timeout(1500)
    texts = card_texts(page)
    check("되돌리기: 순서도 돌아옴", texts[0] == "카드에서 고친 나레이션입니다.", texts[:2])
    added = cards.filter(has=page.locator("textarea", has_text="추가한 구간"))
    added.get_by_role("button", name="🗑").click()  # 다음 테스트를 위해 다시 원래대로
    page.wait_for_timeout(1500)
    check("되돌린 뒤에도 고칠 수 있음", cards.count() == n, cards.count())


def test_music_upload(page, song: Path):
    """설정 팝업 「소리·배경음악」: 음악 올리기 → 보관함에 넣기 → 선택되고 미리 듣기가 보임."""
    open_app(page)
    side = open_settings(page, "🔊 소리·배경음악")
    side.get_by_text("음악 올리기").click()
    side.locator('[data-testid="stFileUploaderDropzoneInput"]').last.set_input_files(str(song))
    side.get_by_role("button", name="보관함에 넣기").wait_for(timeout=30000)
    page.wait_for_timeout(1500)
    side.get_by_role("button", name="보관함에 넣기").click()
    page.wait_for_timeout(2500)
    picked = side.get_by_role("combobox", name="배경음악").input_value()  # 고른 값은 입력칸 value에 있음
    check("올린 음악이 배경음악으로 선택됨", song.stem in picked, picked)
    check("배경음악 미리 듣기", side.locator("audio").count() >= 1)
    check("배경음악 크기 조절", side.get_by_text("배경음악 크기", exact=True).count() == 1)
    close_settings(page)
    summary = page.locator('[data-testid="stSidebar"]').inner_text()
    check("팝업을 닫으면 사이드바 요약에 반영", song.stem in summary, song.stem)


def test_voice_library(page):
    """설정 팝업 「나레이션 목소리」: Chatterbox → 가상 목소리 만들기 → 바로 골라짐 → 미리 듣기 → Edge로 되돌리기."""
    open_app(page)
    side = open_settings(page, "🎙️ 나레이션 목소리")
    side.get_by_text("Chatterbox (GPU, 목소리 샘플 사용 가능)").click()
    page.wait_for_timeout(1500)
    side.get_by_text("➕ 새 목소리 만들기").click()
    side.get_by_role("tab", name="가상 목소리").click()
    side.get_by_label("이름").last.fill("테스트 해설자")
    side.get_by_role("button", name="만들기").click()
    side.get_by_text("목소리를 보관함에 넣고 골랐습니다").wait_for(timeout=60000)
    page.wait_for_timeout(1000)
    picked = side.get_by_role("combobox", name="복제할 목소리").input_value()  # 고른 값은 입력칸 value에 있음
    check("가상 목소리 만들기 → 바로 골라짐", "테스트 해설자" in picked, picked)
    side.get_by_role("button", name="▶ 미리 듣기").click()
    side.locator("audio").first.wait_for(timeout=240000)
    check("고른 목소리로 미리 듣기", side.locator("audio").count() >= 1)
    side.get_by_text("Edge TTS (기본 목소리, 인터넷 필요)").click()
    page.wait_for_timeout(1500)
    close_settings(page)


def test_ai_scene_card(page):
    """✨ AI 장면 추가 → 한글 장면 설명·나레이션 → 🎨 만들기 → 카드에 미리보기 영상."""
    cards = page.locator('[class*="st-key-cardbox_"]')
    n = cards.count()
    page.get_by_role("button", name="✨ AI 장면 추가").first.click()
    page.wait_for_timeout(1500)
    check("AI 장면 카드 추가", cards.count() == n + 1)
    card = cards.nth(n)
    card.get_by_label("장면 설명 (한글로 써도 됩니다)").fill("안개 낀 호숫가의 작은 나무 오두막, 새벽")
    card.get_by_label("장면 설명 (한글로 써도 됩니다)").press("Control+Enter")
    page.wait_for_timeout(800)
    card.get_by_label("나레이션 (자막)").fill("호숫가 작은 오두막에서 조용한 아침을 맞습니다.")
    card.get_by_label("나레이션 (자막)").press("Control+Enter")
    page.wait_for_timeout(1200)
    t = time.time()
    card.get_by_role("button", name="➕ 큐에 넣기").click()
    page.locator('[class*="st-key-q_"]').first.wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    check("큐에 넣기 → 공용 작업 큐에 항목",
          any("🎨 AI 장면" in r.inner_text() for r in page.locator('[class*="st-key-q_"]').all()))
    # 큐가 다 만들면 카드에 미리보기 영상이 붙는다 (큐 패널이 2초마다 확인하고 화면을 새로 그림)
    page.locator('[class*="st-key-cardbox_"]').nth(n).locator("video").wait_for(timeout=300000)
    check("🎨 만들기 → 카드에 AI 장면 미리보기", True, f"{time.time() - t:.0f}초")


def test_ai_music(page):
    """설정 팝업 ✨ AI로 음악 만들기 → 보관함에 들어가고 바로 골라짐 (팝업은 열린 채)."""
    side = open_settings(page, "🔊 소리·배경음악")
    side.get_by_text("✨ AI로 음악 만들기").click()
    side.get_by_label("어떤 음악?").fill("잔잔한 피아노 아침")
    side.get_by_label("어떤 음악?").press("Enter")
    page.wait_for_timeout(800)
    t = time.time()
    side.get_by_role("button", name="음악 만들기").click()
    side.get_by_text("AI 음악을 만들어 보관함에 넣었습니다").wait_for(timeout=300000)
    # 완료 문구는 배경음악 칸보다 위에 먼저 그려지므로, 칸이 새 값으로 바뀔 때까지 잠깐 기다린다
    for _ in range(20):
        picked = side.get_by_role("combobox", name="배경음악").input_value()
        if "잔잔한 피아노" in picked:
            break
        page.wait_for_timeout(500)
    check("AI 음악 만들기 → 바로 골라짐", "잔잔한 피아노" in picked, f"{picked}, {time.time() - t:.0f}초")
    close_settings(page)


def test_full_flow(page, lake: Path, out: Path):
    open_app(page)
    drag(page, 700, 140, lake)
    page.get_by_role("button", name="1단계 · 영상 분석 + 쇼츠 구성").click()
    t = time.time()
    page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=600000)
    page.wait_for_timeout(2000)
    main = page.inner_text('[data-testid="stMain"]')
    cards = page.locator('[class*="st-key-cardbox_"]')
    check("1단계 → 2단계 구간 카드 표시", cards.count() >= 1, f"{cards.count()}개, {time.time() - t:.0f}초")
    check("2단계 오류 없음", "Error" not in main and "실패" not in main,
          next((ln for ln in main.splitlines() if "Error" in ln), ""))
    title = page.get_by_label("제목 (화면 위쪽)").input_value()
    check("제목", len(title) >= 4, title)
    test_cards(page)
    test_ai_scene_card(page)
    test_ai_music(page)

    t = time.time()
    page.get_by_role("button", name="✍️ 나레이션만 다시 쓰기").click()
    page.wait_for_timeout(1500)
    check("나레이션 다시 쓰기 → 공용 큐에 들어감", page.get_by_text("✍️ 나레이션 다시 쓰기").count() >= 1)
    wait_queue_idle(page, 300000)
    check("나레이션만 다시 쓰기", "Error" not in page.inner_text('[data-testid="stMain"]'), f"{time.time() - t:.0f}초")

    t = time.time()
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    try:
        page.get_by_role("button", name="⬇️ 다운로드").wait_for(timeout=300000)
    except Exception:
        page.screenshot(path=str(out / "step3_fail.png"), full_page=True)
        errors = [ln for ln in page.inner_text('[data-testid="stMain"]').splitlines() if "Error" in ln or "실패" in ln]
        check("3단계 → 결과", False, errors[:5])
        raise
    check("3단계 → 결과", page.locator("video").count() >= 2, f"{time.time() - t:.0f}초")
    report_text = page.locator('[class*="st-key-result_"]').first.inner_text()
    check("결과에 AI 장면 포함", "AI 장면" in page.inner_text('[data-testid="stMain"]') or "AI" in report_text)
    with page.expect_download() as dl:
        page.get_by_role("button", name="⬇️ 다운로드").click()
    path = out / dl.value.suggested_filename
    dl.value.save_as(path)
    info = media.probe(path)
    check("다운로드한 쇼츠 1080x1920", (info.width, info.height) == (1080, 1920) and info.has_audio,
          f"{info.width}x{info.height} {info.duration:.1f}초")
    card = page.locator('[class*="st-key-result_"]').first
    with page.expect_download() as dl:
        card.get_by_role("button", name="⬇️ 자막 SRT").click()
    srt = out / dl.value.suggested_filename
    dl.value.save_as(srt)
    check("SRT 자막 다운로드", srt.suffix == ".srt" and "-->" in srt.read_text(encoding="utf-8"), srt.name)
    card.get_by_role("tab", name="썸네일").click()
    page.wait_for_timeout(800)
    check("썸네일 보기", card.locator("img").count() >= 1)
    with page.expect_download() as dl:
        card.get_by_role("button", name="⬇️ 썸네일 JPG").click()
    jpg = out / dl.value.suggested_filename
    dl.value.save_as(jpg)
    check("썸네일 다운로드", jpg.suffix == ".jpg" and jpg.stat().st_size > 20_000, f"{jpg.stat().st_size // 1024}KB")
    page.screenshot(path=str(out / "result.png"), full_page=True)
    test_results_history(page)


def test_results_history(page):
    """결과 기록: 설정을 바꿔 한 번 더 만들면 나란히 쌓이고, 지울 수 있다."""
    results = page.locator('[class*="st-key-result_"]')
    check("결과 기록 1개", results.count() == 1, results.count())
    dialog = open_settings(page, "🎙️ 나레이션 목소리")
    dialog.get_by_role("combobox", name="목소리", exact=True).click()
    page.get_by_role("option", name="인준 (남성)").click()
    page.wait_for_timeout(1000)
    close_settings(page)
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    page.wait_for_function("document.querySelectorAll('[class*=\"st-key-result_\"]').length === 2", timeout=300000)
    page.wait_for_timeout(1500)
    texts = [r.inner_text() for r in results.all()]
    check("다시 만들면 결과 2개 (최신이 앞)", results.count() == 2 and "🆕" in texts[0], [t[:20] for t in texts])
    check("설정 요약으로 차이가 보임", "인준" in texts[0] and "선희" in texts[1], [t.splitlines()[-4:] for t in texts])
    results.nth(1).get_by_role("button", name="🗑").click()
    page.wait_for_timeout(2000)
    check("결과 지우기", results.count() == 1 and "인준" in results.first.inner_text())

    # 업로드 문구: 큐에서 LLM이 제목 후보 3개·설명·해시태그를 쓰고 복사할 수 있게 보여 줌
    results.first.get_by_text("📣 업로드 문구").click()
    page.wait_for_timeout(800)
    results.first.get_by_role("button", name="✍️ 업로드 문구 쓰기").click()
    t = time.time()
    page.wait_for_function("document.querySelectorAll('[class*=\"st-key-result_\"] [data-testid=\"stCode\"]').length >= 5",
                           timeout=300000)
    page.wait_for_timeout(1000)
    codes = [c.inner_text().strip() for c in results.first.locator('[data-testid="stCode"]').all()]
    check("업로드 문구: 제목 후보 3개·설명·해시태그", len(codes) == 5 and codes[4].startswith("#Shorts"),
          f"{time.time() - t:.0f}초, {codes}")
    check("업로드 문구 복사 버튼", results.first.locator('[data-testid="stCode"] button').count() >= 1)


def test_series_ui(page, combo: Path):
    """한 영상에서 쇼츠 2편: 쇼츠 개수 2 → 분석 → 탭 2개 → 모든 쇼츠 만들기 → 결과 2개."""
    open_app(page)
    dialog = open_settings(page, "🧠 대본·구성")
    slider = dialog.get_by_role("slider", name="쇼츠 개수 (한 영상에서)")
    slider.focus()
    slider.press("ArrowRight")
    page.wait_for_timeout(1000)
    close_settings(page)
    page.wait_for_timeout(1500)
    drag(page, 700, 140, combo)
    page.get_by_role("button", name="1단계 · 영상 분석 + 쇼츠 구성").click()
    t = time.time()
    page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=600000)
    page.wait_for_timeout(2000)
    tabs = page.get_by_role("tab")
    check("쇼츠 2편 → 탭 2개", tabs.count() >= 2 and "쇼츠 2편" in page.inner_text('[data-testid="stMain"]'),
          f"{[x.inner_text() for x in tabs.all()]}, {time.time() - t:.0f}초")
    tabs.nth(1).click()
    page.wait_for_timeout(1000)
    check("두 번째 탭에도 구간 카드", page.locator('[class*="st-key-cardbox_"]').count() >= 2)
    page.get_by_role("button", name="🎬 모든 쇼츠 만들기").click()
    page.wait_for_function("document.querySelectorAll('[class*=\"st-key-result_\"]').length >= 2", timeout=600000)
    page.wait_for_timeout(1500)
    titles = [r.inner_text().splitlines()[0] for r in page.locator('[class*="st-key-result_"]').all()]
    check("모든 쇼츠 만들기 → 결과 2개 (편마다 제목)", len(titles) == 2 and len(set(titles)) == 2, titles)


def test_ai_mode(page, out: Path):
    """영상 없이 AI로: 주제 → 대본·장면 카드 → 쇼츠 만들기 → 1080x1920 결과."""
    open_app(page)
    page.get_by_text("✨ 영상 없이 AI로 만들기").click()
    page.wait_for_timeout(1500)
    check("AI 모드에서는 영상 업로드 칸이 없음", page.locator(ZONE).count() == 0)
    models_text = page.locator('[data-testid="stMain"]').inner_text()
    check("AI 모드에 사용하는 AI 모델 표시", "🤖 사용하는 AI 모델" in models_text)
    dialog = open_settings(page, "💻 컴퓨터·AI 모델")
    dialog_text = dialog.inner_text()
    check("컴퓨터 팝업: 사양 (GPU·RAM)과 장치별 추천 모델",
          "RAM" in dialog_text and "추천 모델 (지금 장치 기준)" in dialog_text and "지금 고른 모델" in dialog_text)
    check("컴퓨터 팝업: 계산 장치 설정", dialog.get_by_text("GPU + CPU 함께 (GPU 메모리 아끼기)").count() == 1)
    close_settings(page)
    page.get_by_label("어떤 쇼츠를 만들까요?").fill("비 오는 날 창가에서 즐기는 따뜻한 차 한 잔")
    page.get_by_label("어떤 쇼츠를 만들까요?").press("Control+Enter")
    page.wait_for_timeout(1000)
    t = time.time()
    page.get_by_role("button", name="1단계 · AI 대본 + 장면 구성").click()
    page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=300000)
    page.wait_for_timeout(1500)
    cards = page.locator('[class*="st-key-cardbox_"]')
    prompts = [c.get_by_label("장면 설명 (한글로 써도 됩니다)").input_value() for c in cards.all()]
    check("AI 대본 → AI 장면 카드", cards.count() >= 3 and all(prompts), f"{cards.count()}장면, {time.time() - t:.0f}초")
    check("AI 모드에는 원본 구간 추가 버튼이 없음", page.get_by_role("button", name="➕ 구간 추가").count() == 0)

    # 생성 큐: 모두 넣기 → 대기 항목 순서 바꾸기·빼기 → 남은 장면이 다 만들어짐
    open_queue(page)
    if page.get_by_role("button", name="내 완료 비우기").is_enabled():
        page.get_by_role("button", name="내 완료 비우기").click()
    page.get_by_role("button", name="⏸ 일시정지").click()  # 순서를 바꾸는 동안 멈춰 둠
    page.wait_for_timeout(1000)
    page.get_by_role("button", name=re.compile(r"🎬 AI 장면 \d+개 모두 큐에 넣기")).click()
    page.get_by_text("일시정지됨").wait_for(timeout=120000)
    page.wait_for_timeout(1500)
    rows = page.locator('[class*="st-key-q_"]')
    waiting = [r for r in rows.all() if "번째" in r.inner_text()]
    check("모두 큐에 넣기 → 장면 수만큼 대기", len(waiting) == cards.count(), f"대기 {len(waiting)} / 장면 {cards.count()}")
    row_id = lambda r: next(c for c in r.get_attribute("class").split() if c.startswith("st-key-q_"))  # noqa: E731
    first_id = row_id(waiting[0])
    waiting_rows = lambda: [r for r in page.locator('[class*="st-key-q_"]').all() if "번째" in r.inner_text()]  # noqa: E731

    def click_until(button, done) -> bool:
        """큐 패널은 2초마다 다시 그려져서, 그리는 중에 누른 버튼은 무시될 수 있다. 조용할 때 누르고 바뀔 때까지 확인."""
        for _ in range(3):
            wait_idle(page)
            button().click()
            for _ in range(10):
                page.wait_for_timeout(500)
                if done():
                    return True
        return False

    moved = click_until(lambda: waiting_rows()[0].get_by_role("button", name="⬇"),
                        lambda: row_id(waiting_rows()[1]) == first_id)
    check("큐 순서 바꾸기 (⬇)", moved, f"{first_id} → 2번째")
    removed = click_until(lambda: waiting_rows()[-1].get_by_role("button", name="🗑"),
                          lambda: len(waiting_rows()) == cards.count() - 1)
    check("큐에서 빼기 (🗑)", removed, f"대기 {len(waiting_rows())}")
    t = time.time()
    page.get_by_role("button", name="▶ 이어서").click()
    page.wait_for_function(
        "() => document.querySelectorAll('[class*=\"st-key-cardbox_\"] video').length >= "
        f"{cards.count() - 1}", timeout=900000)
    check("큐가 이어서 만들어 카드에 미리보기", True, f"{time.time() - t:.0f}초")
    page.screenshot(path=str(out / "ai_queue.png"), full_page=True)
    t = time.time()
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    page.get_by_role("button", name="⬇️ 다운로드").wait_for(timeout=900000)
    with page.expect_download() as dl:
        page.get_by_role("button", name="⬇️ 다운로드").first.click()
    path = out / f"ai_{dl.value.suggested_filename}"
    dl.value.save_as(path)
    info = media.probe(path)
    check("AI 모드 쇼츠 1080x1920", (info.width, info.height) == (1080, 1920) and info.duration > 10,
          f"{info.duration:.1f}초, {time.time() - t:.0f}초 걸림")
    page.screenshot(path=str(out / "ai_mode.png"), full_page=True)
    page.get_by_text("📼 영상으로 만들기").click()
    page.wait_for_timeout(1500)
    check("영상 모드로 돌아가면 업로드 칸", page.locator(ZONE).count() == 1)


def test_shared_queue(browser, page):
    """두 사람(A=page, B=새 브라우저)이 같은 큐를 쓴다: B에게 A의 작업이 「다른 사용자」로 보이고 버튼이 없음,
    B는 자기 작업만 다룰 수 있음, 일시정지를 풀면 넣은 순서대로 둘 다 만들어짐."""
    results_before = page.locator('[class*="st-key-result_"]').count()
    open_queue(page)
    page.get_by_role("button", name="⏸ 일시정지").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    wait_idle(page)
    other = browser.new_page(viewport={"width": 1400, "height": 1000})
    other.goto(URL)
    other.wait_for_selector(ZONE, timeout=60000)
    check("사람마다 다른 ID (주소의 u)", "u=" in other.url and other.url != page.url, f"{page.url} / {other.url}")
    other.get_by_text("📂 최근 작업").click()
    other.get_by_role("button", name="열기").first.click()
    other.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=60000)
    wait_idle(other)
    other.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    wait_idle(other)
    other.wait_for_timeout(2000)
    rows = other.locator('[class*="st-key-q_"]')
    texts = [r.inner_text() for r in rows.all()]
    a_row = next((r for r, t in zip(rows.all(), texts) if "다른 사용자" in t), None)
    b_row = next((r for r, t in zip(rows.all(), texts) if "내 작업" in t), None)
    check("B 화면: A의 작업은 「다른 사용자」, 내 작업은 따로", a_row is not None and b_row is not None,
          [t.replace(chr(10), " | ")[:80] for t in texts])
    check("B는 A의 작업을 빼거나 옮길 수 없음", a_row is not None and a_row.get_by_role("button").count() == 0)
    check("B는 A의 작업 앞으로 끼어들 수 없음 (⬆ 비활성)",
          b_row is not None and b_row.get_by_role("button", name="⬆").is_disabled())
    check("B 작업 대기 순서 2번째", b_row is not None and "2번째" in b_row.inner_text())
    page.get_by_role("button", name="▶ 이어서").click()
    wait_queue_idle(page, 600000)
    page.wait_for_function(f"document.querySelectorAll('[class*=\"st-key-result_\"]').length >= {results_before + 2}",
                           timeout=120000)
    check("A·B 작업이 차례로 끝나 결과 2개 추가", True, f"{results_before} → {results_before + 2}")
    other.close()


def test_resource_monitor(page):
    """「💻 컴퓨터·AI 모델」 팝업의 실시간 사용량: CPU·RAM·GPU 값, 1초마다 새로 고침, 프로그램별 표. 사이드바 한 줄 요약."""
    side = page.locator('[data-testid="stSidebar"]').inner_text()
    check("사이드바 실시간 사용량 요약", "CPU" in side and "RAM" in side and "작업 큐:" in side)
    dialog = open_settings(page, "💻 컴퓨터·AI 모델")
    page.wait_for_timeout(2000)
    text = dialog.inner_text()
    check("실시간 사용량: CPU·RAM·GPU 값", all(k in text for k in ("CPU 사용", "RAM 사용", "GPU")), text[:200])
    check("프로그램별 CPU·RAM 표 (앱 서버)", "앱 서버" in text)
    stamp = lambda: next((ln for ln in dialog.inner_text().splitlines() if ln.startswith("측정 ")), "")  # noqa: E731
    first = stamp()
    page.wait_for_timeout(3500)
    check("사용량이 1초마다 새로 고쳐짐", first and stamp() != first, f"{first[:12]} → {stamp()[:12]}")
    close_settings(page)


def test_restore(page):
    """새로고침해도 작업을 이어서 할 수 있는가: 제목을 고치고 새로고침 → 최근 작업에서 열기."""
    box = page.get_by_label("제목 (화면 위쪽)")
    box.fill("새로고침 확인용 제목")
    box.press("Enter")
    page.wait_for_timeout(2000)
    before = card_texts(page)
    page.reload()
    page.wait_for_selector(ZONE, timeout=60000)
    check("새로고침하면 업로드 전 화면", page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").count() == 0)
    page.get_by_text("📂 최근 작업").click()
    page.get_by_role("button", name="열기").first.click()
    page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    check("최근 작업 열기 → 분석 다시 안 하고 2단계로", page.get_by_role("button", name="1단계 · 다시 분석 + 쇼츠 구성").count() == 1)
    check("고친 제목 유지", page.get_by_label("제목 (화면 위쪽)").input_value() == "새로고침 확인용 제목",
          page.get_by_label("제목 (화면 위쪽)").input_value())
    check("카드 순서·나레이션 유지", card_texts(page) == before, card_texts(page))
    check("만든 쇼츠 유지", page.get_by_role("button", name="⬇️ 다운로드").count() == 1)


def test_device_split(page):
    """설정 팝업 「컴퓨터·AI 모델」: 작업별로 나누기 → 작업 6개 장치 선택, 장치를 바꾸면 추천이 바뀜, 추천대로 맞추기."""
    open_app(page)
    side = page.locator('[data-testid="stSidebar"]')
    dialog = open_settings(page, "💻 컴퓨터·AI 모델")
    before = dialog.inner_text()
    dialog.get_by_role("radio", name="작업별로 나누기 (예: 영상은 GPU, 나머지는 CPU)").check(force=True)
    page.wait_for_timeout(2500)
    check("나누기를 골라도 팝업이 열려 있음", page.get_by_role("dialog").count() == 1)
    labels = ["장면 이해·대본 (Ollama LLM)", "대사 인식 (Whisper)", "나레이션 음성 (Chatterbox)", "AI 이미지", "AI 동영상", "AI 배경음악"]
    shown = [label for label in labels if dialog.get_by_text(label, exact=True).count() >= 1]
    check("작업별로 나누기 → 작업 6개 장치 선택", len(shown) == 6, shown)
    after = dialog.inner_text()
    check("장치를 바꾸면 추천 모델이 바뀜 (이미지: GPU → CPU)",
          "FLUX.2 [klein] 9B 4bit" in before and "SDXL-Turbo (fp32)" in after and "Edge TTS" in after,
          "GPU일 때 FLUX.2 9B, CPU일 때 SDXL-Turbo·Edge TTS")
    dialog.get_by_role("button", name="✨ 추천 모델로 맞추기").click()
    page.wait_for_timeout(2500)
    close_settings(page)
    summary = side.inner_text()
    check("기본 나누기: 동영상만 GPU (사이드바 요약)", "AI 동영상만 GPU" in summary, summary[-200:])
    check("추천 모델로 맞추기 → 이미지 SDXL-Turbo, 음성 Edge (사이드바 요약)",
          "SDXL-Turbo" in summary and "Edge ·" in summary)
    dialog = open_settings(page, "💻 컴퓨터·AI 모델")  # 다음 테스트를 위해 되돌림
    dialog.get_by_role("radio", name="GPU만 (가장 빠름)").check(force=True)
    page.wait_for_timeout(1500)
    close_settings(page)

def test_delete_recent(page, work: Path):
    """📂 최근 작업: 🗑 → 확인 팝업 → 지우기 → 목록과 폴더에서 사라짐. 취소하면 그대로."""
    open_app(page)
    main = page.locator('[data-testid="stMain"]')
    main.get_by_text("📂 최근 작업").click()
    page.wait_for_timeout(1000)
    delete_buttons = page.locator('[class*="st-key-delrecent_"]')
    before = delete_buttons.count()
    first = delete_buttons.first
    folder = next(c for c in first.get_attribute("class").split() if c.startswith("st-key-delrecent_"))[len("st-key-delrecent_"):]
    first.get_by_role("button").click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout=30000)
    check("지우기 전에 확인 팝업", "작업을 지울까요?" in dialog.inner_text())
    dialog.get_by_role("button", name="취소").click()
    page.wait_for_timeout(1500)

    # 저장 공간: 항목별 크기, 미리보기 캐시 지우기
    if page.get_by_role("button", name="🧹 저장 공간 보기").count() == 0:
        main.get_by_text("📂 최근 작업").click()
        page.wait_for_timeout(800)
    page.get_by_role("button", name="🧹 저장 공간 보기").click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout=60000)
    dialog.get_by_text("목소리 보관함").wait_for(timeout=60000)  # 크기를 다 잰 뒤에 항목이 나옴
    page.wait_for_timeout(500)
    found = {k: dialog.get_by_text(k).count() for k in ("📂 작업", "미리보기 캐시", "AI 장면·음악 캐시", "목소리 보관함", "드라이브 남은 공간")}
    check("저장 공간: 작업·캐시·보관함 크기", all(found.values()), found)
    preview_dir = work / "workdir" / "preview"
    had = any(preview_dir.rglob("*.png"))
    clear_preview = dialog.locator('[class*="st-key-clear_preview"]').get_by_role("button")
    if had and clear_preview.is_enabled():
        clear_preview.click()
        page.wait_for_timeout(2500)
        check("미리보기 캐시 지우기", not any(preview_dir.rglob("*.png")) and "지웠습니다" in dialog.inner_text())
    else:
        check("미리보기 캐시 지우기", False, f"그림 있음={had}, 버튼 사용 가능={clear_preview.is_enabled()}")
    dialog.get_by_role("button", name="완료").click()
    page.wait_for_timeout(1500)
    if page.locator('[class*="st-key-delrecent_"]').count() == 0:
        main.get_by_text("📂 최근 작업").click()
        page.wait_for_timeout(800)
    page.locator(f'[class*="st-key-delrecent_{folder}"]').get_by_role("button").click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout=30000)
    dialog.get_by_role("button", name="취소").click()
    page.wait_for_timeout(2000)
    check("취소하면 그대로", (work / "workdir" / folder).exists())
    if page.locator('[class*="st-key-delrecent_"]').count() == 0:  # 팝업을 닫으면서 펼침이 접혔으면 다시 펼침
        main.get_by_text("📂 최근 작업").click()
    page.wait_for_timeout(800)
    page.locator(f'[class*="st-key-delrecent_{folder}"]').get_by_role("button").click()
    page.get_by_role("dialog").get_by_role("button", name="🗑 지우기").click()
    page.wait_for_timeout(3000)
    if page.locator('[class*="st-key-delrecent_"]').count() == 0:
        main.get_by_text("📂 최근 작업").click()
        page.wait_for_timeout(800)
    after = page.locator('[class*="st-key-delrecent_"]').count()
    check("지우면 목록과 폴더에서 사라짐", after == before - 1 and not (work / "workdir" / folder).exists(),
          f"{before} → {after}, {folder}")


def main():
    work = Path(tempfile.mkdtemp(prefix="ui_test_"))
    lake = make_video(work, "lake", WALL / "ThemeC/img28.jpg", 14,
                      "안녕하세요. 오늘은 산속 호숫가로 아침 산책을 나왔어요. 물이 거울처럼 잔잔하고 공기가 정말 맑네요.")
    art = make_video(work, "art", WALL / "ThemeB/img24.jpg", 6)
    combo = work / "combo.mp4"  # 주제가 둘인 영상: 호수 풍경(말소리 있음) + 추상 조형물(음악 없음)
    art_long = make_video(work, "art_long", WALL / "ThemeB/img25.jpg", 9)
    media.run_ffmpeg(["-i", str(lake), "-i", str(art_long), "-filter_complex",
                      "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
                      "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(combo)])
    song = work / "test_song.mp3"
    media.run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-ac", "2", str(song)])

    log = open(work / "server.log", "w", encoding="utf-8")  # 서버가 죽으면 원인을 볼 수 있게
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true", "--server.port", str(PORT)],
        cwd=ROOT, env={**os.environ, "SHORTS_WORK_DIR": str(work / "workdir"), "PYTHONFAULTHANDLER": "1"},
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                    break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, accept_downloads=True)
            test_drag_drop(page, lake, art)
            test_music_upload(page, song)
            if not QUICK:
                test_voice_library(page)
                test_full_flow(page, lake, work)
                test_restore(page)
                test_shared_queue(browser, page)
                test_resource_monitor(page)
                test_series_ui(page, combo)
                test_ai_mode(page, work)
                test_device_split(page)
                test_delete_recent(page, work)
            browser.close()
    finally:
        died = server.poll() is not None
        server.terminate()
        server.wait(timeout=30)
        log.close()
        if died:
            print("테스트 서버가 중간에 종료됨. 로그 끝부분:")
            print((work / "server.log").read_text(encoding="utf-8", errors="ignore")[-3000:])

    print(f"\n{sum(results)}/{len(results)} 통과 (임시 폴더: {work})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
