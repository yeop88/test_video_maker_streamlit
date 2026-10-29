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
    box = cards.first.locator("textarea")
    box.fill("카드에서 고친 나레이션입니다.")
    box.press("Control+Enter")
    page.wait_for_timeout(1500)
    check("나레이션 글자 수 표시", f"{len('카드에서 고친 나레이션입니다.')}자 / 권장" in cards.first.inner_text())

    page.get_by_role("button", name="➕ 구간 추가").click()
    page.wait_for_timeout(1500)
    check("구간 추가", cards.count() == n + 1, cards.count())
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


def test_music_upload(page, song: Path):
    """사이드바 배경음악: 음악 올리기 → 보관함에 넣기 → 선택되고 미리 듣기가 보임."""
    open_app(page)
    side = page.locator('[data-testid="stSidebar"]')
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


def test_voice_library(page):
    """목소리 보관함: Chatterbox 고르기 → 가상 목소리 만들기 → 바로 골라짐 → 미리 듣기 → Edge로 되돌리기."""
    open_app(page)
    side = page.locator('[data-testid="stSidebar"]')
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
    card.get_by_role("button", name="🎨 만들기").click()
    page.locator('[class*="st-key-cardbox_"]').nth(n).locator("video").wait_for(timeout=300000)
    check("🎨 만들기 → 카드에 AI 장면 미리보기", True, f"{time.time() - t:.0f}초")


def test_ai_music(page):
    """사이드바 ✨ AI로 음악 만들기 → 보관함에 들어가고 바로 골라짐."""
    side = page.locator('[data-testid="stSidebar"]')
    side.get_by_text("✨ AI로 음악 만들기").click()
    side.get_by_label("어떤 음악?").fill("잔잔한 피아노 아침")
    side.get_by_label("어떤 음악?").press("Enter")
    page.wait_for_timeout(800)
    t = time.time()
    side.get_by_role("button", name="음악 만들기").click()
    side.get_by_text("AI 음악을 만들어 보관함에 넣었습니다").wait_for(timeout=300000)
    picked = side.get_by_role("combobox", name="배경음악").input_value()
    check("AI 음악 만들기 → 바로 골라짐", "잔잔한 피아노" in picked, f"{picked}, {time.time() - t:.0f}초")


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
    page.wait_for_timeout(1000)
    page.get_by_text("나레이션을 쓰는 중...").wait_for(state="detached", timeout=300000)
    page.wait_for_timeout(1500)
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
    page.locator('[data-testid="stSidebar"]').get_by_role("combobox", name="목소리", exact=True).click()
    page.get_by_role("option", name="인준 (남성)").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    page.wait_for_function("document.querySelectorAll('[class*=\"st-key-result_\"]').length === 2", timeout=300000)
    page.wait_for_timeout(1500)
    texts = [r.inner_text() for r in results.all()]
    check("다시 만들면 결과 2개 (최신이 앞)", results.count() == 2 and "🆕" in texts[0], [t[:20] for t in texts])
    check("설정 요약으로 차이가 보임", "인준" in texts[0] and "선희" in texts[1], [t.splitlines()[-4:] for t in texts])
    results.nth(1).get_by_role("button", name="🗑").click()
    page.wait_for_timeout(2000)
    check("결과 지우기", results.count() == 1 and "인준" in results.first.inner_text())


def test_series_ui(page, combo: Path):
    """한 영상에서 쇼츠 2편: 쇼츠 개수 2 → 분석 → 탭 2개 → 모든 쇼츠 만들기 → 결과 2개."""
    open_app(page)
    slider = page.locator('[data-testid="stSidebar"]').get_by_role("slider", name="쇼츠 개수 (한 영상에서)")
    slider.focus()
    slider.press("ArrowRight")
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
                test_series_ui(page, combo)
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
