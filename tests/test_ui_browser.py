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


def test_full_flow(page, lake: Path, out: Path):
    open_app(page)
    drag(page, 700, 140, lake)
    page.get_by_role("button", name="1단계 · 영상 분석 + 쇼츠 구성").click()
    t = time.time()
    page.get_by_text("2단계 · 쇼츠 구성과 나레이션 확인").wait_for(timeout=600000)
    page.wait_for_timeout(2000)
    main = page.inner_text('[data-testid="stMain"]')
    check("1단계 → 2단계 표 표시", page.locator('[data-testid="stDataFrame"]').count() == 1, f"{time.time() - t:.0f}초")
    check("2단계 오류 없음", "Error" not in main and "실패" not in main,
          next((ln for ln in main.splitlines() if "Error" in ln), ""))
    title = page.get_by_label("제목 (화면 위쪽)").input_value()
    check("제목", len(title) >= 4, title)

    t = time.time()
    page.get_by_role("button", name="✍️ 나레이션만 다시 쓰기").click()
    page.wait_for_timeout(1000)
    page.get_by_text("나레이션을 쓰는 중...").wait_for(state="detached", timeout=300000)
    page.wait_for_timeout(1500)
    check("나레이션만 다시 쓰기", "Error" not in page.inner_text('[data-testid="stMain"]'), f"{time.time() - t:.0f}초")

    t = time.time()
    page.get_by_role("button", name="3단계 · 나레이션 + 쇼츠 만들기").click()
    page.get_by_role("button", name="⬇️ 쇼츠 다운로드").wait_for(timeout=600000)
    check("3단계 → 결과", page.locator("video").count() >= 2, f"{time.time() - t:.0f}초")
    with page.expect_download() as dl:
        page.get_by_role("button", name="⬇️ 쇼츠 다운로드").click()
    path = out / dl.value.suggested_filename
    dl.value.save_as(path)
    info = media.probe(path)
    check("다운로드한 쇼츠 1080x1920", (info.width, info.height) == (1080, 1920) and info.has_audio,
          f"{info.width}x{info.height} {info.duration:.1f}초")
    page.screenshot(path=str(out / "result.png"), full_page=True)


def test_restore(page):
    """새로고침해도 작업을 이어서 할 수 있는가: 제목을 고치고 새로고침 → 최근 작업에서 열기."""
    box = page.get_by_label("제목 (화면 위쪽)")
    box.fill("새로고침 확인용 제목")
    box.press("Enter")
    page.wait_for_timeout(2000)
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
    check("만든 쇼츠 유지", page.get_by_role("button", name="⬇️ 쇼츠 다운로드").count() == 1)


def main():
    work = Path(tempfile.mkdtemp(prefix="ui_test_"))
    lake = make_video(work, "lake", WALL / "ThemeC/img28.jpg", 14,
                      "안녕하세요. 오늘은 산속 호숫가로 아침 산책을 나왔어요. 물이 거울처럼 잔잔하고 공기가 정말 맑네요.")
    art = make_video(work, "art", WALL / "ThemeB/img24.jpg", 6)

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true", "--server.port", str(PORT)],
        cwd=ROOT, env={**os.environ, "SHORTS_WORK_DIR": str(work / "workdir")},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            if not QUICK:
                test_full_flow(page, lake, work)
                test_restore(page)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)

    print(f"\n{sum(results)}/{len(results)} 통과 (임시 폴더: {work})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
