"""3가지 조건으로 전체 파이프라인(분석 → 구성 → 나레이션 쇼츠)을 돌리고 수치로 검증한다.

사용법 (프로젝트 루트에서, GPU와 Ollama 실행 중, Edge TTS용 인터넷 필요):
    .venv\\Scripts\\python tests\\verify_pipeline.py [결과 폴더]

테스트 영상은 Windows 기본 배경화면 이미지(호수 풍경, 추상 조형물)와 Edge TTS 음성,
화음 배경음으로 직접 만든다. 장면 내용과 대사를 알고 있으므로 나레이션이 영상 내용에
근거하는지, 구간이 대사 중간에서 끊기지 않는지 확인할 수 있다.
결과 폴더(기본 workdir/verify)에 results.json과 화면 캡처가 저장된다.
"""
import asyncio, difflib, hashlib, json, re, shutil, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts_maker import asr, config, generate, llm, media, music, pipeline, subtitles, tts, voices
from shorts_maker.korean_numbers import number_values
import edge_tts

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "workdir" / "verify"
OUT.mkdir(parents=True, exist_ok=True)


def save_upload(upload_id: str, filename: str, data: bytes) -> Path:
    """앱의 pipeline.save_upload와 같지만 결과 폴더 안(uploads/)에 둔다. workdir에 테스트 업로드 폴더가 쌓이지 않게."""
    folder = OUT / "uploads" / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"source{Path(filename).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path
SR = 44100
WALL = Path("C:/Windows/Web/Wallpaper")
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

TESTS = [
    dict(
        name="T1_korean_lake_vlog",
        size=(1280, 720), bgm=0.08, target=30, style="감성 스토리", engine="edge", layout="smart", subtitles="karaoke",
        # (장면 이미지, 화면 글자, [(목소리, 대사)])
        scenes=[
            (WALL / "ThemeC/img28.jpg", "", [("ko-KR-SunHiNeural", "안녕하세요."),
                                              ("ko-KR-SunHiNeural", "오늘은 산속 호숫가로 아침 산책을 나왔어요."),
                                              ("ko-KR-SunHiNeural", "공기가 정말 맑고, 물이 거울처럼 잔잔하네요.")]),
            (WALL / "ThemeC/img29.jpg", "", [("ko-KR-SunHiNeural", "호수 건너편에는 소나무 숲이 길게 이어져 있어요."),
                                              ("ko-KR-SunHiNeural", "이 길을 따라 한 바퀴 도는 데 한 시간 정도 걸립니다.")]),
            (WALL / "ThemeC/img30.jpg", "", [("ko-KR-SunHiNeural", "마침 해가 산 위로 떠오르기 시작했어요."),
                                              ("ko-KR-SunHiNeural", "이 순간을 보려고 새벽 다섯 시에 일어났답니다.")]),
            (WALL.parent / "Screen/img102.jpg", "", [("ko-KR-SunHiNeural", "낮에는 근처 마을에서 쉬었다가, 저녁에 다시 왔어요."),
                                                     ("ko-KR-SunHiNeural", "노을이 물에 비쳐서 하늘이 두 개가 된 것 같아요.")]),
            (WALL / "ThemeC/img31.jpg", "", [("ko-KR-SunHiNeural", "조용한 곳에서 쉬고 싶다면 꼭 한번 와 보세요."),
                                              ("ko-KR-SunHiNeural", "다음 영상에서 또 만나요.")]),
        ],
        keywords=["호수", "호숫가", "물", "산", "숲", "해", "노을", "풍경", "아침", "새벽", "하늘"],
    ),
    dict(
        name="T2_english_sleep_tips_slides",
        size=(1280, 720), bgm=0.0, target=25, style="정보 전달 (차분한 설명)", engine="chatterbox", layout="crop",
        music=True,  # 8초짜리 배경음악을 올려서 반복·더킹 확인
        scenes=[
            (WALL.parent / "Screen/img101.jpg", "3 Tips for Better Sleep",
             [("en-US-JennyNeural", "Struggling to fall asleep at night?"),
              ("en-US-JennyNeural", "Here are three simple tips that actually work.")]),
            (WALL / "ThemeA/img20.jpg", "TIP 1: Same bedtime every day",
             [("en-US-JennyNeural", "First, go to bed and wake up at the same time every day, even on weekends."),
              ("en-US-JennyNeural", "Your body clock loves routine.")]),
            (WALL / "ThemeA/img21.jpg", "TIP 2: No screens 1 hour before bed",
             [("en-US-JennyNeural", "Second, put your phone away one hour before bed."),
              ("en-US-JennyNeural", "The blue light keeps your brain awake.")]),
            (WALL / "ThemeA/img23.jpg", "TIP 3: Keep your room cool",
             [("en-US-JennyNeural", "Third, keep your bedroom cool, around eighteen degrees."),
              ("en-US-JennyNeural", "A cooler room helps you fall asleep faster.")]),
            (WALL / "ThemeA/img22.jpg", "Sweet dreams!",
             [("en-US-JennyNeural", "Try these tonight and let me know how it goes."),
              ("en-US-JennyNeural", "Sweet dreams!")]),
        ],
        keywords=["잠", "수면", "취침", "침실", "스마트폰", "휴대폰", "화면", "시간"],
        # 팁 3개가 모두 담겼는가 (각 묶음에서 하나 이상)
        topics=[["같은 시간", "일정한 시간", "규칙"], ["휴대폰", "스마트폰", "핸드폰", "폰", "화면", "블루라이트", "파란 빛"],
                ["온도", "시원", "서늘", "18", "십팔"]],
        # 이 말이 들어간 나레이션은 이 장면(번호) 화면 위에 나와야 한다
        sync=[("첫째", 1), ("둘째", 2), ("셋째", 3)],
    ),
    dict(
        name="T3_silent_abstract_art_voice_sample",
        size=(1080, 1080), bgm=0.12, target=20, style="감성 스토리", engine="chatterbox_sample", layout="blur", subtitles="karaoke",
        # 대사 없는 장면: (None, 길이 초)
        scenes=[(WALL / p, "", [(None, 6.0)]) for p in
                ["ThemeB/img24.jpg", "ThemeB/img25.jpg", "ThemeB/img26.jpg", "ThemeB/img27.jpg",
                 "ThemeD/img32.jpg", "ThemeD/img34.jpg", "ThemeD/img35.jpg"]],
        keywords=["색", "모양", "형태", "곡선", "물결", "조형", "리본", "추상", "꽃", "입체", "빛", "흐름", "파란", "주황"],
    ),
]


def norm(text):
    return re.sub(r"[\s\W_]+", "", text.lower())


def coverage(intended, heard):
    """의도한 글자 중 다시 들은 결과에서 순서대로 찾은 비율."""
    a, b = norm(intended), norm(heard)
    matched = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / max(1, len(a))


def tts_edge(text, voice, path):
    asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
    wav = media.load_audio(path, SR, 1)[:, 0]
    loud = np.flatnonzero(np.abs(wav) > 0.01)
    return wav[max(0, loud[0] - 1000): loud[-1] + 1000] if loud.size else wav


def make_video(t, folder):
    """장면별 이미지(천천히 확대) + 대사 + 배경 화음으로 테스트 영상을 만든다.

    돌려주는 값: 영상 경로, 대사 목록 [(시작, 끝, 문장)], 장면 목록 [(시작, 끝, 이미지)]
    """
    shutil.copy("C:/Windows/Fonts/arialbd.ttf", folder / "arialbd.ttf")  # drawtext도 상대 경로로
    w, h = t["size"]
    audio_parts, lines, scenes, clips = [], [], [], []
    cursor = 0.0
    for i, (image, caption, speech) in enumerate(t["scenes"]):
        scene_start = cursor
        part = [np.zeros(int(0.6 * SR), np.float32)]
        cursor += 0.6
        for j, (voice, text) in enumerate(speech):
            if voice is None:
                part.append(np.zeros(int(text * SR), np.float32))
                cursor += text
                continue
            wav = tts_edge(text, voice, folder / f"s{i}_{j}.mp3")
            lines.append((cursor, cursor + len(wav) / SR, text))
            part += [wav, np.zeros(int(0.5 * SR), np.float32)]
            cursor += len(wav) / SR + 0.5
        part.append(np.zeros(int(0.4 * SR), np.float32))
        cursor += 0.4
        audio_parts += part
        dur = sum(len(p) for p in part) / SR
        cursor = scene_start + dur  # 샘플 단위 반올림 오차가 쌓이지 않게
        scenes.append((scene_start, scene_start + dur, image.name))
        clip = folder / f"scene{i}.mp4"
        frames = round(dur * 30)
        vf = (f"scale=1920:-2,zoompan=z='min(zoom+0.0007,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d={frames}:s={w}x{h}:fps=30")
        if caption:  # 글자 속 콜론이 필터 문법과 겹치지 않도록 파일로 넘긴다
            (folder / f"cap{i}.txt").write_text(caption, encoding="utf-8")
            vf += (f",drawtext=fontfile=arialbd.ttf:textfile=cap{i}.txt:fontsize={h // 11}:fontcolor=white"
                   f":box=1:boxcolor=black@0.55:boxborderw=24:x=(w-text_w)/2:y=(h-text_h)/2")
        media.run_ffmpeg(["-loop", "1", "-i", str(image), "-vf", vf, "-frames:v", str(frames),
                          "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", clip.name], cwd=folder)
        clips.append(clip.name)
    speech = np.concatenate(audio_parts)
    tt = np.arange(len(speech)) / SR
    chord = sum(np.sin(2 * np.pi * f * tt) for f in (220.0, 277.18, 329.63)) / 3
    mix = np.clip(speech + t["bgm"] * chord, -1, 1).astype(np.float32)
    sf.write(folder / "audio.wav", mix, SR)
    (folder / "list.txt").write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
    video = folder / f"{t['name']}.mp4"
    media.run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-i", "audio.wav", "-map", "0:v", "-map", "1:a",
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", video.name], cwd=folder)
    return video, lines, scenes


def median_f0(path):
    """목소리 음높이(기본 주파수) 중앙값 Hz. 소리가 큰 부분만."""
    import librosa
    y = media.load_audio(Path(path), 16000, 1)[:, 0]
    f = librosa.yin(y, fmin=60, fmax=400, sr=16000)
    e = librosa.feature.rms(y=y)[0][: len(f)]
    return float(np.median(f[e > 0.3 * e.max()]))


def split_sentences_all(clips):
    return [sent for c in clips for sent in pipeline.split_sentences(c["narration"])]


def ass_cues(path):
    """subs.ass에서 나레이션 자막(Default 스타일) 시간을 읽는다."""
    def sec(s):
        h, m, rest = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    cues = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:") and ",Default," in line:
            parts = line.split(",", 9)
            cues.append((sec(parts[1]), sec(parts[2]), parts[9]))
    return cues


whisper = asr.load_whisper()
chatterbox = None
results = []
sources = {}  # 테스트 이름 → 만든 원본 영상 (T4에서 이어 붙임)
model = config.DEFAULT_OLLAMA_MODEL

for t in TESTS:
    log(f"===== {t['name']} =====")
    folder = OUT / t["name"]
    folder.mkdir(parents=True, exist_ok=True)
    src, true_lines, true_scenes = make_video(t, folder)
    info = media.probe(src)
    r = {"test": t["name"], "source": f"{info.width}x{info.height} {info.duration:.1f}s", "target": t["target"]}
    timings = {}

    # 1단계 (app.py와 같은 순서)
    t1 = time.time()
    data = src.read_bytes()  # 앱처럼 내용 해시를 ID로 써야 다시 만든 테스트 영상이 이전 파일에 가려지지 않는다
    source = save_upload(f"verify_{t['name']}_{hashlib.sha1(data).hexdigest()[:8]}", src.name, data)
    job = pipeline.prepare_job(source, 0.0, info.duration)
    pipeline.transcribe(job, whisper)
    pipeline.capture_frames(job)
    pipeline.describe_frames(job, model)
    title, clips = pipeline.plan(job, model, t["target"], llm.STYLES[t["style"]])
    timings["analyze_plan"] = round(time.time() - t1, 1)

    r["frames"] = len(job.frames)
    r["frames_described"] = sum(1 for f in job.frames if f["caption"])
    r["speech_segments"] = len(job.segments)
    r["title"] = title
    r["clips"] = clips
    narration = " ".join(c["narration"] for c in clips)
    r["clip_count"] = len(clips)
    r["clips_total_sec"] = round(sum(c["end"] - c["start"] for c in clips), 1)
    r["narration_foreign_script"] = bool(llm.FOREIGN_SCRIPT.search(narration + title))
    r["narration_hangul_ratio"] = round(len(re.findall(r"[가-힣]", narration)) / max(1, len(norm(narration))), 2)
    r["keywords_hit"] = [k for k in t["keywords"] if k in narration]
    r["topics_missing"] = [g[0] for g in t.get("topics", []) if not any(k in narration for k in g)]
    # 화면 점검에서 나온 문제: 제목이 "3" 하나, "두 번째·세 번째"만 있고 첫 번째가 빠짐
    orders = llm.ordinals(narration)
    r["ordinals"] = sorted(orders)
    r["ordinals_skipped"] = [n for n in range(1, max(orders, default=0) + 1) if n not in orders]
    # 원본(대사, 화면 글자)에 없는 숫자를 지어냈는가
    truth_text = " ".join(text for _, _, text in true_lines) + " " + " ".join(cap for _, cap, _ in t["scenes"])
    r["invented_numbers"] = sorted(number_values(narration) - number_values(truth_text) - {1})
    # 나레이션과 화면이 맞는가: 그 말을 하는 구간이 해당 장면과 1.5초 이상 겹치는가
    r["out_of_sync"] = []
    for word, idx in t.get("sync", []):
        sa, sb, _ = true_scenes[idx]
        for c in clips:
            if word in c["narration"] and min(c["end"], sb) - max(c["start"], sa) < 1.5:
                r["out_of_sync"].append(f"{word}: clip {c['start']:.1f}-{c['end']:.1f}, scene {sa:.1f}-{sb:.1f}")
    # 구간 경계가 원본 대사 중간에 걸리는가 (앞뒤 0.15초 여유)
    r["cuts_mid_sentence"] = sum(
        1 for c in clips for x in (c["start"], c["end"])
        if any(a + 0.15 < x < b - 0.15 for a, b, _ in true_lines)
    )
    r["clips_overlap"] = sum(1 for a, b in zip(clips, clips[1:]) if b["start"] < a["end"])

    # 3단계
    t2 = time.time()
    if t["engine"] == "edge":
        engine = tts.EdgeTTSEngine("ko-KR-SunHiNeural", 10)
    else:
        if chatterbox is None:
            chatterbox = tts.load_chatterbox()
        reference = None
        if t["engine"] == "chatterbox_sample":  # 목소리 보관함의 가상 목소리 (낮게 바꾼 남성 목소리)
            reference = voices.sample_path(voices.create_virtual("검증용 가상 목소리", "ko-KR-InJoonNeural", -3, -2))
            r["voice_sample_f0"] = round(median_f0(reference))
        engine = tts.ChatterboxEngine(chatterbox, reference, 0.5)
    music_path = None
    if t.get("music"):
        tt = np.arange(int(8 * SR)) / SR  # 0.5초마다 바뀌는 아르페지오 (C-E-G-E)
        freq = np.array([523.25, 659.25, 783.99, 659.25])[(tt * 2).astype(int) % 4]
        song = (0.3 * np.sin(2 * np.pi * freq * tt)).astype(np.float32)
        sf.write(folder / "song.wav", np.stack([song, song], 1), SR)
        music_path = music.track_path(music.add_upload("song.wav", (folder / "song.wav").read_bytes()))
    out, report = pipeline.make_short(job, title, clips, engine, 0.8, 0.2, t["layout"], 72, 20, "#FFFFFF",
                                      subtitle_style=t.get("subtitles", "plain"), highlight_color="#FFE600",
                                      music_path=music_path, music_gain=0.3, music_duck=0.4)
    timings["make_short"] = round(time.time() - t2, 1)

    oi = media.probe(out)
    r["output"] = f"{oi.width}x{oi.height} {oi.duration:.2f}s audio={oi.has_audio}"
    cues = ass_cues(job.dir / "subs.ass")
    r["cues"] = len(cues)
    r["report"] = report
    r["narration_cut"] = sum(1 for x in report if "잘림" in x["상태"])

    # 결과 소리를 다시 인식: 나레이션이 제대로 들리는가 (원본 대사도 섞여 들리므로 포함 비율로 봄)
    segs, out_lang = asr.transcribe(whisper, out)
    heard = " ".join(s.text for s in segs)
    r["heard"] = heard
    r["narration_heard_coverage"] = round(coverage(narration, heard), 3)

    # 나레이션 중 원본 소리 크기 / 나레이션 없을 때 원본 소리 크기 (진폭 비율, 기대값 = duck 0.2)
    # 최종 소리에서 나레이션 트랙을 빼면 원본 소리만 남는다 (클리핑 방지 배율은 최소제곱으로 맞춤)
    mixed = media.load_audio(job.dir / "mixed.wav", SR, 1)[:, 0]
    narr = media.load_audio(job.dir / "narration.wav", SR, 1)[:, 0]
    n = min(len(mixed), len(narr))
    mixed, narr = mixed[:n], narr[:n]
    scale = float(np.dot(mixed, narr) / max(1e-9, np.dot(narr, narr)))
    rest = mixed - scale * narr
    mask = np.zeros(n, bool)
    spans = []  # 단어 강조 자막은 낱말마다 이벤트라 이어진 이벤트를 하나로 합쳐서 본다
    for a, b, _ in sorted(cues):
        if spans and a <= spans[-1][1] + 0.05:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    for a, b in spans:
        mask[int((a + 0.2) * SR): int((b - 0.2) * SR)] = True  # 소리 크기가 바뀌는 앞뒤 0.2초는 제외
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    r["duck_amplitude_ratio"] = round(rms(rest[mask]) / max(1e-9, rms(rest[~mask])), 3) if (~mask).sum() > SR else None

    # 화면 캡처 (첫 자막 중간, 마지막 자막 중간)
    for k, cue in (("first", cues[0]), ("last", cues[-1])):
        mid = (cue[0] + cue[1]) / 2
        media.run_ffmpeg(["-ss", f"{mid:.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=360:640",
                          str(OUT / f"{t['name']}_{k}.png")])

    checks = {
        "output_1080x1920_audio": (oi.width, oi.height) == (1080, 1920) and oi.has_audio,
        "frames_described": r["frames_described"] == r["frames"] and r["frames"] > 0,
        "clips_valid": 1 <= len(clips) <= 6 and r["clips_overlap"] == 0
                       and all(0 <= c["start"] < c["end"] <= job.duration + 0.01 for c in clips),
        "length_near_target": 0.5 * t["target"] <= oi.duration <= 1.6 * t["target"],
        "narration_korean": not r["narration_foreign_script"] and r["narration_hangul_ratio"] >= 0.9,
        "narration_grounded(keywords>=2)": len(r["keywords_hit"]) >= 2,
        "all_topics_covered": not r["topics_missing"],
        "title_meaningful(hangul>=4)": len(re.findall(r"[가-힣]", title)) >= 4,
        "no_skipped_ordinals": not r["ordinals_skipped"],
        "no_invented_numbers": not r["invented_numbers"],
        "narration_matches_scene": not r["out_of_sync"],
        "no_mid_sentence_cut": r["cuts_mid_sentence"] == 0,
        "narration_heard>=0.85": r["narration_heard_coverage"] >= 0.85,
        "subs_within_video": bool(cues) and cues[-1][1] <= oi.duration + 0.05,
        "no_narration_cut": r["narration_cut"] == 0,
    }
    # 함께 내보내는 파일: SRT(문장마다 하나, 시간 순서, 영상 안), 썸네일(1080x1920 JPG)
    srt = out.with_suffix(".srt")
    blocks = [b.split("\n") for b in srt.read_text(encoding="utf-8").strip().split("\n\n")] if srt.exists() else []
    def srt_sec(x):
        h, m, rest = x.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))
    spans_srt = [tuple(srt_sec(x) for x in b[1].split(" --> ")) for b in blocks if len(b) >= 3]
    r["srt_blocks"] = len(spans_srt)
    checks["srt_one_per_sentence"] = len(spans_srt) == len(split_sentences_all(clips))
    checks["srt_in_order_in_video"] = bool(spans_srt) and all(
        a < b <= oi.duration + 0.05 for a, b in spans_srt) and all(x[1] <= y[0] for x, y in zip(spans_srt, spans_srt[1:]))
    thumb = out.with_suffix(".jpg")
    from PIL import Image
    checks["thumbnail_1080x1920"] = thumb.exists() and Image.open(thumb).size == (1080, 1920)
    shutil.copy(thumb, OUT / f"{t['name']}_thumbnail.jpg") if thumb.exists() else None
    if t.get("music"):
        track = media.load_audio(job.dir / "music.wav", SR, 1)[:, 0]
        m = min(len(track), len(mixed))
        r["music_seconds"] = round(len(track) / SR, 2)
        r["music_in_output"] = round(float(np.dot(mixed[:m], track[:m]) / max(1e-9, np.dot(track[:m], track[:m]))), 2)
        last = track[-int(4 * SR): -int(2 * SR)]  # 끝 페이드 직전: 8초 곡이 반복되어 이어지는지
        r["music_tail_rms"] = round(rms(last), 4)
        # 나레이션 바깥은 나레이션에서 0.8초 이상 떨어진 곳만 (문장 사이 짧은 쉼은 계속 줄여 두므로 제외),
        # 처음 1초·끝 2초(페이드)도 제외한다
        near = np.convolve(mask[:len(track)].astype(float), np.ones(int(1.6 * SR)), mode="same") > 0
        away = ~near
        away[: int(1 * SR)] = False
        away[-int(2 * SR):] = False
        inside = mask[:len(track)].copy()
        inside[-int(2 * SR):] = False
        r["music_duck_ratio"] = round(rms(track[inside]) / max(1e-9, rms(track[away])), 3) if away.sum() > SR else None
        checks["music_full_length"] = abs(r["music_seconds"] - oi.duration) < 0.3 and r["music_tail_rms"] > 0.01
        checks["music_in_output"] = 0.7 <= r["music_in_output"] <= 1.3
        if r["music_duck_ratio"] is not None:  # 나레이션이 거의 계속 나오면 비교할 구간이 없어 판정하지 않음
            checks["music_ducked(<0.55)"] = r["music_duck_ratio"] < 0.55
    if t.get("subtitles") == "karaoke":
        # 단어 강조: 낱말마다 이벤트 하나, 시간 순서대로, 강조된 낱말 수 = 나레이션 낱말 수
        events = [c for c in cues if r"\c&H" in c[2]]
        r["karaoke_events"] = len(events)
        r["karaoke_tokens"] = sum(len(subtitles.clean_text(txt).split()) for txt in split_sentences_all(clips))
        checks["karaoke_one_event_per_word"] = len(events) == r["karaoke_tokens"]
        checks["karaoke_in_order"] = all(a[0] <= b[0] for a, b in zip(events, events[1:]))
    if r.get("voice_sample_f0"):  # 제로샷 복제: 나레이션 음높이가 목소리 샘플과 비슷한가
        r["narration_f0"] = round(median_f0(job.dir / "narration.wav"))
        checks["voice_clone_pitch(±15%)"] = abs(r["narration_f0"] - r["voice_sample_f0"]) <= 0.15 * r["voice_sample_f0"]
    if t["name"].startswith("T3"):
        checks["ducking(ratio<0.35)"] = r["duck_amplitude_ratio"] is not None and r["duck_amplitude_ratio"] < 0.35
        checks["no_speech_in_silent_video"] = r["speech_segments"] == 0
    checks = {k: bool(v) for k, v in checks.items()}
    r["checks"] = checks
    r["passed"] = all(checks.values())
    r["timings"] = timings
    r["true_scenes"] = [f"{a:.1f}-{b:.1f} {n}" for a, b, n in true_scenes]
    r["frame_captions"] = [f"{f['t']:.1f}s {f['caption']}" for f in job.frames]
    r["subtitles"] = [f"{a:.1f}-{b:.1f} {txt}" for a, b, txt in cues]
    r["output_path"] = str(out)
    results.append(r)
    sources[t["name"]] = src
    log(f"{t['name']} passed={r['passed']} {checks}")


def test_series():
    """T4: 한 영상에서 쇼츠 2편. T1(호수 브이로그)과 T2(수면 팁) 영상을 이어 붙여 주제가 둘인 75초 영상을 만든다."""
    name = "T4_two_topics_series"
    log(f"===== {name} =====")
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / f"{name}.mp4"
    a, b = sources["T1_korean_lake_vlog"], sources["T2_english_sleep_tips_slides"]
    media.run_ffmpeg(["-i", str(a), "-i", str(b), "-filter_complex",
                      "[0:v]fps=30,format=yuv420p[v0];[1:v]fps=30,format=yuv420p[v1];"
                      "[0:a]aresample=44100[a0];[1:a]aresample=44100[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][au]",
                      "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(src)])
    split_at = media.probe(a).duration
    data = src.read_bytes()
    source = save_upload(f"verify_{name}_{hashlib.sha1(data).hexdigest()[:8]}", src.name, data)
    t1 = time.time()
    job = pipeline.prepare_job(source, 0.0, media.probe(src).duration)
    pipeline.transcribe(job, whisper)
    pipeline.capture_frames(job)
    pipeline.describe_frames(job, model)
    shorts = pipeline.plan_series(job, model, 2, 25, llm.STYLES["정보 전달 (차분한 설명)"])
    r = {"test": name, "source": f"{job.duration:.1f}s (주제 경계 {split_at:.1f}s)", "timings": {"plan": round(time.time() - t1, 1)}}
    r["shorts"] = [{"title": s["title"], "range": [round(x, 1) for x in s["range"]], "focus": s["focus"],
                    "clips": s["clips"]} for s in shorts]
    lake_words = ["호수", "산", "숲", "노을", "산책", "풍경", "물"]
    sleep_words = ["잠", "수면", "취침", "스마트폰", "온도", "침실"]
    texts = [" ".join(c["narration"] for c in s["clips"]) + " " + s["title"] for s in shorts]
    kinds = [("lake" if sum(w in x for w in lake_words) > sum(w in x for w in sleep_words) else "sleep") for x in texts]
    r["kinds"] = kinds
    checks = {
        "two_shorts": len(shorts) == 2,
        "ranges_not_overlapping": len(shorts) == 2 and shorts[0]["range"][1] <= shorts[1]["range"][0] + 0.5,
        "clips_inside_own_range": all(s["range"][0] - 1 <= c["start"] and c["end"] <= s["range"][1] + 3.5
                                      for s in shorts for c in s["clips"]),
        "topics_separated(lake, sleep)": sorted(kinds) == ["lake", "sleep"],
        "titles_distinct": len({s["title"] for s in shorts}) == len(shorts),
    }
    # 컷 전환(호수 → 수면 팁)을 찾고, 나레이션이 길어 구간을 늘려도 컷을 넘지 않는지
    # (예전에는 뒤쪽 주제의 첫 구간이 앞으로 늘어나 호수 화면 위에 다른 주제 나레이션이 나왔음)
    cuts = pipeline.scene_cuts(job)
    r["cuts"] = cuts
    checks["cut_found_at_topic_boundary"] = any(abs(c - split_at) < 0.3 for c in cuts)
    crossing = []
    for s in shorts:
        clips = pipeline.normalize_clips(job, s["clips"])
        spans = pipeline.fit_clips(job, clips, [(c["end"] - c["start"]) * 2 + 2 for c in clips])  # 일부러 긴 나레이션
        crossing += [(a, b) for a, b in spans for c in cuts if a + 0.05 < c < b - 0.05 and not any(
            x["start"] + 0.05 < c < x["end"] - 0.05 for x in clips)]  # 원래 구간이 컷을 품고 있던 경우는 제외
    r["spans_crossing_cut"] = crossing
    checks["extended_clips_dont_cross_cut"] = not crossing
    outs = []
    engine = tts.EdgeTTSEngine("ko-KR-InJoonNeural", 10)
    for s in shorts:
        out, _ = pipeline.make_short(job, s["title"], pipeline.normalize_clips(job, s["clips"]), engine, 0.8, 0.2,
                                     "blur", 72, 20, "#FFFFFF")
        outs.append(out)
    checks["both_rendered_1080x1920"] = len(outs) == 2 and all(
        (media.probe(o).width, media.probe(o).height) == (1080, 1920) for o in outs)
    r["outputs"] = [f"{media.probe(o).duration:.1f}s" for o in outs]
    r["checks"] = {k: bool(v) for k, v in checks.items()}
    r["passed"] = all(r["checks"].values())
    results.append(r)
    log(f"{name} passed={r['passed']} {r['checks']}")


def test_ai_only():
    """T5: 영상 없이 주제만으로 (AI 대본 → AI 장면: 동영상·이미지 섞기)."""
    name = "T5_ai_only_no_upload"
    log(f"===== {name} =====")
    ok, message = generate.available()
    if not ok:
        log(f"{name} 건너뜀: {message}")
        return
    folder = OUT / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    topic = "비 오는 날 창가에서 즐기는 따뜻한 차 한 잔"
    t1 = time.time()
    title, cards = pipeline.plan_ai(model, topic, 20, llm.STYLES["감성 스토리"], "mix")
    r = {"test": name, "topic": title, "timings": {"script": round(time.time() - t1, 1)}}
    narration = " ".join(c["narration"] for c in cards)
    r["scenes"] = [{k: c[k] for k in ("engine", "seconds", "prompt_ko", "prompt", "narration")} for c in cards]
    t2 = time.time()
    engine = tts.EdgeTTSEngine("ko-KR-SunHiNeural", 0)
    out, report = pipeline.make_short(None, title, cards, engine, 0.8, 0.2, "crop", 72, 20, "#FFFFFF",
                                      work_dir=folder / "work", out_dir=folder, subtitle_style="karaoke")
    r["timings"]["make_short"] = round(time.time() - t2, 1)
    oi = media.probe(out)
    r["output"] = f"{oi.width}x{oi.height} {oi.duration:.2f}s"
    segs, _ = asr.transcribe(whisper, out)
    heard = " ".join(seg.text for seg in segs)
    r["narration_heard_coverage"] = round(coverage(narration, heard), 3)
    # 장면이 서로 다른가 (장면 가운데 화면끼리 차이), AI 동영상 장면이 움직이는가
    def frame(t):
        raw = media.run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=54:96,format=gray",
                                "-f", "rawvideo", "-"])
        return np.frombuffer(raw[: 54 * 96], np.uint8).astype(int)
    starts = [float(x["쇼츠 위치"].split("~")[0]) for x in report]
    ends = [float(x["쇼츠 위치"].split("~")[1].rstrip("초")) for x in report]
    mids = [frame((a + b) / 2) for a, b in zip(starts, ends)]
    r["scene_difference"] = [round(float(np.abs(x - y).mean()), 1) for x, y in zip(mids, mids[1:])]
    motion = [round(float(np.abs(frame(a + 0.4) - frame(min(b - 0.3, a + 3.0))).mean()), 1)
              for c, a, b in zip(cards, starts, ends) if c["engine"] == "video"]
    r["video_scene_motion"] = motion
    checks = {
        "output_1080x1920": (oi.width, oi.height) == (1080, 1920) and oi.has_audio,
        "length_near_target": 10 <= oi.duration <= 32,
        "3+_scenes_mixed": len(cards) >= 3 and {"image", "video"} <= {c["engine"] for c in cards},
        "narration_korean": not llm.FOREIGN_SCRIPT.search(narration + title),
        "prompts_english": all(c["prompt"] and not re.search(r"[가-힣]", c["prompt"]) for c in cards),
        "no_invented_numbers": not (number_values(narration + " " + title) - number_values(topic) - {1}),
        "narration_heard>=0.85": r["narration_heard_coverage"] >= 0.85,
        "scenes_look_different": all(d > 8 for d in r["scene_difference"]),
        "ai_video_moves": bool(motion) and all(m > 2 for m in motion),
        "no_narration_cut": not any("잘림" in x["상태"] for x in report),
        "srt_thumbnail": out.with_suffix(".srt").exists() and out.with_suffix(".jpg").exists(),
    }
    r["checks"] = {k: bool(v) for k, v in checks.items()}
    r["passed"] = all(r["checks"].values())
    r["report"] = report
    r["heard"] = heard
    shutil.copy(out.with_suffix(".jpg"), OUT / f"{name}_thumbnail.jpg")
    for i, (a, b) in enumerate(zip(starts, ends)):
        media.run_ffmpeg(["-ss", f"{(a + b) / 2:.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=270:480",
                          str(OUT / f"{name}_scene{i}.png")])
    results.append(r)
    log(f"{name} passed={r['passed']} {r['checks']}")


if {"T1_korean_lake_vlog", "T2_english_sleep_tips_slides"} <= set(sources):
    test_series()
test_ai_only()

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
log("ALL DONE " + " ".join(f"{r['test']}={'PASS' if r['passed'] else 'FAIL'}" for r in results))
