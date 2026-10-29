// 화면 어디에 영상 파일을 놓아도 업로드되게 한다.
// components.html의 iframe 안에서 실행되며, 같은 출처라서 Streamlit 본 화면(window.parent)에 이벤트를 단다.
// - 파일을 끌고 오면 화면 전체 안내를 띄운다.
// - 업로드 칸 밖에 놓으면 브라우저가 파일을 열어 앱을 떠나는 대신, 그 파일을 업로드 칸에 넘긴다.
(function () {
  const win = window.parent;
  const doc = win.document;
  if (win.__shortsDropInstalled) return; // 다시 실행돼도 한 번만
  win.__shortsDropInstalled = true;

  const overlay = doc.createElement("div");
  overlay.id = "shorts-drop-overlay";
  overlay.textContent = "🎬 여기에 놓으면 영상이 업로드됩니다";
  doc.body.appendChild(overlay);

  const ZONE = '.st-key-video_drop [data-testid="stFileUploaderDropzone"]';
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  let depth = 0; // dragenter/dragleave는 자식 요소마다 발생하므로 깊이로 센다

  const stop = () => {
    depth = 0;
    doc.body.classList.remove("shorts-dragging");
  };

  doc.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    depth += 1;
    if (doc.querySelector(ZONE)) doc.body.classList.add("shorts-dragging");
  }, true);

  doc.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) doc.body.classList.remove("shorts-dragging");
  }, true);

  // 기본 동작(파일 열기)을 막아야 drop 이벤트가 온다
  doc.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });

  doc.addEventListener("drop", (e) => {
    stop();
    if (!hasFiles(e)) return;
    // 업로드 칸(영상·목소리 샘플) 위에 놓았으면 원래대로 처리
    if (e.target.closest && e.target.closest('[data-testid="stFileUploaderDropzone"]')) return;
    e.preventDefault(); // 칸 밖: 브라우저가 파일을 열지 않게
    const zone = doc.querySelector(ZONE);
    const file = e.dataTransfer.files[0];
    if (!zone || !file) return;
    const dt = new DataTransfer(); // 업로드 칸은 파일 하나만 받으므로 첫 파일만 넘긴다
    dt.items.add(file);
    // drop만 보낸다. 업로드 칸(react-dropzone)은 dragenter를 받으면 "끄는 중" 상태를 비동기로 켜는데,
    // dragenter와 drop을 연달아 보내면 drop이 상태를 끈 뒤에 다시 켜져서 칸이 "Drag and drop" 모양으로 남는다.
    zone.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: dt }));
  });

  win.addEventListener("blur", stop); // 창 밖으로 끌고 나간 경우
})();
