"""Assemble GitHub Pages assets using only the Python standard library."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "airline_rnn/web"
OUT = ROOT / ".build/site"
MODEL = ROOT / "deploy/browser-model"


def main():
    manifest = json.loads((MODEL / "manifest.json").read_text())
    for filename, expected in manifest["assets"].items():
        if hashlib.sha256((MODEL / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Browser asset checksum mismatch: {filename}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    for path in WEB.iterdir():
        if path.is_file() and path.name != "index.html":
            shutil.copy2(path, OUT / "assets" / path.name)
    shutil.copytree(MODEL, OUT / "model")
    page = (WEB / "index.html").read_text(encoding="utf-8")
    page = page.replace('src="/assets/app.js"', 'type="module" src="./assets/browser-transport.mjs"')
    page = page.replace('="/assets/', '="./assets/')
    page = page.replace("Đang kết nối mô hình", "Đang tải mô hình RNN")
    page = page.replace("Thời gian máy chủ của lượt hiện tại gồm chi phí khởi tạo nếu có.",
                        "Mô hình chạy trên trình duyệt; thời gian phụ thuộc thiết bị của bạn.")
    page = page.replace("Các bước xử lý văn bản và dự đoán cảm xúc của mô hình RNN cho câu input trên.",
                        "Mô hình RNN chạy trực tiếp trong trình duyệt với trọng số đã huấn luyện.")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / ".nojekyll").touch()
    print(f"Static site: {OUT}")


if __name__ == "__main__":
    main()
