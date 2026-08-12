from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


MODEL = "caidas/swin2SR-lightweight-x2-64"
model = processor = None
model_lock = threading.Lock()


def latest_run() -> Path:
    runs = sorted(path.parent for path in Path("results").glob("*/detections.json"))
    if not runs:
        raise SystemExit("Готовые результаты не найдены.")
    return runs[-1].resolve()


class ReportHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/enhance":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            index = int(json.loads(self.rfile.read(length))["index"])
            detections = json.loads((self.directory_path / "detections.json").read_text(encoding="utf-8"))
            item = detections[index]
            crop = (self.directory_path / item["crop_image"]).resolve()
            if self.directory_path not in crop.parents or not crop.is_file():
                raise ValueError("Исходный фрагмент не найден")
            target_dir = self.directory_path / "enhanced"; target_dir.mkdir(exist_ok=True)
            target = target_dir / crop.name
            if not target.exists():
                enhance(crop, target)
            self.respond(200, {"enhanced": f"enhanced/{target.name}"})
        except Exception as error:
            self.respond(500, {"error": str(error)})

    def respond(self, status: int, value: dict):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def enhance(source: Path, target: Path):
    global model, processor
    with model_lock:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if model is None:
            print("Загрузка Swin2SR…")
            processor = AutoImageProcessor.from_pretrained(MODEL)
            model = Swin2SRForImageSuperResolution.from_pretrained(MODEL).to(device).eval()
        image = Image.open(source).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs).reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        Image.fromarray((np.moveaxis(output, 0, -1) * 255).round().astype(np.uint8)).save(target, quality=96)


def main():
    run = latest_run()
    handler = lambda *args, **kwargs: ReportHandler(*args, directory=str(run), **kwargs)
    ReportHandler.directory_path = run
    server = ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    url = "http://127.0.0.1:8765/report.html"
    print(f"Отчёт: {url}\nНе закрывайте это окно во время AI-улучшения.")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
