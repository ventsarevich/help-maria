from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from batch_analyze import Detection, make_report


MODEL = "caidas/swin2SR-lightweight-x2-64"


def select_device(requested: str) -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="AI-восстановление качества найденных фрагментов")
    parser.add_argument("run", nargs="?", type=Path, help="папка запуска; по умолчанию последняя")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--limit", type=int, default=0, help="обработать только N лучших; 0 — все")
    args = parser.parse_args()
    run_dir = args.run
    if run_dir is None:
        candidates = sorted(path.parent for path in args.results.glob("*/detections.json"))
        if not candidates:
            raise SystemExit("Готовые результаты не найдены.")
        run_dir = candidates[-1]

    raw = json.loads((run_dir / "detections.json").read_text(encoding="utf-8"))
    settings_path = run_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    device = select_device(args.device)
    print(f"Загрузка AI-модели восстановления {MODEL} ({device})…")
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = Swin2SRForImageSuperResolution.from_pretrained(MODEL).to(device).eval()
    enhanced_dir = run_dir / "enhanced"
    enhanced_dir.mkdir(exist_ok=True)
    selected = raw[:args.limit] if args.limit > 0 else raw

    for index, item in enumerate(selected, 1):
        crop_path = run_dir / item["crop_image"]
        target = enhanced_dir / crop_path.name
        if not crop_path.exists() or target.exists():
            continue
        image = Image.open(crop_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs).reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, 0, -1)
        Image.fromarray((output * 255).round().astype(np.uint8)).save(target, quality=96)
        print(f"[{index}/{len(selected)}] {item['video']} · {item['timecode']}")

    detections = []
    for item in raw:
        item.setdefault("enhanced_image", "")
        item.setdefault("analyzers", "")
        item.setdefault("analyzer_scores", "")
        detection = Detection(**item)
        detection.enhanced_image = ""
        target = enhanced_dir / Path(detection.crop_image).name
        if target.exists():
            detection.enhanced_image = target.relative_to(run_dir).as_posix()
        detections.append(detection)
    make_report(run_dir, detections, settings)
    print(f"Готово: {(run_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
