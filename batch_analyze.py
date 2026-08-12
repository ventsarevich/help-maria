from __future__ import annotations

import argparse
import csv
import html
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
AVAILABLE_PROMPTS = ["person", "human", "man", "backpack", "jacket", "clothes", "human body"]

def videos_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)

def load_configurations(path: Path) -> dict[str, list[str]]:
    if not path.exists(): return {"full_search": AVAILABLE_PROMPTS}
    return json.loads(path.read_text(encoding="utf-8"))

def choose_option(title: str, options: list[tuple[str, str]], default: int = 1) -> str:
    print(f"\n{title}:\n")
    for index, (_, description) in enumerate(options, 1): print(f"  {index}. {description}")
    while True:
        raw = input(f"\nНомер [{default}]: ").strip()
        if not raw: return options[default - 1][0]
        try: return options[int(raw) - 1][0]
        except (ValueError, IndexError): print("Введите номер из списка.")

def choose_prompts(configurations: dict[str, list[str]]) -> list[str]:
    entries = list(configurations.items())
    print("\nКонфигурации поиска:\n")
    for index, (name, prompts) in enumerate(entries, 1): print(f"  {index}. {name}: {', '.join(prompts)}")
    while True:
        try: return entries[int(input("\nНомер конфигурации: ")) - 1][1]
        except (ValueError, IndexError): print("Введите номер из списка.")

def tile_images(frame, size: int, overlap: float):
    height, width = frame.shape[:2]; tile = min(size, width, height)
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), stride)); ys = list(range(0, max(1, height - tile + 1), stride))
    if not xs or xs[-1] != width - tile: xs.append(max(0, width - tile))
    if not ys or ys[-1] != height - tile: ys.append(max(0, height - tile))
    for y in dict.fromkeys(ys):
        for x in dict.fromkeys(xs): yield frame[y:y + tile, x:x + tile], x, y

def predict_boxes(model, frame, device, imgsz, conf, mode, tile_size, overlap, quantize="fp32"):
    sources = [(frame, 0, 0)] if mode == "full" else list(tile_images(frame, tile_size, overlap))
    results = model.predict([item[0] for item in sources], device=device, imgsz=imgsz, conf=conf, quantize=quantize, verbose=False)
    boxes, scores, classes = [], [], []
    for result, (_, ox, oy) in zip(results, sources):
        if result.boxes is None: continue
        for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = xyxy; boxes.append([int(x1 + ox), int(y1 + oy), int(x2 - x1), int(y2 - y1)]); scores.append(float(score)); classes.append(int(class_id))
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45) if boxes else []
    return [(boxes[i], scores[i], classes[i]) for i in np.array(keep).reshape(-1).tolist()] if len(keep) else []


BATCH_PROFILES = {
    "quick": {"sample_seconds": 5.0, "mode": "full", "imgsz": 640, "tile_size": 1280, "overlap": 0.10},
    "balanced": {"sample_seconds": 3.0, "mode": "tiled", "imgsz": 640, "tile_size": 1280, "overlap": 0.10},
    "deep": {"sample_seconds": 1.5, "mode": "tiled", "imgsz": 960, "tile_size": 960, "overlap": 0.20},
}


@dataclass
class Detection:
    video: str
    video_path: str
    timestamp: float
    timecode: str
    label: str
    confidence: float
    motion: float
    stable: bool
    score: float
    frame_image: str
    crop_image: str
    box: list[int]
    enhanced_image: str = ""
    analyzers: str = ""
    analyzer_scores: str = ""


def timecode(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


class YoloEngine:
    def __init__(self, model_name: str, prompts: list[str], device: str):
        from ultralytics import YOLOE

        self.model = YOLOE(model_name)
        self.model.set_classes(prompts)
        self.prompts = prompts
        self.device = device

    def predict(self, frame, settings):
        raw = predict_boxes(
            self.model, frame, self.device, settings["imgsz"], settings["conf"],
            settings["mode"], settings["tile_size"], settings["overlap"], settings["quantize"],
        )
        return [(box, score, self.prompts[class_id] if class_id < len(self.prompts) else str(class_id))
                for box, score, class_id in raw]


class GroundingDinoEngine:
    def __init__(self, model_name: str, prompts: list[str], device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.prompts = prompts
        self.text = ". ".join(prompts) + "."
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()

    def _one(self, image, offset_x: int, offset_y: int, conf: float):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=self.text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=conf, text_threshold=0.20,
            target_sizes=[(image.shape[0], image.shape[1])],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        found = []
        for box, score, label in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), labels):
            x1, y1, x2, y2 = box
            found.append(([int(x1 + offset_x), int(y1 + offset_y), int(x2 - x1), int(y2 - y1)],
                          float(score), str(label)))
        return found

    def predict(self, frame, settings):
        sources = [(frame, 0, 0)] if settings["mode"] == "full" else list(
            tile_images(frame, settings["tile_size"], settings["overlap"])
        )
        detections = []
        for image, x, y in sources:
            detections.extend(self._one(image, x, y, settings["conf"]))
        if not detections:
            return []
        boxes = [item[0] for item in detections]
        scores = [item[1] for item in detections]
        keep = cv2.dnn.NMSBoxes(boxes, scores, settings["conf"], 0.45)
        return [detections[index] for index in np.array(keep).reshape(-1).tolist()] if len(keep) else []


class SiglipEngine:
    def __init__(self, model_name: str, prompts: list[str], device: str):
        try:
            import sentencepiece  # noqa: F401
            import google.protobuf  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "Для тройного анализа нужны библиотеки sentencepiece и protobuf. "
                "Перезапустите static_analyze.command — он установит недостающие зависимости."
            ) from error
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.prompts = prompts
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def predict(self, frame, settings):
        sources = list(tile_images(frame, settings["tile_size"], settings["overlap"]))
        found = []
        texts = [f"a drone photograph containing {prompt}" for prompt in self.prompts]
        for image, x, y in sources:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(text=texts, images=[rgb], padding="max_length", return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            probabilities = self.torch.sigmoid(outputs.logits_per_image[0]).float().cpu().numpy()
            best = int(np.argmax(probabilities)); score = float(probabilities[best])
            if score >= settings["siglip_conf"]:
                found.append(([x, y, image.shape[1], image.shape[0]], score, self.prompts[best]))
        return found


def intersection_over_union(first, second) -> float:
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    x1, y1, x2, y2 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def merge_predictions(predictions: list[tuple[str, list]], weights: dict[str, float]) -> list[tuple]:
    merged = []
    for analyzer, detections in predictions:
        for box, confidence, label in detections:
            matched = None
            for item in merged:
                # SigLIP returns a whole tile, so containment also counts as agreement.
                bx, by, bw, bh = box; ix, iy, iw, ih = item["box"]
                center_inside = bx <= ix + iw / 2 <= bx + bw and by <= iy + ih / 2 <= by + bh
                reverse_inside = ix <= bx + bw / 2 <= ix + iw and iy <= by + bh / 2 <= iy + ih
                if intersection_over_union(box, item["box"]) >= 0.20 or (analyzer == "siglip" and center_inside) or ("siglip" in item["scores"] and reverse_inside):
                    matched = item; break
            if matched is None:
                merged.append({"box": box, "label": label, "scores": {analyzer: confidence}})
            else:
                matched["scores"][analyzer] = max(confidence, matched["scores"].get(analyzer, 0.0))
                if analyzer != "siglip" and confidence >= max(matched["scores"].values()):
                    matched["box"], matched["label"] = box, label
    result = []
    for item in merged:
        votes = len(item["scores"])
        weighted = [value * float(weights.get(name, 1.0)) for name, value in item["scores"].items()]
        best = max(weighted)
        score = min(1.0, best + float(weights.get("agreement_bonus", 0.12)) * (votes - 1))
        result.append((item["box"], score, item["label"], item["scores"]))
    return sorted(result, key=lambda item: item[1], reverse=True)


def save_candidate(frame, detection, frame_path: Path, crop_path: Path, caption: str):
    box, score, label = detection
    x, y, width, height = box
    marked = frame.copy()
    cv2.rectangle(marked, (x, y), (x + width, y + height), (0, 70, 255), max(3, frame.shape[1] // 900))
    cv2.putText(marked, caption, (x, max(32, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 70, 255), 2, cv2.LINE_AA)
    pad = max(80, int(max(width, height) * 1.5))
    crop = frame[max(0, y - pad):min(frame.shape[0], y + height + pad),
                 max(0, x - pad):min(frame.shape[1], x + width + pad)]
    cv2.imwrite(str(frame_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 91])
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])


def make_report(run_dir: Path, detections: list[Detection], settings: dict):
    grouped: dict[str, list[tuple[int, Detection]]] = {}
    for index, item in enumerate(detections):
        grouped.setdefault(item.video, []).append((index, item))
    navigation, sections = [], []
    for video_index, (video, items) in enumerate(grouped.items(), 1):
        section_id = f"video-{video_index}"
        video_safe = html.escape(video)
        navigation.append(f'<a href="#{section_id}" data-video="{video_safe}"><span>{video_safe}</span><b>{len(items)}</b></a>')
        cards = []
        for index, item in items:
            label = html.escape(item.label)
            cards.append(f'''<article class="card" data-index="{index}" data-video="{video_safe}" data-label="{label}" data-stable="{str(item.stable).lower()}" data-confidence="{item.confidence}" data-votes="{len([x for x in item.analyzers.split(',') if x.strip()])}" data-time="{item.timestamp}">
<button class="image" onclick="openViewer({index})"><img loading="lazy" src="{item.crop_image}" alt="{label}"></button>
<div class="body"><b>{label} · {item.confidence:.2f}</b><span>{item.timecode}</span>
<span>анализаторы: {html.escape(item.analyzers or 'yoloe')}</span><span>рейтинг {item.score:.3f}</span><button onclick="openViewer({index})">Открыть и приблизить</button></div></article>''')
        sections.append(f'<section id="{section_id}" data-video="{video_safe}"><h2>{video_safe}<small>{len(items)} кандидатов</small></h2><div class="grid">{"".join(cards)}</div></section>')
    body = "\n".join(sections) or "<p class='empty'>Кандидаты не найдены. Попробуйте снизить confidence или профиль «Глубокий».</p>"
    settings_json = html.escape(json.dumps(settings, ensure_ascii=False, indent=2))
    viewer_data = json.dumps([{"frame": item.frame_image, "crop": item.crop_image,
                               "enhanced": item.enhanced_image, "video": item.video,
                               "label": item.label, "timecode": item.timecode, "confidence": item.confidence}
                              for item in detections], ensure_ascii=False).replace("</", "<\\/")
    document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Результаты анализа</title><style>
:root{{--bg:#101217;--panel:#1b1f28;--text:#f4f6fb;--muted:#aab2c2;--accent:#ffb454}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui}}header{{position:sticky;top:0;z-index:3;padding:14px 22px;background:#101217f2;border-bottom:1px solid #303642}}h1{{margin:0 0 9px;font-size:21px}}.controls{{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}input[type=text]{{min-width:280px;padding:9px;background:#252a35;color:white;border:1px solid #424a59;border-radius:8px}}.layout{{display:grid;grid-template-columns:270px 1fr}}nav{{position:sticky;top:103px;height:calc(100vh - 103px);overflow:auto;padding:16px;border-right:1px solid #303642}}nav a{{display:flex;gap:8px;justify-content:space-between;padding:9px;color:var(--text);text-decoration:none;border-radius:7px}}nav a:hover{{background:#252b36}}nav span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}nav b{{color:var(--accent)}}main{{padding:22px;min-width:0}}section{{scroll-margin-top:112px;margin-bottom:34px}}h2{{display:flex;justify-content:space-between;gap:15px;font-size:19px}}h2 small{{color:var(--muted);font-weight:400}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:16px}}.card{{background:var(--panel);border:1px solid #303642;border-radius:12px;overflow:hidden}}.image{{width:100%;height:220px;padding:0;border:0;background:#08090c;cursor:zoom-in}}.image img{{width:100%;height:100%;object-fit:contain}}.body{{display:grid;gap:6px;padding:12px}}.body span{{color:var(--muted)}}button,a{{color:var(--accent)}}.body button,.viewerbar button{{background:#303746;color:white;border:0;border-radius:6px;padding:8px;cursor:pointer}}
#modal{{display:none;position:fixed;inset:0;z-index:9;background:#000f;overflow:hidden;user-select:none}}#modal.open{{display:block}}#large{{position:absolute;left:50%;top:50%;max-width:none;transform-origin:center;cursor:grab}}#large:active{{cursor:grabbing}}.viewerbar{{position:fixed;left:12px;right:12px;top:12px;z-index:10;display:flex;align-items:center;gap:9px;padding:9px;background:#151922e8;border-radius:10px}}#caption{{flex:1;text-align:center}}#close{{font-size:22px}}.hint{{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);background:#111d;padding:8px 12px;border-radius:8px}}details{{margin-left:auto}}pre{{white-space:pre-wrap;color:#c9d0dd}}.empty{{padding:30px}}@media(max-width:760px){{.layout{{display:block}}nav{{position:static;height:auto;border:0;display:flex;overflow:auto}}nav a{{min-width:190px}}}}
</style></head><body><header><h1>Кандидаты: {len(detections)} · видео: {len(grouped)}</h1><div class="controls"><input id="filter" type="text" placeholder="Фильтр по видео или объекту"><select id="sort"><option value="confidence">Сначала высокий confidence</option><option value="votes">Сначала согласие моделей</option><option value="time">По таймкоду</option></select><label><input id="stable" type="checkbox"> показать стабильные</label><details><summary>Настройки запуска</summary><pre>{settings_json}</pre></details></div></header><div class="layout"><nav>{''.join(navigation)}</nav><main>{body}</main></div>
<div id="modal"><div class="viewerbar"><button onclick="move(-1)">← Предыдущий</button><button onclick="move(1)">Следующий →</button><button onclick="setSource('frame')">Кадр</button><button onclick="setSource('crop')">Фрагмент</button><button id="enhanceButton" onclick="enhanceCurrent()">✨ Улучшить</button><b id="caption"></b><button id="close">Вернуться ×</button></div><img id="large"><div class="hint">← → навигация · колесо или +/− зум до 40× · перетаскивание · AI может дорисовывать детали · Esc назад</div></div><script>
const data={viewer_data};const cards=[...document.querySelectorAll('.card')],sections=[...document.querySelectorAll('section')],filter=document.querySelector('#filter'),stable=document.querySelector('#stable'),sort=document.querySelector('#sort');function visible(){{return [...document.querySelectorAll('.card')].filter(c=>!c.hidden).map(c=>+c.dataset.index)}}function apply(){{let q=filter.value.toLowerCase();cards.forEach(c=>c.hidden=(!stable.checked&&c.dataset.stable==='true')||!(c.dataset.video+' '+c.dataset.label).toLowerCase().includes(q));sections.forEach(s=>{{s.hidden=![...s.querySelectorAll('.card')].some(c=>!c.hidden);let grid=s.querySelector('.grid'),ordered=[...grid.children].sort((a,b)=>sort.value==='time'?+a.dataset.time-+b.dataset.time:sort.value==='votes'?+b.dataset.votes-+a.dataset.votes||+b.dataset.confidence-+a.dataset.confidence:+b.dataset.confidence-+a.dataset.confidence);ordered.forEach(c=>grid.appendChild(c))}})}}filter.oninput=stable.onchange=sort.onchange=apply;apply();
const modal=document.querySelector('#modal'),img=document.querySelector('#large'),caption=document.querySelector('#caption'),enhanceButton=document.querySelector('#enhanceButton');let current=0,source='frame',scale=1,x=0,y=0,drag=false,sx=0,sy=0;function render(){{img.style.transform=`translate(calc(-50% + ${{x}}px),calc(-50% + ${{y}}px)) scale(${{scale}})`}}function load(){{const d=data[current];img.src=d[source]||d.crop;caption.textContent=`${{d.video}} · ${{d.timecode}} · ${{d.label}} ${{d.confidence.toFixed(2)}} · ${{source==='enhanced'?'AI-улучшенный фрагмент':source==='crop'?'исходный фрагмент':'полный кадр'}}`;enhanceButton.textContent=d.enhanced?'✨ Показать улучшенное':'✨ Улучшить';scale=1;x=y=0;render()}}function setSource(value){{source=value;load()}}async function enhanceCurrent(){{const d=data[current];if(d.enhanced){{setSource('enhanced');return}}enhanceButton.disabled=true;enhanceButton.textContent='⏳ Улучшаю…';caption.textContent='Загрузка модели и AI-улучшение — первый раз может занять несколько минут';try{{const response=await fetch('/api/enhance',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{index:current}})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Ошибка улучшения');d.enhanced=result.enhanced+'?v='+Date.now();source='enhanced';load()}}catch(error){{alert(error.message+'\\n\\nОткройте отчёт через open_report.command.');load()}}finally{{enhanceButton.disabled=false}}}}function openViewer(i){{current=i;source='frame';load();modal.classList.add('open');history.replaceState(null,'','#candidate-'+i)}}function closeViewer(){{modal.classList.remove('open');history.replaceState(null,'',location.pathname+location.search);document.querySelector(`[data-index="${{current}}"]`)?.scrollIntoView({{block:'center'}})}}function move(step){{let list=visible(),p=list.indexOf(current);if(list.length){{current=list[(p+step+list.length)%list.length];load()}}}}document.querySelector('#close').onclick=closeViewer;document.onkeydown=e=>{{if(!modal.classList.contains('open'))return;if(e.key==='Escape')closeViewer();if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key==='+'||e.key==='='){{scale=Math.min(40,scale*1.25);render()}}if(e.key==='-'){{scale=Math.max(.15,scale/1.25);render()}}}};modal.onwheel=e=>{{e.preventDefault();scale=Math.max(.15,Math.min(40,scale*(e.deltaY<0?1.22:.82)));render()}};img.onmousedown=e=>{{drag=true;sx=e.clientX-x;sy=e.clientY-y}};onmouseup=()=>drag=false;onmousemove=e=>{{if(drag){{x=e.clientX-sx;y=e.clientY-sy;render()}}}};
</script></body></html>'''
    (run_dir / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Пакетный поиск кандидатов во всех DJI-видео")
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--engine", choices=["yoloe", "grounding-dino"])
    parser.add_argument("--analysis-mode", choices=["single", "triple"])
    parser.add_argument("--configurations", type=Path, default=Path("configurations.json"))
    parser.add_argument("--config")
    parser.add_argument("--classes")
    parser.add_argument("--profile", choices=list(BATCH_PROFILES))
    parser.add_argument("--sample-seconds", type=float)
    parser.add_argument("--mode", choices=["full", "tiled"])
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quantize", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--yolo-model", default="yoloe-11s-seg.pt")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--siglip-conf", type=float, default=0.55)
    parser.add_argument("--weights", type=Path, default=Path("analyzer_weights.json"))
    args = parser.parse_args()
    analyzer_weights = json.loads(args.weights.read_text(encoding="utf-8")) if args.weights.exists() else {
        "yoloe": 1.0, "grounding-dino": 0.9, "siglip": 0.65, "agreement_bonus": 0.12
    }

    videos = videos_in(args.videos)
    if not videos:
        raise SystemExit("В папке videos нет видео.")
    configs = load_configurations(args.configurations)
    prompts = ([x.strip() for x in args.classes.split(",") if x.strip()] if args.classes else
               configs.get(args.config, []) if args.config else choose_prompts(configs))
    if not prompts:
        raise SystemExit(f"Неизвестная конфигурация. Доступны: {', '.join(configs)}")
    analysis_mode = args.analysis_mode or choose_option("Вариант анализа", [("single", "Обычный — один статический анализатор"),
                                                                             ("triple", "Тройной — YOLOE + Grounding DINO + SigLIP")], 1)
    engine_name = "triple" if analysis_mode == "triple" else (args.engine or choose_option("Модель", [("yoloe", "YOLOE — быстрее"),
                                                           ("grounding-dino", "Grounding DINO — точнее по тексту, но медленнее")], 1))
    profile_name = args.profile or choose_option("Глубина анализа", [("quick", "Быстрый — кадр каждые 5 секунд"),
                                                                      ("balanced", "Баланс — плитки каждые 3 секунды"),
                                                                      ("deep", "Глубокий — мелкие плитки каждые 1.5 секунды")], 2)
    settings = dict(BATCH_PROFILES[profile_name])
    for key in ("sample_seconds", "mode", "imgsz", "tile_size", "overlap"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings.update(engine=engine_name, analysis_mode=analysis_mode, profile=profile_name, prompts=prompts, conf=args.conf,
                    siglip_conf=args.siglip_conf,
                    analyzer_weights=analyzer_weights,
                    analysis="static-independent-frames", device=args.device, quantize=args.quantize,
                    videos=len(videos))

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"\nЗагрузка {engine_name}. Ищем: {', '.join(prompts)}")
    if analysis_mode == "triple":
        engines = [("yoloe", YoloEngine(args.yolo_model, prompts, args.device)),
                   ("grounding-dino", GroundingDinoEngine(args.dino_model, prompts, args.device)),
                   ("siglip", SiglipEngine(args.siglip_model, prompts, args.device))]
    else:
        engine = (YoloEngine(args.yolo_model, prompts, args.device) if engine_name == "yoloe" else
                  GroundingDinoEngine(args.dino_model, prompts, args.device))
        engines = [(engine_name, engine)]
    run_dir = args.results / datetime.now().strftime("%Y%m%d-%H%M%S")
    frames_dir, crops_dir, enhanced_dir = run_dir / "frames", run_dir / "crops", run_dir / "enhanced"
    frames_dir.mkdir(parents=True)
    crops_dir.mkdir()
    enhanced_dir.mkdir()
    found: list[Detection] = []
    started = time.perf_counter()

    for video_index, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if total else 0
        timestamp = 0.0
        print(f"[{video_index}/{len(videos)}] {video.name} ({timecode(duration)})")
        while timestamp <= duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            predictions = [(name, engine.predict(frame, settings)) for name, engine in engines]
            candidates = merge_predictions(predictions, analyzer_weights) if analysis_mode == "triple" else [
                (box, confidence, label, {engine_name: confidence}) for box, confidence, label in predictions[0][1]
            ]
            for box, confidence, label, analyzer_scores in candidates:
                movement, stable, rank = 0.0, False, confidence
                stem = f"{video.stem}_{int(timestamp * 1000):010d}_{len(found):06d}"
                frame_rel, crop_rel = f"frames/{stem}.jpg", f"crops/{stem}.jpg"
                save_candidate(frame, (box, confidence, label), run_dir / frame_rel, run_dir / crop_rel,
                               f"{label} {confidence:.2f} {timecode(timestamp)}")
                found.append(Detection(video.name, str(video.resolve()), timestamp, timecode(timestamp), label,
                                       confidence, movement, stable, rank, frame_rel, crop_rel, box, "",
                                       ", ".join(analyzer_scores), json.dumps(analyzer_scores, ensure_ascii=False)))
            timestamp += settings["sample_seconds"]
        cap.release()

    found.sort(key=lambda item: item.score, reverse=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "detections.json").write_text(json.dumps([asdict(x) for x in found], ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(found[0]).keys()) if found else list(Detection.__annotations__))
        writer.writeheader()
        for item in found:
            row = asdict(item); row["box"] = json.dumps(row["box"]); writer.writerow(row)
    make_report(run_dir, found, settings)
    print(f"\nГотово: {len(found)} кандидатов за {(time.perf_counter() - started) / 60:.1f} мин.")
    print(f"Отчёт: {(run_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
