from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLOE


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
WINDOW = "Drone AI Viewer"
VIEWER_PROFILES = {
    "fast": {"imgsz": 512, "analysis_interval": 1.2, "description": "Быстрый — меньше нагрузка и плавнее видео"},
    "balanced": {"imgsz": 640, "analysis_interval": 0.65, "description": "Баланс — рекомендуемый режим"},
    "quality": {"imgsz": 960, "analysis_interval": 1.0, "description": "Качественный — лучше мелкие объекты, но медленнее"},
}


def videos_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def choose(title: str, entries: list[tuple[str, str]]) -> str:
    print(f"\n{title}:\n")
    for index, (_, description) in enumerate(entries, 1):
        print(f"  {index}. {description}")
    while True:
        try:
            return entries[int(input("\nНомер: ")) - 1][0]
        except (ValueError, IndexError):
            print("Введите номер из списка.")


def fit(frame, width=1600, height=930):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h, 1.0)
    return frame if scale >= 1 else cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def zoom_view(frame, zoom: float, center: tuple[float, float]):
    if zoom <= 1.001:
        return frame
    height, width = frame.shape[:2]
    crop_width, crop_height = max(1, int(width / zoom)), max(1, int(height / zoom))
    cx = int(max(crop_width / 2, min(width - crop_width / 2, center[0] * width)))
    cy = int(max(crop_height / 2, min(height - crop_height / 2, center[1] * height)))
    x1, y1 = cx - crop_width // 2, cy - crop_height // 2
    crop = frame[y1:y1 + crop_height, x1:x1 + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def clock(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02}:{value % 60:02}"


def draw(frame, detections, prompts):
    for box, score, class_id in detections:
        x1, y1, x2, y2 = box
        label = prompts[class_id] if 0 <= class_id < len(prompts) else str(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 70, 255), 3)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(28, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, (0, 70, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="DJI-видеоплеер с YOLOE")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--configurations", type=Path, default=Path("configurations.json"))
    parser.add_argument("--config")
    parser.add_argument("--model", default="yoloe-11s-seg.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--profile", choices=list(VIEWER_PROFILES))
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, default=.12)
    parser.add_argument("--analysis-interval", type=float)
    args = parser.parse_args()

    videos = videos_in(args.videos)
    if args.video:
        video = args.video
    else:
        if not videos:
            raise SystemExit("В videos нет видео.")
        selected = choose("Видео", [(str(path), path.name) for path in videos])
        video = Path(selected)
    configs = json.loads(args.configurations.read_text(encoding="utf-8"))
    if args.config:
        prompts = configs.get(args.config)
        if not prompts:
            raise SystemExit(f"Нет конфигурации {args.config}")
    else:
        name = choose("Что искать", [(name, f"{name}: {', '.join(values)}") for name, values in configs.items()])
        prompts = configs[name]
    profile_name = args.profile or choose(
        "Качество обработки видео",
        [(name, values["description"]) for name, values in VIEWER_PROFILES.items()],
    )
    profile = VIEWER_PROFILES[profile_name]
    imgsz = args.imgsz if args.imgsz is not None else profile["imgsz"]
    analysis_interval = args.analysis_interval if args.analysis_interval is not None else profile["analysis_interval"]

    print(f"\nЗагрузка YOLOE. Профиль: {profile_name}, размер: {imgsz}, AI каждые {analysis_interval:g} сек.")
    print(f"Ищем: {', '.join(prompts)}")
    model = YOLOE(args.model)
    model.set_classes(prompts)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    requests: queue.Queue = queue.Queue(maxsize=1)
    responses: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                frame, position, confidence = requests.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                result = model.predict(frame, device=args.device, imgsz=imgsz, conf=confidence,
                                       quantize="fp32", verbose=False)[0]
                values = [] if result.boxes is None else [
                    ([int(v) for v in box], float(score), int(class_id))
                    for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(),
                                                    result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())
                ]
                while not responses.empty():
                    responses.get_nowait()
                responses.put((position, values))
            except Exception as error:
                print(f"AI: {error}")

    threading.Thread(target=worker, daemon=True).start()
    paused, speed, confidence = False, 1.0, args.conf
    detections, last_submit, last_frame = [], -999.0, None
    zoom, zoom_center = 1.0, [0.5, 0.5]
    dragging, drag_start = False, (0, 0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, _):
        nonlocal zoom, zoom_center, dragging, drag_start
        if event == cv2.EVENT_MOUSEWHEEL:
            zoom = min(12.0, zoom * 1.25) if flags > 0 else max(1.0, zoom / 1.25)
        elif event == cv2.EVENT_LBUTTONDOWN and zoom > 1:
            dragging, drag_start = True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx, dy = x - drag_start[0], y - drag_start[1]
            zoom_center[0] = max(0.0, min(1.0, zoom_center[0] - dx / 1600 / zoom))
            zoom_center[1] = max(0.0, min(1.0, zoom_center[1] - dy / 930 / zoom))
            drag_start = (x, y)

    cv2.setMouseCallback(WINDOW, mouse)
    print("\nПробел пауза · ←/→ ±5с · J/L ±30с · 1/2/4 скорость · +/- зум · 0 сброс · [/] confidence · S кадр · Q выход")
    try:
        while True:
            if not paused or last_frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame = frame
            else:
                frame = last_frame.copy()
            position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if position - last_submit >= analysis_interval and requests.empty():
                requests.put((frame.copy(), position, confidence)); last_submit = position
            try:
                while True:
                    _, detections = responses.get_nowait()
            except queue.Empty:
                pass
            view = frame.copy(); draw(view, detections, prompts)
            cv2.rectangle(view, (0, 0), (view.shape[1], 68), (15, 15, 15), -1)
            cv2.putText(view, f"{video.name}  {clock(position)}/{clock(duration)}  {'PAUSE' if paused else f'{speed:g}x'}  {profile_name}  conf:{confidence:.2f}",
                        (18, 43), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2, cv2.LINE_AA)
            if zoom > 1:
                cv2.putText(view, f"ZOOM {zoom:.1f}x", (view.shape[1] - 210, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, .85, (80, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, fit(zoom_view(view, zoom, tuple(zoom_center))))
            delay = 30 if paused else max(1, int(1000 / fps / speed))
            key = cv2.waitKeyEx(delay)
            if key in (27, ord('q'), ord('Q')): break
            if key == 32: paused = not paused
            if key in (2424832, 81, ord('a'), ord('A')): cap.set(cv2.CAP_PROP_POS_MSEC, max(0, position - 5) * 1000); last_frame = None
            if key in (2555904, 83, ord('d'), ord('D')): cap.set(cv2.CAP_PROP_POS_MSEC, min(duration, position + 5) * 1000); last_frame = None
            if key in (ord('j'), ord('J')): cap.set(cv2.CAP_PROP_POS_MSEC, max(0, position - 30) * 1000); last_frame = None
            if key in (ord('l'), ord('L')): cap.set(cv2.CAP_PROP_POS_MSEC, min(duration, position + 30) * 1000); last_frame = None
            if key in (ord('1'), ord('2'), ord('4')): speed = float(chr(key))
            if key in (ord('+'), ord('=')): zoom = min(12.0, zoom * 1.25)
            if key == ord('-'): zoom = max(1.0, zoom / 1.25)
            if key == ord('0'): zoom, zoom_center = 1.0, [0.5, 0.5]
            if key == ord(']'): confidence = min(.95, confidence + .02)
            if key == ord('['): confidence = max(.01, confidence - .02)
            if key in (ord('s'), ord('S')):
                folder = Path("captures"); folder.mkdir(exist_ok=True)
                target = folder / f"{video.stem}_{int(position * 1000):010d}.jpg"
                cv2.imwrite(str(target), frame); print(f"Сохранено: {target}")
    finally:
        stop.set(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
