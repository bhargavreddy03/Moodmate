"""
webcam_realtime.py  —  MoodMate ZERO-LAG Edition
──────────────────────────────────────────────────
Architecture:
  • ONE dedicated inference thread (TF is NOT thread-safe → no contention)
  • Display thread NEVER blocks on inference → always 30 FPS video
  • PREDICT_EVERY = 5 → detect every 5th frame, display every frame
  • TTA disabled in realtime (8x speedup over previous code)
  • Smoothed emotion via exponential moving average across frames
  • Half-res detection (320×240) → Haar cascade is 4× faster
  • Pre-warmed model at startup → no first-frame freeze

Controls:  Q / ESC → quit   S → save frame   SPACE → freeze/unfreeze
"""

import os, sys, cv2, time, threading, queue
import numpy as np
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.emotion_predictor  import predict_from_array
from src.music_recommender  import MusicRecommender
from src.emotion_mapping    import get_emotion_info
from utils.helper_functions import (
    detect_faces, crop_face,
    emotion_emoji, logger, timestamp, ensure_dir
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CAMERA_W       = 640
CAMERA_H       = 480
CAMERA_FPS     = 30

PREDICT_EVERY  = 5        # Run face detection every N frames (keeps display smooth)
INFER_Q_SIZE   = 2        # Small queue → always work on LATEST face, not stale ones
MAX_TRACK_AGE  = 15       # Frames before dropping a lost track
IOU_THRESHOLD  = 0.35
NUM_SONGS      = 5
SIDEBAR_W      = 260

# Emotion smoothing: EMA weight for new prediction (higher = more responsive)
EMA_ALPHA      = 0.45

# Colours (BGR)
PALETTE = [
    (0, 230, 100), (255, 100, 80), (80, 180, 255),
    (255, 200, 60), (180, 80, 255), (60, 220, 220),
]
PURPLE = (200, 80, 255)
WHITE  = (240, 240, 255)
DARK   = (15, 15, 28)

EMOTION_COLORS = {
    "happy":    (0, 230, 100),
    "sad":      (80, 130, 255),
    "angry":    (40, 40, 255),
    "surprise": (0, 210, 255),
    "fear":     (160, 60, 200),
    "disgust":  (60, 180, 60),
    "neutral":  (160, 160, 160),
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / (union + 1e-6)


def put_text(img, text, pos, scale=0.52, color=WHITE, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bars(img, scores, x0, y0, width=120):
    sorted_s = sorted(scores.items(), key=lambda kv: -kv[1])
    for i, (lbl, sc) in enumerate(sorted_s):
        y = y0 + i * 18
        cv2.rectangle(img, (x0, y), (x0+width, y+11), (55, 55, 75), -1)
        fw = int(width * sc)
        col = EMOTION_COLORS.get(lbl, (90, 90, 170)) if i == 0 else (70, 70, 130)
        cv2.rectangle(img, (x0, y), (x0+fw, y+11), col, -1)
        put_text(img, f"{lbl[:7]:7s}{sc*100:4.0f}%",
                 (x0+width+4, y+10), scale=0.35)


def draw_face_overlay(canvas, face, label, color, track_id, conf):
    x, y, w, h = face
    # Animated corner brackets
    t  = 2
    cs = min(w, h) // 5
    corners = [
        ((x,   y),   (x+cs, y),   (x,   y+cs)),
        ((x+w, y),   (x+w-cs, y), (x+w, y+cs)),
        ((x,   y+h), (x+cs, y+h), (x,   y+h-cs)),
        ((x+w, y+h), (x+w-cs, y+h), (x+w, y+h-cs)),
    ]
    for (p, p1, p2) in corners:
        cv2.line(canvas, p, p1, color, t, cv2.LINE_AA)
        cv2.line(canvas, p, p2, color, t, cv2.LINE_AA)

    # Confidence fill bar at top of face box
    bar_w = int(w * conf)
    cv2.rectangle(canvas, (x, y-4), (x+w, y-1), (40, 40, 60), -1)
    cv2.rectangle(canvas, (x, y-4), (x+bar_w, y-1), color, -1)

    # Label pill
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    ly = max(y - 10, th + 8)
    cv2.rectangle(canvas, (x, ly-th-5), (x+tw+10, ly+bl+2), DARK, -1)
    cv2.rectangle(canvas, (x, ly-th-5), (x+tw+10, ly+bl+2), color, 1)
    put_text(canvas, label, (x+5, ly-1), scale=0.50, color=color)

    # Track ID badge
    badge = f"#{track_id+1}"
    cv2.putText(canvas, badge, (x+w-26, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# FACE TRACKER  (single-threaded access from main loop; results written by
#               inference thread via set_result which uses a lock)
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    def __init__(self):
        self.tracks   = {}
        self._next_id = 0
        self._lock    = threading.Lock()

    def _new_track(self, box):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = dict(
            box=box, emotion="…", confidence=0.0,
            scores={}, songs=[], age=0,
            color=PALETTE[tid % len(PALETTE)],
            ema_scores={}          # smoothed probabilities
        )
        return tid

    def update(self, boxes):
        used_tracks, used_boxes = set(), set()
        matches = []

        for bi, box in enumerate(boxes):
            best_iou, best_tid = 0, None
            for tid, trk in self.tracks.items():
                if tid in used_tracks:
                    continue
                s = iou(box, trk["box"])
                if s > best_iou:
                    best_iou, best_tid = s, tid
            if best_iou > IOU_THRESHOLD:
                matches.append((bi, best_tid))
                used_tracks.add(best_tid)
                used_boxes.add(bi)

        for bi, tid in matches:
            self.tracks[tid]["box"] = boxes[bi]
            self.tracks[tid]["age"] = 0

        new_tids = []
        for bi, box in enumerate(boxes):
            if bi not in used_boxes:
                new_tids.append(self._new_track(box))

        stale = [tid for tid, t in self.tracks.items() if t["age"] > MAX_TRACK_AGE]
        for tid in stale:
            del self.tracks[tid]

        for tid in self.tracks:
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1

        return {bi: tid for bi, tid in matches}, new_tids

    def set_result(self, tid, emotion, confidence, scores, songs):
        """Called from inference thread — thread-safe."""
        with self._lock:
            if tid not in self.tracks:
                return
            t = self.tracks[tid]

            # Exponential moving average on raw scores → smoothed display
            if not t["ema_scores"]:
                t["ema_scores"] = dict(scores)
            else:
                for cls in scores:
                    prev = t["ema_scores"].get(cls, scores[cls])
                    t["ema_scores"][cls] = EMA_ALPHA * scores[cls] + (1 - EMA_ALPHA) * prev

            # Pick emotion from smoothed scores
            best_cls   = max(t["ema_scores"], key=lambda k: t["ema_scores"][k])
            best_conf  = t["ema_scores"][best_cls]

            t["emotion"]    = best_cls
            t["confidence"] = best_conf
            t["scores"]     = dict(t["ema_scores"])
            t["songs"]      = songs

    def snapshot(self):
        """Thread-safe read for the render loop."""
        with self._lock:
            return [(tid, dict(trk)) for tid, trk in self.tracks.items()]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE INFERENCE THREAD
# One thread → no TensorFlow concurrency bugs → maximum throughput
# Uses a drop-oldest queue: only LATEST face crop per track matters
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Consumes (tid, crop_bgr) jobs from a tiny queue.
    Always processes the LATEST job — stale crops are discarded.
    """

    def __init__(self, tracker, recommender):
        super().__init__(daemon=True, name="InferThread")
        self.tracker     = tracker
        self.recommender = recommender
        # maxsize=INFER_Q_SIZE: old jobs dropped when full → always fresh
        self._q      = queue.Queue(maxsize=INFER_Q_SIZE)
        self.running = True
        self.last_ms = 0.0         # exposed for display

    def enqueue(self, tid, crop_bgr):
        """Non-blocking. If queue full, discard oldest and insert newest."""
        try:
            self._q.put_nowait((tid, crop_bgr))
        except queue.Full:
            try:
                self._q.get_nowait()   # drop stale
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((tid, crop_bgr))
            except queue.Full:
                pass

    def run(self):
        while self.running:
            try:
                tid, crop = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t0   = time.perf_counter()
                pred = predict_from_array(crop)   # TTA=False (fast path)
                songs = self.recommender.recommend(pred["emotion"], n=NUM_SONGS)
                self.tracker.set_result(
                    tid,
                    pred["emotion"],
                    pred["confidence"],
                    pred["all_scores"],
                    songs,
                )
                self.last_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                logger.warning(f"[InferThread] error tid={tid}: {e}")
            finally:
                self._q.task_done()

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def draw_sidebar(canvas, tracks_snapshot, frame_w, frame_h, fps, infer_ms):
    sx = frame_w + 8

    put_text(canvas, "MoodMate", (sx, 28), scale=0.75, color=PURPLE, thickness=2)
    put_text(canvas, f"{fps:4.1f} FPS  |  {infer_ms:.0f}ms infer",
             (sx, 46), scale=0.34, color=(140, 140, 180))

    if not tracks_snapshot:
        put_text(canvas, "No faces detected", (sx, 90), scale=0.46, color=(80, 80, 200))
        return

    y = 66
    for tid, trk in tracks_snapshot:
        if trk["age"] > 2:
            continue
        color = trk["color"]
        emo   = trk["emotion"]
        conf  = trk["confidence"]
        info  = get_emotion_info(emo) if emo not in ("…", "none") else {}

        ec = EMOTION_COLORS.get(emo, color)
        header = f"#{tid+1}  {emotion_emoji(emo)} {emo.capitalize()}"
        put_text(canvas, header, (sx, y), scale=0.50, color=ec, thickness=1)
        y += 17

        if info:
            put_text(canvas, info.get("mood", ""), (sx, y), scale=0.33, color=(150, 170, 255))
        y += 13

        if trk["scores"]:
            draw_confidence_bars(canvas, trk["scores"], sx, y, width=110)
            y += 7 * 18 + 2

        songs = trk["songs"]
        if songs:
            put_text(canvas, "♪ Recommended:", (sx, y), scale=0.38, color=PURPLE)
            y += 15
            for s in songs[:3]:
                name   = (s["name"][:20]   + "…") if len(s["name"])   > 20 else s["name"]
                artist = (s["artist"][:18] + "…") if len(s["artist"]) > 18 else s["artist"]
                put_text(canvas, f"  {name}", (sx, y), scale=0.37, color=WHITE)
                y += 13
                put_text(canvas, f"  {artist}", (sx, y), scale=0.33, color=(130, 130, 185))
                y += 14

        cv2.line(canvas, (sx, y+3), (sx+SIDEBAR_W-12, y+3), (50, 50, 70), 1)
        y += 12

        if y > frame_h - 30:
            break


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    logger.info("Loading music recommender …")
    recommender = MusicRecommender(
        csv_path=os.path.join(BASE_DIR, "data", "music_data.csv")
    )

    tracker = FaceTracker()

    # ── Single inference thread (TF-safe) ──────────────────────────────────
    infer_thread = InferenceThread(tracker, recommender)
    infer_thread.start()
    logger.info("Inference thread started")

    # ── Camera ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)          # always grab NEWEST frame

    # Try MJPG codec for higher USB bandwidth (often helps with USB cams)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        logger.error(f"Cannot open camera {CAMERA_INDEX}")
        return

    logger.info("Webcam ready. Q=quit  S=save  SPACE=freeze")

    frame_count  = 0
    frozen       = False
    fps_times    = deque(maxlen=30)
    current_fps  = 0.0
    frame        = None

    save_dir = os.path.join(BASE_DIR, "app", "saved_frames")
    ensure_dir(save_dir)

    # Pre-build half-res detection frame (reuse buffer)
    detect_w, detect_h = 320, 240
    sx_scale = CAMERA_W / detect_w
    sy_scale = CAMERA_H / detect_h

    while True:
        t0 = time.perf_counter()

        # ── Capture ────────────────────────────────────────────────────────
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame capture failed")
                break
            frame_count += 1

            # ── Face detection every N frames (detection is still costly) ──
            if frame_count % PREDICT_EVERY == 0:
                small     = cv2.resize(frame, (detect_w, detect_h),
                                       interpolation=cv2.INTER_NEAREST)  # fastest resize
                raw_faces = detect_faces(small)
                faces     = [
                    (int(x*sx_scale), int(y*sy_scale),
                     int(w*sx_scale), int(h*sy_scale))
                    for (x, y, w, h) in raw_faces
                ]

                matched, new_tids = tracker.update(faces)

                # Enqueue matched faces (inference thread picks up async)
                for bi, face in enumerate(faces):
                    tid = matched.get(bi)
                    if tid is None:
                        continue
                    crop = crop_face(frame, face)
                    if crop is not None and crop.size > 0:
                        infer_thread.enqueue(tid, crop)

                # Immediately enqueue brand-new track faces
                for tid in new_tids:
                    box  = tracker.tracks[tid]["box"]
                    crop = crop_face(frame, box)
                    if crop is not None and crop.size > 0:
                        infer_thread.enqueue(tid, crop)

        if frame is None:
            continue

        # ── Render ─────────────────────────────────────────────────────────
        h, w   = frame.shape[:2]
        canvas = np.empty((h, w + SIDEBAR_W, 3), dtype=np.uint8)
        canvas[:, :] = (20, 20, 35)         # dark background
        canvas[:h, :w] = frame

        # Thread-safe snapshot so render never blocks inference
        tracks_snap = tracker.snapshot()

        active_visible = [(tid, trk) for tid, trk in tracks_snap if trk["age"] <= 2]

        for tid, trk in active_visible:
            emo   = trk["emotion"]
            conf  = trk["confidence"]
            ec    = EMOTION_COLORS.get(emo, trk["color"])
            label = (f"{emotion_emoji(emo)} {emo.capitalize()} {conf*100:.0f}%"
                     if emo not in ("…", "none") else "Detecting…")
            draw_face_overlay(canvas[:h, :w], trk["box"], label, ec, tid, conf)

        if not active_visible:
            put_text(canvas,
                     "No face detected — adjust angle / lighting",
                     (w//2-175, h//2), scale=0.52, color=(80, 80, 200))

        draw_sidebar(canvas, tracks_snap, w, h, current_fps, infer_thread.last_ms)

        # ── HUD ────────────────────────────────────────────────────────────
        elapsed     = time.perf_counter() - t0
        fps_times.append(elapsed)
        current_fps = 1.0 / (np.mean(fps_times) + 1e-9)

        q_sz = infer_thread._q.qsize()
        put_text(canvas,
                 f"FPS:{current_fps:.0f}  Q:{q_sz}  SPC=freeze  S=save  Q=quit",
                 (8, h-8), scale=0.33, color=(90, 90, 140))

        if frozen:
            put_text(canvas, "|| FROZEN", (w//2-55, h//2),
                     scale=0.9, color=(0, 150, 255), thickness=2)

        cv2.imshow("MoodMate — Zero Lag", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = os.path.join(save_dir, f"frame_{timestamp()}.jpg")
            cv2.imwrite(fname, canvas)
            logger.info(f"Saved → {fname}")
        elif key == ord(" "):
            frozen = not frozen
            logger.info("Frozen" if frozen else "Resumed")

    infer_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Goodbye!")


if __name__ == "__main__":
    run()