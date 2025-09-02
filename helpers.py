import time
import cv2
import torch
import numpy as np

def load_class_map(path):
    idx2name = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split(" ", 1)
            idx2name[int(k)] = v
    return idx2name

def preprocess_face(img_bgr, size=48):
    # img_bgr: ROI wajah BGR
    face = cv2.resize(img_bgr, (size, size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))  # CHW
    face = torch.tensor(face).unsqueeze(0)  # (1,3,H,W)
    return face

def draw_label(img, xyxy, label, conf=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    text = f"{label}" if conf is None else f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, max(0, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

class FPSMeter:
    def __init__(self):
        self.last = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps

def put_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
