import cv2
import torch
from ultralytics import YOLO
from emotion_model import EmotionCNN
from helpers import load_class_map, preprocess_face, draw_label, FPSMeter, put_fps

YOLO_WEIGHTS = "yolov8n.pt"   # otomatis download dari Ultralytics
EMO_WEIGHTS  = "../models/emotion_cnn.pt"
CLASS_MAP    = "../data/class_map.txt"

IMG_SIZE = 48
THRESH_FACE = 0.5

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Load YOLO (deteksi wajah)
    face_detector = YOLO(YOLO_WEIGHTS)  # model face, output bbox wajah

    # 2) Load Classifier Emosi
    model = EmotionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(EMO_WEIGHTS, map_location=device))
    model.eval()

    idx2name = load_class_map(CLASS_MAP)
    fpsm = FPSMeter()

    # 3) Buka webcam (0 atau 1 sesuai perangkat)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows biasanya stabil dengan CAP_DSHOW
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak dapat dibuka. Coba index 1 atau cek driver.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 4) Deteksi wajah
            # Ultralytics YOLO menghasilkan result per frame
            results = face_detector(frame, verbose=False)  # default bgr np.array
            for r in results:
                if r.boxes is None: 
                    continue
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0].item())
                    if conf < THRESH_FACE: 
                        continue
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        continue

                    # 5) Preprocess & prediksi emosi
                    inp = preprocess_face(face_roi, size=IMG_SIZE).to(device)
                    with torch.no_grad():
                        logits = model(inp)
                        pred_idx = int(torch.argmax(logits, dim=1).item())
                        label = idx2name.get(pred_idx, str(pred_idx))

                    draw_label(frame, (x1, y1, x2, y2), label, conf=conf)

            fps = fpsm.tick()
            put_fps(frame, fps)
            cv2.imshow("YOLO Face + Emotion", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
