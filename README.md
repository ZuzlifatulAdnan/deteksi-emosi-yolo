# Emosi Wajah — Real-time Emotion Recognition

> README lengkap untuk proyek deteksi & klasifikasi emosi wajah secara real-time menggunakan YOLO untuk deteksi wajah dan CNN (dilatih dari dataset `fer2013.csv`) untuk klasifikasi emosi.

---

## Ringkasan

Proyek ini menangkap video dari webcam, mendeteksi wajah secara cepat dengan model YOLO (atau varian deteksi wajah lainnya), lalu mengekstrak area wajah untuk diklasifikasikan menjadi emosi menggunakan model CNN yang dilatih pada dataset `fer2013.csv`. Cocok untuk demo, penelitian ringan, atau integrasi ke aplikasi GUI/embedded.

## Fitur

* Deteksi wajah real-time (webcam) menggunakan YOLO.
* Klasifikasi emosi menggunakan CNN terlatih pada `fer2013.csv` (ekspresi seperti: `angry, disgust, fear, happy, sad, surprise, neutral`).
* Script untuk pelatihan (`train.py` / `train_classifier.py`) dan inferensi real-time (`realtime_infer.py` / `detect_and_classify.py`).
* Export model (TorchScript / ONNX) untuk deployment ringan.

## Prasyarat

* Python 3.8 — 3.11 (disarankan 3.10)
* GPU dengan CUDA untuk pelatihan cepat (opsional untuk inferensi CPU)
* Sistem operasi: Windows, Linux, atau macOS

## Dependensi (contoh)

Gunakan virtual environment (venv / conda). Contoh file `requirements.txt` minimal:

```
torch>=1.12
torchvision
opencv-python
numpy
pandas
matplotlib
scikit-learn
tensorflow==2.10.0   # jika memilih TF untuk training/konversi (opsional)
Pillow
tqdm
albumentations
```

Jika menggunakan YOLOv5/YOLOv8, tambahkan dependensi sesuai repo (mis. `git+https://github.com/ultralytics/ultralytics.git` atau clone repo yolov5 dan install requirements).

## Struktur Folder (disarankan)

```
emosi-wajah/
├─ data/
│  ├─ fer2013.csv
│  ├─ datasets/            # folder dataset yang sudah dikelompokkan (train/val/test)
├─ models/
│  ├─ detector/            # weights untuk YOLO (detector)
│  └─ classifier/          # weights untuk CNN (emotion classifier)
├─ notebooks/
├─ src/
│  ├─ train_classifier.py
│  ├─ detect_and_classify.py
│  ├─ realtime_infer.py
│  ├─ dataset_utils.py
│  └─ utils.py
├─ requirements.txt
├─ README.md
└─ weights/
   ├─ yolov5_face.pt
   └─ emotion_cnn.pth
```

## Persiapan Dataset

1. Jika Anda memiliki `fer2013.csv`:

   * File `fer2013.csv` biasa berisi kolom `emotion`, `pixels`, dan `Usage`.
   * Gunakan `dataset_utils.py` untuk mengonversi `pixels` (string berisi pixel) menjadi gambar dan menyimpan ke folder `datasets/train`, `datasets/val`, `datasets/test`.

Contoh (ringkas) langkah konversi:

```python
# dataset_utils.py (sekilas)
import numpy as np
from PIL import Image

# parsing pixels -> numpy array 48x48
pixels = '70 80 82 ...'
arr = np.fromstring(pixels, sep=' ', dtype=np.uint8).reshape(48,48)
img = Image.fromarray(arr)
img.save('class_x/xxxx.png')
```

## Pelatihan Classifier (contoh)

1. Siapkan struktur folder `datasets/train/<label>`, `datasets/val/<label>`.
2. Jalankan:

```
python src/train_classifier.py --data datasets --epochs 40 --batch-size 64 --lr 1e-3 --img-size 48
```

Contoh flags yang umum:

* `--model` : arsitektur CNN (mis. custom / resnet18 / mobilenet)
* `--pretrained` : gunakan pre-trained backbone
* `--augment` : aktifkan augmentasi (albumentations)

## Inference Real-time

Contoh penggunaan `realtime_infer.py`:

```
python src/realtime_infer.py --detector weights/yolov5_face.pt --classifier weights/emotion_cnn.pth --camera 0
```

Flow sederhana script inference:

1. Baca frame dari webcam.
2. Jalankan detector untuk mendapatkan bounding box wajah.
3. Crop, resize (48x48 atau ukuran model), normalisasi.
4. Predict dengan classifier.
5. Tampilkan label + confidence di frame dan overlay bounding box.

## File penting & fungsi singkat

* `src/detect_and_classify.py` — alternatif script yang memisah deteksi & klasifikasi untuk saving frames
* `src/realtime_infer.py` — script utama untuk demo webcam
* `src/train_classifier.py` — training loop, checkpointing, augmentasi, scheduler
* `src/dataset_utils.py` — helper untuk konversi `fer2013.csv` -> folder gambar
* `src/utils.py` — fungsi util: load\_weights, preprocess, postprocess, label\_map

## Tips untuk akurasi lebih baik

* Gunakan augmentasi: rotations, shifts, horizontal flips (hati-hati untuk ekspresi lateral), brightness/contrast jitter.
* Pretrain backbone (misal ResNet18) lalu fine-tune classifier.
* Atur class-weight atau gunakan oversampling untuk dataset tidak seimbang.
* Eksperimen ukuran input: 48x48 sederhana, namun 64x64 atau 96x96 bisa meningkatkan akurasi jika model cukup besar.
* Validasi silang (k-fold) untuk evaluasi yang lebih handal.

## Troubleshooting umum

* **Webcam tidak terbaca**: pastikan indeks kamera benar (`--camera 0`, `1`, ...), atau perangkat lain tidak menguasai kamera.
* **Model gagal dimuat**: pastikan versi PyTorch kompatibel, dan path weights benar.
* **Detektor tidak mendeteksi wajah kecil**: gunakan model deteksi dengan resolusi input lebih tinggi atau sesuaikan NMS/confidence threshold.

## Perintah berguna

* Eksport model PyTorch -> TorchScript:

```python
# contoh
traced = torch.jit.trace(model, dummy_input)
traced.save('classifier_script.pt')
```

* Convert to ONNX (untuk integrasi ke platform lain):

```python
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=12)
```

## Evaluasi & Metrics

* Hitung `accuracy`, `precision`, `recall`, `f1-score` per kelas.
* Confusion matrix sangat membantu untuk melihat kelas mana yang saling tertukar.

## Contoh `requirements.txt`

```
numpy
pandas
opencv-python
matplotlib
torch
torchvision
albumentations
scikit-learn
Pillow
tqdm

# opsional (jika menggunakan ultralytics YOLO):
# ultralytics
```

## Lisensi

Lisensi proyek: MIT (atau sesuaikan dengan kebutuhan Anda).

## Kontributor

* Nama Anda — pengembang utama

---

Jika Anda mau, saya bisa:

* Membuat `train_classifier.py` lengkap (loop training, validasi, checkpointing).
* Menulis `realtime_infer.py` siap pakai yang cocok untuk Windows + webcam.
* Menyediakan template `requirements.txt` dan `dataset_utils.py` untuk langsung dijalankan.

Tentukan salah satu yang Anda mau saya buat sekarang dan saya akan langsung tuliskan kodenya.
