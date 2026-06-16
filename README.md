# Rupiah Vision — Deteksi & Identifikasi Nominal Mata Uang Rupiah

Aplikasi *computer vision* sederhana untuk mendeteksi dan mengenali nominal uang kertas Rupiah secara *realtime* lewat webcam. Dibuat untuk UAS mata kuliah **TIF24 - Komputer Vision**.

Proyek ini sengaja dibuat **modular tapi sederhana**: setiap tahap pipeline ada di satu file Python sendiri, semua file berada di folder utama (flat), dan saling dihubungkan dari `main.py`. Tidak memakai struktur enterprise (`src/`, `services/`, dll), supaya mudah dipahami dan dipresentasikan per anggota.

## Pipeline

```
Webcam  ->  Preprocessing  ->  Segmentation  ->  Feature Extraction  ->  Classification  ->  Output
(cam)      (preprocess)       (segmentation)     (feature_extract)        (classifier)       (layar)
```

Penjelasan tiap tahap:

1. **Webcam** — `cam.py` membaca frame dari kamera.
2. **Preprocessing** — `preprocess.py` melakukan resize, grayscale, Gaussian blur, dan thresholding (binerisasi).
3. **Segmentation** — `segmentation.py` mendeteksi tepi (Canny), mencari kontur, lalu meng-crop uang dari background.
4. **Feature Extraction** — `feature_extract.py` mengubah gambar uang menjadi *feature vector* (warna, histogram, bentuk, tekstur).
5. **Classification** — `classifier.py` memprediksi nominal uang dari feature vector (KNN).
6. **Output** — `main.py` + `helper.py` menggambar kotak dan menampilkan nominal di layar.

## Struktur Folder

```
rupiah-vision/
├── main.py            # menggabungkan semua modul + titik mulai program
├── cam.py             # webcam capture
├── preprocess.py      # resize, blur, threshold
├── segmentation.py    # edge, contour, crop
├── feature_extract.py # feature vector (warna/histogram/bentuk/tekstur)
├── classifier.py      # klasifikasi nominal (KNN)
├── build_dataset.py   # ubah folder gambar -> dataset.csv
├── helper.py          # fungsi bantu (gambar kotak, teks, baca dataset)
├── requirements.txt
├── .gitignore         # mengecualikan folder gambar mentah dari GitHub
├── models/            # tempat model terlatih (model.pkl)
├── data/              # tempat dataset.csv & folder gambar (dibuat saat training)
└── README.md
```

## Cara Menjalankan

**1. Pasang library yang dibutuhkan**

```bash
pip install -r requirements.txt
```

**2. Siapkan dataset & latih model**

Letakkan folder gambar uang di `data/dataset/`, satu subfolder per nominal.
Tiap subfolder berisi banyak foto uang sesuai nominalnya (bukan hanya satu):

```
data/dataset/
├── 1000/      (kumpulan foto uang Rp1.000)
├── 2000/      (kumpulan foto uang Rp2.000)
├── 5000/      (kumpulan foto uang Rp5.000)
├── 10000/     (kumpulan foto uang Rp10.000)
├── 20000/     (kumpulan foto uang Rp20.000)
├── 50000/     (kumpulan foto uang Rp50.000)
└── 100000/    (kumpulan foto uang Rp100.000)
```

Lalu ubah gambar menjadi dataset angka, dan latih model:

```bash
python build_dataset.py   # gambar -> data/dataset.csv
python main.py train      # dataset.csv -> models/model.pkl
```

`build_dataset.py` membaca tiap folder (nama folder = label nominal), meng-crop
gambar, lalu mengubahnya jadi feature vector memakai `feature_extract.py`.
Jumlah gambar per nominal dibatasi lewat `MAKS_PER_NOMINAL` di file itu (default 50).

> Folder `data/dataset/` (gambar mentah) sengaja diabaikan oleh `.gitignore`,
> jadi TIDAK ikut ter-upload ke GitHub. Untuk demo, program cukup memakai `models/model.pkl`.

> Kalau model belum dilatih, program tetap bisa jalan memakai **mode tebakan warna**
> (dummy) supaya pipeline bisa didemokan saat presentasi.

**3. Jalankan deteksi realtime**

```bash
python main.py
```

Arahkan uang ke kamera. Tekan **`q`** untuk keluar.

## Pembagian Tugas & Kontribusi

| Anggota | NIM | File | Tanggung Jawab |
|---------|-----|------|----------------|
| **Christopher Haris** | 32230163 | `cam.py`, `preprocess.py`, `build_dataset.py` | Webcam capture, resize, Gaussian blur, thresholding, dan penyiapan dataset (mengubah folder gambar menjadi dataset.csv) |
| **Nicholas Agustine** | 32230146 | `segmentation.py`, `feature_extract.py` | Edge detection, contour detection, crop uang dari background, lalu ekstraksi fitur warna, histogram, bentuk, tekstur menjadi feature vector |
| **Valwa Giraldy** | 32230178 | `classifier.py`, `main.py` | Klasifikasi nominal, menampilkan hasil, menggabungkan seluruh modul jadi satu aplikasi |

Karena tiap tahap pipeline berada di file terpisah, masing-masing anggota bisa mempresentasikan bagiannya sendiri sebagai nilai individu.

## Sumber Dataset

Gambar uang yang dipakai untuk training berasal dari dataset publik di Kaggle:
`https://www.kaggle.com/datasets/anidwiastuti/rupiah-banknotes-dataset`

Folder gambar mentah dan `dataset.csv` TIDAK disertakan di repo ini (gambar
ukurannya besar). Yang disertakan cukup `models/model.pkl` (model terlatih),
sehingga program bisa langsung dijalankan untuk demo tanpa perlu training ulang.

Kalau ingin melatih ulang model, unduh gambar dari link Kaggle di atas, taruh di
`data/dataset/<nominal>/`, lalu jalankan `python build_dataset.py` dan
`python main.py train`. Kedua langkah itu akan membuat ulang `dataset.csv` dan `model.pkl`.

## Catatan Pengembangan

- Akurasi bergantung pada banyak dan kualitas data latih. Untuk menambah data,
  tambahkan lebih banyak gambar ke folder `data/dataset/<nominal>/`, lalu jalankan
  ulang `python build_dataset.py` dan `python main.py train`. Jangan menambah baris
  CSV secara manual, biarkan `build_dataset.py` yang membuatnya.
- Jumlah gambar per nominal dibatasi lewat `MAKS_PER_NOMINAL` di `build_dataset.py`
  (default 50). Naikkan nilainya kalau ingin model lebih akurat.
- Pencahayaan saat demo sebaiknya stabil dan latar polos supaya uang terdeteksi konsisten.
- Untuk pengembangan ke skripsi, KNN bisa diganti dengan model deep learning (mis. CNN) tanpa mengubah modul lain, cukup ubah `classifier.py`.