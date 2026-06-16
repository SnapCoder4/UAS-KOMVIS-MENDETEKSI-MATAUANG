import os
import csv
import cv2


def gambar_kotak(frame, kotak):
    """Menggambar kotak hijau di sekeliling objek yang terdeteksi."""
    if kotak is None:
        return frame
    x, y, w, h = kotak
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def tulis_teks(frame, teks, posisi=(10, 30)):
    """Menulis teks (mis. hasil nominal) di pojok kiri atas frame."""
    cv2.putText(
        frame, teks, posisi,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    return frame


def baca_dataset(path=os.path.join("data", "dataset.csv")):
    """
    Membaca dataset untuk training.
    Format CSV: kolom terakhir = label, kolom lain = fitur.
    Mengembalikan (X, y). Kosong kalau file belum ada.
    """
    X, y = [], []
    if not os.path.exists(path):
        return X, y

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for baris in reader:
            if not baris:
                continue
            fitur = [float(n) for n in baris[:-1]]
            label = baris[-1]
            X.append(fitur)
            y.append(label)
    return X, y
