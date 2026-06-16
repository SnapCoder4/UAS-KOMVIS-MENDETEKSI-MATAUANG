import cv2
import numpy as np


def fitur_warna(crop):
    """
    Mengambil rata-rata warna dalam ruang HSV.
    Mengembalikan [mean_H, mean_S, mean_V].
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])
    return [mean_h, mean_s, mean_v]


def fitur_histogram(crop, bins=8):
    """
    Membuat histogram warna Hue lalu dinormalisasi.
    Mengembalikan daftar angka sepanjang 'bins'.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()


def fitur_bentuk(crop):
    """
    Menghitung aspect ratio (lebar dibagi tinggi).
    Membantu membedakan orientasi/ukuran lembar uang.
    """
    tinggi, lebar = crop.shape[:2]
    if tinggi == 0:
        return [0.0]
    return [lebar / tinggi]


def fitur_tekstur(crop):
    """
    Mengukur tekstur dari standar deviasi gambar abu-abu.
    Nilai besar = banyak detail/pola, nilai kecil = polos.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return [float(np.std(gray))]


def buat_feature_vector(crop):
    """
    Menggabungkan semua fitur menjadi SATU feature vector.
    Input  : gambar uang hasil crop
    Output : list angka (gabungan warna + histogram + bentuk + tekstur)
    """
    if crop is None or crop.size == 0:
        return None

    # samakan ukuran dulu supaya fitur konsisten
    crop = cv2.resize(crop, (200, 100))

    vektor = []
    vektor += fitur_warna(crop)
    vektor += fitur_histogram(crop)
    vektor += fitur_bentuk(crop)
    vektor += fitur_tekstur(crop)
    return vektor
