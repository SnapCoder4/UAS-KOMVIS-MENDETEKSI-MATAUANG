import cv2
import numpy as np


def deteksi_tepi(biner):
    tepi = cv2.Canny(biner, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    tepi = cv2.dilate(tepi, kernel, iterations=1)
    tepi = cv2.morphologyEx(tepi, cv2.MORPH_CLOSE, kernel)
    return tepi


def cari_kontur(tepi):
    kontur, _ = cv2.findContours(
        tepi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return kontur


def ambil_kontur_uang(kontur, area_minimal=1500):
    kandidat = []
    for k in kontur:
        area = cv2.contourArea(k)
        if area < area_minimal:
            continue

        x, y, w, h = cv2.boundingRect(k)
        if h == 0:
            continue
        rasio = w / h

        if 1.6 <= rasio <= 3.0:
            kandidat.append((area, k))

    if len(kandidat) == 0:
        return None

    kandidat.sort(key=lambda t: t[0], reverse=True)
    return kandidat[0][1]


def crop_objek(frame, kontur_uang):
    if kontur_uang is None:
        return None, None

    x, y, w, h = cv2.boundingRect(kontur_uang)
    crop = frame[y:y + h, x:x + w]
    return crop, (x, y, w, h)


def jalankan_segmentasi(frame, biner):
    tepi = deteksi_tepi(biner)
    kontur = cari_kontur(tepi)
    terpilih = ambil_kontur_uang(kontur)
    crop, kotak = crop_objek(frame, terpilih)
    return crop, kotak
