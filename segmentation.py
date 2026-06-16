import cv2


def deteksi_tepi(biner):
    """Mendeteksi tepi objek dengan algoritma Canny."""
    return cv2.Canny(biner, 50, 150)


def cari_kontur(tepi):
    """
    Mencari semua kontur (garis tepi tertutup) pada gambar tepi.
    Mengembalikan daftar kontur.
    """
    kontur, _ = cv2.findContours(
        tepi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return kontur


def ambil_kontur_terbesar(kontur):
    """
    Memilih kontur dengan area paling besar.
    Asumsinya, objek terbesar di layar adalah uang.
    Mengembalikan None kalau tidak ada kontur.
    """
    if len(kontur) == 0:
        return None
    return max(kontur, key=cv2.contourArea)


def crop_objek(frame, kontur_terbesar, area_minimal=3000):
    """
    Memotong (crop) uang dari background memakai bounding box.
    Input  : frame berwarna + kontur terbesar
    Output : gambar hasil crop, atau None kalau objek terlalu kecil.
    """
    if kontur_terbesar is None:
        return None, None

    if cv2.contourArea(kontur_terbesar) < area_minimal:
        return None, None

    x, y, w, h = cv2.boundingRect(kontur_terbesar)
    crop = frame[y:y + h, x:x + w]
    return crop, (x, y, w, h)


def jalankan_segmentasi(frame, biner):
    """
    Menggabungkan seluruh proses segmentasi.
    Input  : frame berwarna + gambar biner dari preprocessing
    Output : (gambar_crop, kotak) -> kotak = (x, y, w, h) untuk digambar
    """
    tepi = deteksi_tepi(biner)
    kontur = cari_kontur(tepi)
    terbesar = ambil_kontur_terbesar(kontur)
    crop, kotak = crop_objek(frame, terbesar)
    return crop, kotak
