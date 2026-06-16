import cv2

def buka_kamera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka. Cek webcam atau ganti index.")
    return cap


def ambil_frame(cap):
    berhasil, frame = cap.read()
    return berhasil, frame


def tutup_kamera(cap):
    """Menutup webcam dan menutup semua jendela OpenCV."""
    cap.release()
    cv2.destroyAllWindows()
