
import cv2


def resize_gambar(frame, lebar=640, tinggi=480):
    """Mengubah ukuran gambar agar konsisten."""
    return cv2.resize(frame, (lebar, tinggi))


def ke_grayscale(frame):
    """Mengubah gambar berwarna (BGR) menjadi abu-abu."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, kernel=5):
    """
    Menghaluskan gambar untuk mengurangi noise.
    kernel harus angka ganjil (3, 5, 7).
    """
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def thresholding(blur):
    """
    Binerisasi: mengubah gambar jadi hitam-putih.
    Memakai Otsu agar nilai ambang dihitung otomatis.
    """
    _, biner = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return biner


def jalankan_preprocessing(frame):
    """
    Menggabungkan semua tahap pre-processing menjadi satu.
    Input  : frame berwarna dari kamera
    Output : (frame_resize, gambar_biner)
    """
    frame = resize_gambar(frame)
    gray = ke_grayscale(frame)
    blur = gaussian_blur(gray)
    biner = thresholding(blur)
    return frame, biner
