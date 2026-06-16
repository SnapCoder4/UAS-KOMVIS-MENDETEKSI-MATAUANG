import os
import pickle
import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_ADA = True
except ImportError:
    SKLEARN_ADA = False

MODEL_PATH = os.path.join("models", "model.pkl")


def latih_model(X, y, k=3):
    """
    Melatih model KNN.
    X = daftar feature vector, y = daftar label nominal.
    """
    if not SKLEARN_ADA:
        raise RuntimeError("scikit-learn belum terpasang. Jalankan: pip install scikit-learn")

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def muat_model():
    "Memuat model terlatih dari file. None kalau belum ada."
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _prediksi_dummy(vektor):
    """
    Mode cadangan tanpa model terlatih.
    Menebak nominal berdasarkan warna dominan (Hue).
    Hanya untuk demonstrasi pipeline, bukan hasil akurat.
    """
    hue = vektor[0]  # mean Hue dari feature_extract
    if hue < 15 or hue >= 170:
        return "Rp100.000 (merah?)"
    elif hue < 35:
        return "Rp5.000 (cokelat?)"
    elif hue < 85:
        return "Rp20.000 (hijau?)"
    elif hue < 130:
        return "Rp50.000 (biru?)"
    else:
        return "Rp10.000 (ungu?)"


def prediksi(vektor, model=None):
    """
    Memprediksi nominal uang dari feature vector.
    Kalau model tersedia -> pakai KNN.
    Kalau tidak -> pakai aturan warna sederhana (dummy).
    """
    if vektor is None:
        return "Tidak ada objek"

    if model is not None:
        hasil = model.predict([np.array(vektor)])
        return str(hasil[0])

    return _prediksi_dummy(vektor)
