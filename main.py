import sys
import cam
import preprocess
import segmentation
import feature_extract
import classifier
import helper


def mode_training():
    """Melatih model KNN dari dataset CSV."""
    X, y = helper.baca_dataset()
    if len(X) == 0:
        print("Dataset kosong. Isi dulu data/dataset.csv sebelum training.")
        return
    classifier.latih_model(X, y)
    print(f"Model berhasil dilatih dari {len(X)} data. Disimpan di models/model.pkl")


def mode_realtime():
    """Menjalankan deteksi nominal uang secara realtime dari webcam."""
    import cv2
    from collections import deque, Counter

    cap = cam.buka_kamera(0)
    model = classifier.muat_model()  # None kalau belum ada -> pakai mode dummy

    if model is None:
        print("Model belum ada. Memakai mode tebakan warna (dummy).")
        print("Latih model dulu dengan: python main.py train")

    print("Tekan 'q' untuk keluar.")

    # Menyimpan beberapa tebakan terakhir, lalu menampilkan yang PALING SERING
    # muncul (voting). Tujuannya supaya hasil tidak kedip-kedip tiap frame.
    riwayat = deque(maxlen=15)

    while True:
        # 1. WEBCAM
        berhasil, frame = cam.ambil_frame(cap)
        if not berhasil:
            break

        # 2. PREPROCESSING
        frame, biner = preprocess.jalankan_preprocessing(frame)

        # 3. SEGMENTATION
        crop, kotak = segmentation.jalankan_segmentasi(frame, biner)

        # 4. FEATURE EXTRACTION
        vektor = feature_extract.buat_feature_vector(crop)

        # 5. CLASSIFICATION
        hasil = classifier.prediksi(vektor, model)

        # haluskan hasil dari 15 frame terakhir
        if vektor is not None:
            riwayat.append(hasil)

        if len(riwayat) > 0:
            # nominal yang paling sering muncul belakangan ini
            hasil_stabil = Counter(riwayat).most_common(1)[0][0]
        else:
            hasil_stabil = "Arahkan uang ke kamera"

        # 6. OUTPUT
        frame = helper.gambar_kotak(frame, kotak)
        frame = helper.tulis_teks(frame, f"Nominal: {hasil_stabil}")
        cv2.imshow("Deteksi Nominal Rupiah", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.tutup_kamera(cap)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        mode_training()
    else:
        mode_realtime()