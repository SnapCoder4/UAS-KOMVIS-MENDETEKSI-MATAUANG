import os
import csv
import cv2

import feature_extract

# folder berisi subfolder per nominal
FOLDER_GAMBAR = os.path.join("data", "dataset")
FILE_OUTPUT = os.path.join("data", "dataset.csv")

# batas jumlah gambar yang diambil per nominal 
MAKS_PER_NOMINAL = None

# ekstensi gambar yang diterima
EKSTENSI = (".jpg", ".jpeg", ".png", ".bmp")


def daftar_gambar(folder):
    "Mengembalikan daftar nama file gambar di sebuah folder"
    semua = [f for f in os.listdir(folder) if f.lower().endswith(EKSTENSI)]
    semua.sort()
    if MAKS_PER_NOMINAL is not None:
        semua = semua[:MAKS_PER_NOMINAL]
    return semua


def main():
    if not os.path.isdir(FOLDER_GAMBAR):
        print(f"Folder '{FOLDER_GAMBAR}' tidak ditemukan.")
        print("Pastikan hasil ekstrak ada di data/dataset/<nominal>/")
        return

    # ambil daftar subfolder nominal
    nominal_list = [
        d for d in os.listdir(FOLDER_GAMBAR)
        if os.path.isdir(os.path.join(FOLDER_GAMBAR, d))
    ]
    nominal_list.sort()

    if len(nominal_list) == 0:
        print("Tidak ada subfolder nominal di dalam data/dataset/")
        return

    baris_terkumpul = []
    total = 0
    gagal = 0

    for nominal in nominal_list:
        folder = os.path.join(FOLDER_GAMBAR, nominal)
        files = daftar_gambar(folder)
        terpakai = 0

        for nama in files:
            path = os.path.join(folder, nama)
            gambar = cv2.imread(path)
            if gambar is None:
                gagal += 1
                continue

            # ubah gambar jadi feature vector (modul Anggota 2)
            vektor = feature_extract.buat_feature_vector(gambar)
            if vektor is None:
                gagal += 1
                continue

            # baris CSV: fitur..., label(nominal)
            baris_terkumpul.append(vektor + [nominal])
            terpakai += 1
            total += 1

        print(f"  {nominal:>8} : {terpakai} gambar diproses")

    # tulis ke CSV
    os.makedirs("data", exist_ok=True)
    with open(FILE_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(baris_terkumpul)

    print("-" * 40)
    print(f"Total {total} data ditulis ke {FILE_OUTPUT}")
    if gagal:
        print(f"({gagal} gambar dilewati karena gagal dibaca)")
    print("Selanjutnya jalankan: python main.py train")


if __name__ == "__main__":
    main()