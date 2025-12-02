# src/03_regresi_dummy_uji_hipotesis.py

import pandas as pd
import statsmodels.api as sm


def build_model():
    # 1. Baca data
    df = pd.read_csv("../data/gaji_dummy.csv")

    print("=== Tipe data awal ===")
    print(df.dtypes, "\n")

    # 2. Pastikan kolom numerik benar-benar numerik
    numeric_cols = ["experience_years", "salary_million"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Jika ada NaN akibat gagal konversi, buang barisnya
    if df[numeric_cols].isna().sum().any():
        print("PERINGATAN: Ada nilai yang tidak bisa dikonversi ke angka.")
        print(df[numeric_cols].isna().sum())
        print("Baris bermasalah akan dibuang.\n")
        df = df.dropna(subset=numeric_cols)

    # 3. Buat dummy variabel kategori dan pastikan numerik
    dummies_gender = pd.get_dummies(
        df["gender"], prefix="gender", drop_first=True
    ).astype(float)

    dummies_edu = pd.get_dummies(
        df["education_level"], prefix="edu", drop_first=True
    ).astype(float)

    # 4. Susun matriks X
    X = pd.concat(
        [df[["experience_years"]], dummies_gender, dummies_edu],
        axis=1
    )

    # Tambahkan konstanta (intercept)
    X = sm.add_constant(X)

    # Paksa semua kolom X jadi float
    X = X.astype(float)

    # 5. Variabel dependen y
    y = df["salary_million"].astype(float)

    print("=== Kolom X dan tipenya setelah dibersihkan ===")
    print(X.dtypes, "\n")

    print("=== Nama parameter (kolom X) ===")
    print(list(X.columns), "\n")

    # 6. Bangun model
    model = sm.OLS(y, X).fit()
    return df, X, y, model


def main():
    df, X, y, model = build_model()

    print("=== Ringkasan Model ===")
    print(model.summary())

    # -----------------------------
    # Uji F (simultan)
    # -----------------------------
    print("\n=== Uji F (Simultan) ===")
    print(f"F-statistic    : {model.fvalue:.4f}")
    print(f"Prob(F-stat)   : {model.f_pvalue:.4f}")
    print("H0 : semua koefisien (kecuali intercept) = 0")
    print("H1 : minimal ada satu koefisien ≠ 0")

    if model.f_pvalue < 0.05:
        print("Kesimpulan     : Tolak H0 → model signifikan secara simultan (α = 5%)")
    else:
        print("Kesimpulan     : Gagal tolak H0 → model tidak signifikan (α = 5%)")

    # -----------------------------
    # Uji t (parsial) contoh
    # -----------------------------
    print("\n=== Uji t untuk masing-masing koefisien (contoh) ===")

    # 1) Pengalaman kerja
    print("\nKoefisien: experience_years")
    t_test_exp = model.t_test("experience_years = 0")
    print(t_test_exp)
    print("H0 : pengalaman tidak berpengaruh pada gaji")
    print("H1 : pengalaman berpengaruh pada gaji (α = 5%)")

    # 2) Gender laki-laki (dibanding baseline = perempuan), hanya kalau kolomnya ada
    if "gender_L" in X.columns:
        print("\nKoefisien: gender_L")
        t_test_gender = model.t_test("gender_L = 0")
        print(t_test_gender)
        print("H0 : tidak ada perbedaan gaji antara L dan P")
        print("H1 : ada perbedaan gaji antara L dan P (α = 5%)")
    else:
        print("\n[INFO] Kolom 'gender_L' tidak ditemukan di X, lewati uji t gender.")

    # -----------------------------
    # Uji F parsial untuk dummy pendidikan (edu_*)
    # -----------------------------
    kolom_uji = [c for c in X.columns if c.startswith("edu_")]

    if len(kolom_uji) > 0:
        print("\n=== Uji F Parsial: pengaruh pendidikan (edu_*) secara bersama ===")
        # Susun string seperti "edu_S1 = 0, edu_S2 = 0"
        hipotesis = ", ".join([f"{col} = 0" for col in kolom_uji])
        print("Hipotesis H0 :", hipotesis)

        f_test_edu = model.f_test(hipotesis)
        print(f_test_edu)

        print("H0 : semua koefisien pendidikan = 0 (tidak berpengaruh)")
        print("H1 : minimal satu koefisien pendidikan ≠ 0 (berpengaruh)")
    else:
        print("\n[INFO] Tidak ada kolom yang diawali 'edu_' di X, lewati uji F parsial pendidikan.")

    print("\nCatatan interpretasi umum:")
    print("- p-value < 0.05 → tolak H0 (ada pengaruh signifikan pada α = 5%)")
    print("- p-value ≥ 0.05 → gagal tolak H0 (tidak signifikan)")


if __name__ == "__main__":
    main()
