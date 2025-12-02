# src/01_regresi_dummy_basic.py

import pandas as pd
import statsmodels.formula.api as smf

def main():
    # 1. Load data
    df = pd.read_csv("../data/gaji_dummy.csv")

    print("=== 5 Data Pertama ===")
    print(df.head(), "\n")

    print("=== Distribusi Kategori ===")
    print(df[["gender", "education_level"]].value_counts(), "\n")

    # 2. Model regresi dengan variabel dummy (pakai formula)
    # salary_million ~ experience_years + gender + education_level
    model = smf.ols(
        "salary_million ~ experience_years + C(gender) + C(education_level)",
        data=df
    ).fit()

    print("=== Ringkasan Model Regresi Dummy (Dasar) ===")
    print(model.summary())

    # Catatan interpretasi singkat:
    print("\nCatatan:")
    print("- Intercept    : kategori baseline (misal: P & SMA) dengan experience_years = 0")
    print("- experience_years : tambahan gaji (juta) tiap 1 tahun pengalaman")
    print("- C(gender)[T.L]   : selisih gaji L terhadap P (baseline)")
    print("- C(education_level)[T.S1], [T.S2] : selisih terhadap SMA (baseline)")


if __name__ == "__main__":
    main()
