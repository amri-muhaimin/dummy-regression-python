# src/02_regresi_dummy_asumsi_klasik.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt


def build_model():
    df = pd.read_csv("../data/gaji_dummy.csv")

    print("=== Tipe data awal ===")
    print(df.dtypes, "\n")

    # Pastikan kolom numerik benar-benar numerik
    numeric_cols = ["experience_years", "salary_million"]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Kalau ada yang NaN setelah konversi, kasih tahu
    if df[numeric_cols].isna().sum().any():
        print("PERINGATAN: Ada nilai yang tidak bisa dikonversi ke angka.")
        print(df[numeric_cols].isna().sum())
        print("Baris bermasalah akan dibuang.\n")
        df = df.dropna(subset=numeric_cols)

    # Buat dummy, lalu pastikan juga numerik (int)
    dummies_gender = pd.get_dummies(df["gender"], prefix="gender", drop_first=True).astype(int)
    dummies_edu = pd.get_dummies(df["education_level"], prefix="edu", drop_first=True).astype(int)

    X = pd.concat(
        [df[["experience_years"]], dummies_gender, dummies_edu],
        axis=1
    )

    # Tambah konstanta (intercept)
    X = sm.add_constant(X)

    # Paksa X dan y menjadi float supaya statsmodels tidak komplain
    X = X.astype(float)
    y = df["salary_million"].astype(float)

    print("=== Tipe data X setelah dibersihkan ===")
    print(X.dtypes, "\n")

    model = sm.OLS(y, X).fit()
    return df, X, y, model


def plot_diagnostics(model):
    residuals = model.resid
    fitted = model.fittedvalues

    # 1. Histogram residual
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=8)
    plt.title("Histogram Residual")
    plt.xlabel("Residual")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()

    # 2. QQ-plot residual
    sm.qqplot(residuals, line="45", fit=True)
    plt.title("QQ-Plot Residual")
    plt.tight_layout()
    plt.show()

    # 3. Residual vs Fitted
    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, residuals)
    plt.axhline(y=0, linestyle="--")
    plt.title("Residual vs Fitted")
    plt.xlabel("Nilai Fitted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


def test_normality(model):
    residuals = model.resid
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

    print("=== Uji Normalitas (Jarque-Bera) ===")
    print(f"Statistik JB  : {jb_stat:.4f}")
    print(f"p-value       : {jb_pvalue:.4f}")
    print(f"Skewness      : {skew:.4f}")
    print(f"Kurtosis      : {kurtosis:.4f}")

    if jb_pvalue > 0.05:
        print("Kesimpulan    : Gagal tolak H0 → residual ~ normal (α = 5%)")
    else:
        print("Kesimpulan    : Tolak H0 → residual tidak normal (α = 5%)")
    print()


def test_heteroskedasticity(model, X):
    residuals = model.resid

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X)

    print("=== Uji Heteroskedastisitas (Breusch-Pagan) ===")
    print(f"LM Statistic  : {lm_stat:.4f}")
    print(f"LM p-value    : {lm_pvalue:.4f}")
    print(f"F Statistic   : {f_stat:.4f}")
    print(f"F p-value     : {f_pvalue:.4f}")

    if lm_pvalue > 0.05:
        print("Kesimpulan    : Gagal tolak H0 → tidak ada indikasi heteroskedastisitas (α = 5%)")
    else:
        print("Kesimpulan    : Tolak H0 → ada indikasi heteroskedastisitas (α = 5%)")
    print()


def calculate_vif(X):
    # Buang konstanta saat hitung VIF
    X_no_const = X.drop(columns=["const"])
    vif_data = []

    for i, col in enumerate(X_no_const.columns):
        vif = variance_inflation_factor(X_no_const.values, i)
        vif_data.append((col, vif))

    print("=== Multikolinearitas (VIF) ===")
    for col, vif in vif_data:
        print(f"{col:15s} : VIF = {vif:.4f}")

    print("\nInterpretasi singkat:")
    print("- VIF ≈ 1       : tidak ada multikolinearitas")
    print("- VIF 1–5       : masih aman")
    print("- VIF > 10      : indikasi multikolinearitas tinggi")
    print()


def test_autocorrelation(model):
    dw = durbin_watson(model.resid)
    print("=== Uji Autokorelasi (Durbin-Watson) ===")
    print(f"Nilai DW       : {dw:.4f}")
    print("Rule of thumb: nilai mendekati 2 → tidak ada autokorelasi kuat.")
    print()


def main():
    df, X, y, model = build_model()

    print("=== 5 Data Pertama ===")
    print(df.head(), "\n")

    # Visualisasi asumsi
    plot_diagnostics(model)

    # Uji asumsi klasik
    test_normality(model)
    test_heteroskedasticity(model, X)
    calculate_vif(X)
    test_autocorrelation(model)


if __name__ == "__main__":
    main()
