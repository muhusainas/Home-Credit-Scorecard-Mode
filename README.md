# Home Credit Default Risk — Prediksi Risiko Gagal Bayar

Analisis end-to-end untuk memprediksi risiko klien mengalami kesulitan pembayaran pinjaman, menggunakan dataset **Home Credit Default Risk**. Project ini mencakup keseluruhan alur data science: eksplorasi data, feature engineering dari data relasional, pemodelan machine learning, evaluasi, hingga penerjemahan hasil ke rekomendasi bisnis.

## Latar Belakang

Home Credit melayani nasabah yang minim atau tanpa riwayat kredit formal (*underbanked*). Tantangannya: memberi akses kredit yang adil, tanpa menanggung risiko gagal bayar yang tinggi.

**Goal:** Mengidentifikasi klien berisiko gagal bayar sedini mungkin, agar keputusan persetujuan pinjaman lebih tepat sasaran — tanpa menutup akses kredit bagi klien yang layak.

**Objective:** Membangun model *binary classification* yang memprediksi probabilitas klien mengalami kesulitan pembayaran (`TARGET = 1`), sebagai alat bantu keputusan (*decision support*) pada proses *underwriting*.

## Dataset

9 tabel relasional yang saling terhubung lewat `SK_ID_CURR` / `SK_ID_PREV` / `SK_ID_BUREAU`:

| Tabel | Baris | Deskripsi |
|---|---|---|
| `application_train/test.csv` | 307,511 / 48,744 | Tabel utama — satu baris = satu aplikasi pinjaman |
| `bureau.csv` | 1,716,428 | Riwayat kredit klien di lembaga keuangan lain |
| `bureau_balance.csv` | 27,299,925 | Saldo bulanan dari kredit di `bureau.csv` |
| `previous_application.csv` | 1,670,214 | Riwayat aplikasi pinjaman sebelumnya di Home Credit |
| `POS_CASH_balance.csv` | 10,001,358 | Saldo bulanan pinjaman POS/cash sebelumnya |
| `credit_card_balance.csv` | 3,840,312 | Saldo bulanan kartu kredit sebelumnya |
| `installments_payments.csv` | 13,605,401 | Riwayat pembayaran cicilan |

> Dataset tidak disertakan di repo ini  karena ukurannya besar. Unduh dari [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dan taruh di folder `data/`.

## Metodologi

1. **EDA** — distribusi target, deteksi anomali (`DAYS_EMPLOYED`), korelasi fitur, konsistensi antar tabel
2. **Data Cleaning & Feature Engineering** — penanganan anomali/outlier, agregasi 5 tabel riwayat ke level klien (`SK_ID_CURR`), menghasilkan 213 fitur
3. **Insight Extraction** — segmentasi risiko berdasarkan kombinasi sinyal (skor eksternal + perilaku pembayaran)
4. **Modeling** — Logistic Regression (dengan hyperparameter tuning), Random Forest, HistGradientBoosting
5. **Evaluasi** — AUC-ROC, PR-AUC, confusion matrix pada berbagai threshold, feature importance
6. **Business Impact** — analisis cost-benefit untuk menentukan threshold operasional optimal

## Hasil Utama

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Logistic Regression (tuned) | 0.7709 | 0.2557 |
| Random Forest | 0.7543 | 0.2300 |
| **HistGradientBoosting** | **0.7802** | **0.2762** |

**Insight kunci:** kombinasi sinyal risiko dari sumber berbeda (skor eksternal rendah + riwayat telat bayar) menghasilkan rate default **21.6%**, dibanding **4.7%** pada klien "relatif aman" — perbedaan 4.5x lipat.

**Dampak bisnis:** pada asumsi *Loss Given Default* 45% dan margin keuntungan 10%, threshold operasional **0.65** (bukan default 0.5) memberi dampak finansial terbaik — threshold yang terlalu agresif mengejar recall justru merugikan karena opportunity cost dari menolak klien baik.

## Struktur Repository

```
├── home_credit_default_risk.ipynb   # Notebook utama (end-to-end pipeline)
├── data/                             # Dataset CSV (tidak di-track git — unduh dari Kaggle)
├── output/                           # Hasil intermediate (model, dataset gabungan — tidak di-track git)
├── .gitignore
└── README.md
```

## Cara Menjalankan

1. Clone repository ini
2. Unduh dataset dari Kaggle, taruh semua file `.csv` di folder `data/`
3. Install dependency:
   ```
   pip install pandas numpy scikit-learn matplotlib joblib
   ```
4. Jalankan `home_credit_default_risk.ipynb` dari awal sampai akhir (Run All)

## Tech Stack

- Python (pandas, numpy, scikit-learn, matplotlib)
- Jupyter Notebook

## Rekomendasi Bisnis

1. Operasikan pada threshold **~0.65**, bukan 0.5
2. Terapkan sebagai sistem *tiering* (auto-approve / review manual / auto-reject), bukan keputusan biner
3. Gunakan **HistGradientBoosting** sebagai model utama; **Logistic Regression** sebagai pendamping untuk *explainability* & compliance

---
