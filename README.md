# Used Car Price Prediction Engine

[![Link Streamlit App](https://static.streamlit.io/badge_gradient.svg)](https://used-car-price-prediction-rosi.streamlit.app/)

Aplikasi berbasis web untuk mengestimasi harga jual wajar mobil bekas menggunakan algoritma *Ensemble Learning*. Proyek ini dikembangkan sebagai *Final Project* Bootcamp Data Science di Digital Skola.

---

## Business Understanding & Objective
Proses penentuan harga mobil bekas sering kali bersifat subjektif dan bervariasi. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga kendaraan secara akurat berdasarkan tren data pasar historis, membantu dealer maupun pembeli perorangan mendapatkan estimasi harga yang adil.

## Dataset Overview
* **Sumber Data:** Kaggle Playground Series S4E9
* **Ukuran Data:** 188.000+ baris data transaksi
* **Fitur Utama:** Merek, Model, Tahun Produksi, Jarak Tempuh (*Milage*), Jenis Bahan Bakar, Spesifikasi Mesin, Transmisi, Riwayat Kecelakaan, dan Status Keaslian Surat (*Clean Title*).

## Tech Stack & Library
* **Language:** Python 3.11
* **Framework Web:** Streamlit
* **Machine Learning:** CatBoost Regressor
* **Data Processing:** Pandas, NumPy, Scikit-Learn

## Model Performance
Model dievaluasi menggunakan metrik statistik standar industri setelah melalui tahap *data pruning* (pembersihan noise ekstrem) dan *hyperparameter tuning*:
* **$R^2$ Score:** `0.934` (Model mampu menjelaskan 93.4% variasi harga di pasar)
* **Status:** Siap digunakan untuk Deployment skala besar.

## Cara Menjalankan Secara Lokal
1. Clone repositori ini
2. Instal library: `pip install -r requirements.txt`
3. Jalankan diterminal: `streamlit run app.py`

---
*Dibuat oleh [Rosi Rahmawati](https://www.linkedin.com/in/rosi-rahmawati/)*
