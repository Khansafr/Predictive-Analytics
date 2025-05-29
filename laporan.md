# Laporan Proyek: Prediksi Harga Apartemen di Jakarta

## 1. Domain Proyek

Permasalahan harga properti, khususnya apartemen di Jakarta, merupakan isu penting yang berdampak langsung terhadap keputusan investasi, pembelian hunian, dan kebijakan perumahan. Dengan populasi yang terus meningkat dan keterbatasan lahan, harga apartemen di Jakarta cenderung mengalami fluktuasi yang signifikan. Oleh karena itu, prediksi harga apartemen menjadi penting agar dapat memberikan insight kepada calon pembeli, investor, dan pengembang properti.

Masalah ini perlu diselesaikan dengan pendekatan berbasis data dan pemodelan machine learning karena:

* Diperlukan metode prediksi yang andal untuk memperkirakan harga berdasarkan faktor-faktor kompleks seperti lokasi, luas bangunan, dan fasilitas.
* Dapat membantu berbagai pihak untuk mengambil keputusan strategis secara objektif.

**Referensi**:

* Susanto, A., et al. (2023). "Property Price Forecasting Using Machine Learning: A Case Study in Jakarta." *International Journal of Data Science*, 8(2), 120–132.
* BPS DKI Jakarta (2024). Statistik Harga Properti Residensial.

## 2. Business Understanding

### Problem Statement

Bagaimana cara memprediksi harga apartemen di Jakarta berdasarkan fitur-fitur seperti lokasi, luas, jumlah kamar tidur, dan hak guna lahan?

### Goals

Membangun model prediksi harga apartemen di Jakarta yang akurat dan dapat digunakan untuk mendukung pengambilan keputusan.

### Solution Statement

Untuk mencapai tujuan tersebut, solusi yang ditawarkan mencakup:

1. **Baseline Model**: Membangun model prediksi harga menggunakan tiga algoritma:

   * Random Forest
   * XGBoost
   * Decision Tree

2. **Model Improvement**:

   * Melakukan **Hyperparameter Tuning** terhadap masing-masing model untuk meningkatkan akurasi prediksi.

Model dievaluasi menggunakan metrik evaluasi yang tepat, yaitu **R2, MAE, MSE, dan RMSE**.

## 3. Data Understanding

### Sumber Data

* Dataset diperoleh dari hasil scraping situs properti *Pinhome.id* yang berisi informasi apartemen di Jakarta.
* [Link Dataset](https://drive.google.com/file/d/1VUa17MfU3vf9sMmxc23D0kcpQD45mCUs/view?usp=sharing)

### Informasi Data

* Jumlah baris awal: 3.993
* Kolom-kolom utama: `Harga`, `Luas Bangunan`, `Kamar Tidur`, `Kelurahan`, `Kecamatan`, `Kota`, `Hak Guna`.

### Deskripsi Fitur

| Fitur         | Deskripsi                                     |
| ------------- | --------------------------------------------- |
| Harga         | Harga apartemen (dalam format string rupiah)  |
| Luas Bangunan | Luas bangunan apartemen                       |
| Kamar Tidur   | Jumlah kamar tidur                            |
| Kelurahan     | Nama kelurahan apartemen                      |
| Kecamatan     | Nama kecamatan apartemen                      |
| Kota          | Nama kota/kabupaten apartemen                 |
| Hak Guna      | Hak kepemilikan (HGB, SHM, Strata Title, dll) |

### Eksplorasi Awal

* Beberapa baris memiliki missing value.
* Kolom harga perlu dibersihkan dari satuan `Rp`, `Jt`, dan `M`.
* Kolom `Kecamatan` menyertakan data redundan dengan kata `Kota`.

### Visualisasi

#### Gambar 1: Scatter Plot Fitur vs Log Harga
![Scatter Plot](https://drive.google.com/uc?export=view&id=1-Djta6hl_kvasmn5zCa1sFAb7uayKE0L)

#### Gambar 2: Heatmap Korelasi
![Heatmap Korelasi](https://drive.google.com/uc?export=view&id=1QU6iXnfphjy-wKIw4I4wN-PSRjucfaji)


## 4. Data Preparation

### Teknik dan Proses:

* **Pembersihan Missing Value**: Drop baris dengan missing value.
* **Cleaning Format Harga**: Konversi harga dari format string ke float menggunakan konversi `Rp`, `Jt`, dan `M`.
* **Ekstraksi Angka** dari kolom `Luas Bangunan` dan `Kamar Tidur`.
* **Estimasi Harga per m2**: `Harga / Luas Bangunan`
* **Transformasi Logaritmik**: `Log Harga` sebagai target untuk mengurangi skewness.
* **Encoding Kategori**: Menggunakan **Frequency Encoding** untuk `Kelurahan`, `Kecamatan`, `Kota`, dan `Hak Guna`.
* **Standardisasi**: `Luas` dan `Kamar Tidur` di-scaling menggunakan `StandardScaler`.

### Alasan Tahapan

* Mengatasi perbedaan skala antar fitur.
* Menangani data kategorikal tanpa meningkatkan dimensionalitas.
* Menyesuaikan distribusi target agar lebih normal.

## 5. Modeling

### Algoritma yang Digunakan

* **Random Forest**: Model ensemble berbasis pohon keputusan.
* **XGBoost**: Gradient boosting dengan kemampuan penyesuaian regularisasi.
* **Decision Tree**: Model pohon dasar sebagai baseline sederhana.

### Parameter Baseline:

* Random Forest: `n_estimators=150, max_depth=20`
* XGBoost: `n_estimators=100, max_depth=7, learning_rate=0.1`
* Decision Tree: `max_depth=10`

### Hyperparameter Tuning:

* GridSearchCV dengan cross-validation (`cv=3`) dilakukan pada setiap model.
* Parameter disesuaikan untuk menemukan konfigurasi optimal.

### Kelebihan dan Kekurangan:

| Model         | Kelebihan                          | Kekurangan                                  |
| ------------- | ---------------------------------- | ------------------------------------------- |
| Random Forest | Tahan terhadap overfitting, robust | Training time lebih lama                    |
| XGBoost       | Akurat dan efisien untuk boosting  | Kompleks dan lebih sensitif terhadap tuning |
| Decision Tree | Sederhana, interpretatif           | Mudah overfitting                           |


## 6. Evaluation

### Metrik Evaluasi:

* **R² Score**: Mengukur proporsi variansi yang dapat dijelaskan oleh model.
  $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

### Hasil Evaluasi Model (Baseline vs Tuned)

| Model                  | R² Score | MSE     | RMSE    | MAE     |
|------------------------|----------|---------|---------|---------|
| **Baseline Models**    |          |         |         |         |
| XGBoost                | 0.9923   | 0.0280  | 0.1672  | 0.0870  |
| Random Forest          | 0.9935   | 0.0236  | 0.1537  | 0.0368  |
| Decision Tree          | 0.9900   | 0.0364  | 0.1908  | 0.0694  |
| **Tuned Models**       |          |         |         |         |
| Random Forest Tuned    | **0.9941** | **0.0215** | **0.1468** | 0.0410  |
| XGBoost Tuned          | 0.9928   | 0.0260  | 0.1612  | 0.0622  |
| Decision Tree Tuned    | 0.9896   | 0.0377  | 0.1941  | 0.0540  |

### Model Terbaik
- **Random Forest Tuned** menjadi model terbaik berdasarkan skor **R² = 0.9941**, serta nilai MSE dan RMSE yang paling rendah.
- Model ini menunjukkan kinerja prediktif paling tinggi dalam memprediksi harga apartemen di Jakarta.

#### Gambar 3: Perbandingan Harga Aktual vs Prediksi (Random Forest Tuned)
![Actual vs Predicted](https://drive.google.com/uc?export=view&id=1PKL8gjSYwSv2ahWRkW7Tma2Zcflgj5tK)

#### Gambar 4: Distribusi Persentase Error Prediksi
![Percentage Error](https://drive.google.com/uc?export=view&id=1hwxQd9brPgoDofaYv91iKR8qIrclZ8Zj)

---

**Catatan**: Notebook ini mendukung reproduksibilitas penuh dengan menyimpan model terbaik (`joblib`) dan scaler yang digunakan.
