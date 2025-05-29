# Laporan Proyek: Prediksi Harga Apartemen di Jakarta

## Domain Proyek

Harga apartemen di Jakarta merupakan indikator penting dalam sektor properti yang dipengaruhi oleh pertumbuhan populasi, keterbatasan lahan, dan dinamika pasar. Kebutuhan akan prediksi harga yang akurat menjadi krusial bagi pembeli, investor, dan pembuat kebijakan guna mendukung keputusan berbasis data.

Pendekatan machine learning dinilai efektif untuk memodelkan harga berdasarkan berbagai atribut seperti lokasi, luas bangunan, dan fasilitas (Susanto et al., 2023). Dengan memanfaatkan data historis dan statistik dari BPS DKI Jakarta (2024), model prediktif dapat dikembangkan untuk meningkatkan transparansi dan efisiensi pasar properti.

## Business Understanding

Harga apartemen di Jakarta sulit diprediksi karena dipengaruhi banyak faktor, sehingga menyulitkan pembeli dan investor dalam menilai harga yang wajar. Untuk itu, proyek ini mengembangkan model machine learning yang dapat memperkirakan harga apartemen secara akurat berdasarkan fitur properti.

### Problem Statement

* Bagaimana memprediksi harga apartemen secara akurat?
* Algoritma mana yang paling optimal untuk prediksi harga?
* Apakah tuning dapat meningkatkan performa model?

### Goals

* Membangun model prediksi harga apartemen berdasarkan fitur properti.
* Menentukan algoritma terbaik dengan evaluasi metrik.
* Meningkatkan akurasi model dengan tuning.

### Solution Statements

* Menerapkan tiga algoritma machine learning: **Decision Tree Regressor**, **Random Forest Regressor**, dan **Gradient Boosting Regressor**.
* Melakukan **hyperparameter tuning** dengan **GridSearchCV** untuk meningkatkan akurasi model.
* Mengevaluasi model menggunakan metrik yang terukur, yaitu **MAE**, **RMSE**, dan **R² Score** pada data uji.
* Memilih model dengan performa terbaik untuk diimplementasikan dalam sistem prediksi harga apartemen di Jakarta.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini diperoleh melalui proses web scraping dari situs Pinhome.id [https://www.pinhome.id/]. Data yang difokuskan hanya pada apartemen di wilayah Jakarta karena memiliki volume data yang cukup besar dan relevansi tinggi terhadap kebutuhan prediksi harga properti di kawasan urban. Data mencakup fitur seperti harga, lokasi (kelurahan, kecamatan, kota), luas bangunan, jumlah kamar tidur, status kepemilikan (Hak Guna), dan estimasi harga per meter persegi.

Dataset yang digunakan dalam penelitian ini merupakan hasil koleksi pribadi dan dapat dilihat melalui tautan berikut:  
[Lihat Dataset di GitHub](https://github.com/Khansafr/Predictive-Analytics/blob/main/dataset_apartemen_pinhome.csv)



### Deskripsi Dataset

| Fitur          | Tipe Data | Keterangan                                                                                          |
|----------------|-----------|---------------------------------------------------------------------------------------------------|
| Harga          | Object    | Harga apartemen dalam format rentang (contoh: "Rp1,2 M - Rp1,5 M")                                |
| Kelurahan      | Object    | Nama kelurahan lokasi apartemen                                                                    |
| Kecamatan      | Object    | Nama kecamatan lokasi apartemen                                                                    |
| Kota           | Object    | Nama kota administratif, terdapat beberapa missing value                                           |
| Kamar Tidur    | Object    | Jumlah kamar tidur, ada beberapa missing value dan data tidak konsisten                            |
| Luas Bangunan  | Object    | Luas bangunan dalam format rentang dengan teks (misal "35m²"), perlu pembersihan                    |
| Hak Guna       | Object    | Status kepemilikan (contoh: Hak Milik), terdapat missing value cukup tinggi (~39%)                 |
| Harga per m2   | Object    | Harga per meter persegi dalam format rentang, perlu konversi ke numerik                            |

Dataset terdiri dari **3993 baris** dan **8 kolom**, semuanya bertipe teks (object). Beberapa kolom seperti `Kota`, `Kamar Tidur`, `Luas Bangunan`, dan `Hak Guna` memiliki data yang hilang (missing values).

### Statistik Deskriptif dan Insight Awal

- Harga: Terdapat 386 variasi harga, dengan harga paling sering adalah Rp1,2 M (154 unit).
- Kelurahan & Kecamatan: Meliputi 204 kelurahan dan 45 kecamatan, dengan Kelurahan Tanjung Duren Selatan dan Kecamatan Setiabudi paling dominan.
- Kota: Terdapat 5 kota administratif, dengan Jakarta Selatan terbanyak (1266 unit).
- Kamar Tidur: Variasi mencapai 55 kategori, 2 kamar tidur paling umum (1911 unit).
- Luas Bangunan: Terdapat 284 nilai unik, 35m² paling sering muncul (173 unit).
- Hak Guna: Mayoritas apartemen memiliki Sertifikat Hak Milik (1092 unit dari 2421 data yang tersedia).
- Harga per m²: Memiliki 572 nilai unik, nilai Rp16,7 Jt/m² paling umum (71 unit).

### Missing Values

- Kolom `Kota`: sekitar 16% data kosong.
- Kolom `Hak Guna`: sekitar 39% missing.
- Kolom `Kamar Tidur` dan `Luas Bangunan`: missing value sedikit.

### Tantangan Data

- Kolom `Harga`, `Luas Bangunan`, dan `Harga per m2` dalam format rentang dan teks, perlu ekstraksi nilai minimum dan maksimum serta konversi ke numerik.
- Beberapa kolom lokasi (`Kelurahan`, `Kecamatan`, dan `Kota`) terdapat ketidakkonsistenan dan pencampuran data antar kolom.
- Fitur `Kamar Tidur` mengandung nilai tidak valid dan campuran data, perlu pembersihan.
- Fitur `Hak Guna` memiliki missing data cukup banyak.
- Semua fitur masih dalam format string yang memerlukan preprocessing agar siap dipakai dalam modeling.
  
### Univariate Analysis

Visualisasi distribusi tiap fitur (Top 50 kategori):
![Univariate](https://github.com/user-attachments/assets/2e208e7e-d76e-4407-adc3-42d10a32ff4d)

**Insight:**

- **Harga & Harga per m2**  
  Data masih berupa teks dengan satuan “M” (Miliar) dan “Jt” (Juta). Perlu dilakukan konversi ke format numerik agar dapat dianalisis secara kuantitatif.

- **Kelurahan**  
  Terdapat banyak kesalahan data berupa pencampuran nama kelurahan dengan nama kecamatan, menyebabkan ketidakkonsistenan dan kesalahan penempatan lokasi.

- **Kecamatan**  
  Beberapa data di kolom ini sebenarnya adalah nama kota, bukan kecamatan, sehingga terdapat pergeseran data antar kolom.

- **Kota**  
  Ada kemungkinan data kota tercampur ke kolom kecamatan, yang harus diperbaiki untuk menjaga konsistensi.

- **Kamar Tidur**  
  Nilai unik sangat banyak dan terdapat kesalahan data, misalnya tercampurnya data luas bangunan di sini, menandakan adanya kesalahan input yang harus dibersihkan.

- **Luas Bangunan**  
  Data berformat campuran antara angka dan teks (contoh: “Luas Bangunan 35m”), perlu pembersihan agar menjadi nilai numerik yang valid.

- **Hak Guna**  
  Tipe hak guna relatif konsisten dengan empat kategori utama.

---
## Data Preparation

Pada tahap data preparation, dilakukan serangkaian proses pembersihan dan transformasi data agar dataset siap digunakan untuk pelatihan model prediksi harga apartemen. Proses ini penting untuk memastikan kualitas data yang baik dan meningkatkan performa model. Berikut adalah tahapan yang dilakukan secara berurutan beserta alasan dan penjelasannya:

### Filter Baris Valid

Fungsi `filter_baris_valid()` diterapkan untuk menyaring baris data yang memenuhi kriteria valid pada setiap kolom utama, yaitu:

- **Harga** harus berupa string yang mengandung "Rp" sebagai tanda format harga.
- **Kelurahan** harus berupa string yang diawali dengan "Kel.".
- **Kecamatan** harus berupa string yang diawali dengan "Kec.".
- **Kota** harus berupa string yang diawali dengan "Kota".
- **Kamar Tidur** harus mengandung teks "Kamar Tidur".
- **Luas Bangunan** harus mengandung teks "Luas Bangunan" dan satuan "m".
- **Hak Guna** harus berupa string yang mengandung salah satu kata kunci legalitas seperti "hak milik", "hak guna", "girik", atau "strata".
- **Harga per m2** harus berupa string yang mengandung "Rp" dan "/m²".

**Alasan:**  
Memastikan hanya data yang benar dan konsisten yang digunakan agar tidak terjadi kesalahan atau bias pada model akibat data yang tidak valid, misalnya nilai kosong, format salah, atau entri campur aduk.

### Ekstraksi dan Konversi Fitur Numerik

Data asli mengandung informasi numerik yang dikemas dalam format string campuran teks dan angka (misal: "Rp 1,5 M", "3 Kamar Tidur", "100 m2"). Untuk dapat diproses oleh model, dilakukan transformasi:

- **`clean_rupiah(value)`**  
  Membersihkan dan mengkonversi harga dari format teks yang mengandung simbol rupiah ("Rp"), jutaan ("Jt"), dan milyaran ("M") menjadi angka numerik (tipe float). Fungsi juga dapat menangani rentang harga seperti "1-2 M" dengan mengambil rata-rata.

- **`extract_m2(value)`**  
  Mengekstrak nilai numerik luas bangunan (dalam meter persegi) dari string yang mengandung angka dan teks satuan.

- **`extract_kamar_tidur(value)`**  
  Mengekstrak jumlah kamar tidur sebagai angka dari string deskriptif.

**Alasan:**  
Model machine learning hanya bisa mengolah data numerik untuk fitur numerik. Oleh karena itu, ekstraksi ini penting agar fitur dapat dimasukkan ke dalam model secara tepat.

### Filtering Logis Tambahan

Data dengan jumlah kamar tidur yang sangat besar (>10 kamar) dianggap tidak wajar dan kemungkinan adalah outlier atau kesalahan input data. Baris dengan kondisi ini dihapus.

**Alasan:**  
Outlier yang ekstrim dapat mempengaruhi kestabilan dan akurasi model, sehingga perlu dihilangkan untuk menjaga kualitas data.

### Imputasi Nilai Hilang (Missing Values)

Beberapa kolom memiliki nilai kosong (missing values) yang perlu diisi agar data lengkap dan tidak menimbulkan error saat pemodelan:

- Untuk kolom kategorikal seperti `Kota` dan `Hak Guna`, nilai kosong diisi dengan **modus** (nilai yang paling sering muncul) dari kolom tersebut.
- Untuk kolom numerik seperti `Luas (m2)` dan `Kamar Tidur (n)`, nilai kosong diisi dengan **median** (nilai tengah) dari kolom tersebut.

**Alasan:**  
Mengisi missing value dengan nilai statistik yang representatif dapat mempertahankan distribusi data dan menghindari kehilangan informasi yang penting.

### Frequency Encoding untuk Fitur Kategorikal

Fitur kategorikal seperti `Kelurahan`, `Kecamatan`, `Kota`, dan `Hak Guna` diubah menjadi nilai numerik menggunakan **frequency encoding**, yaitu mengganti setiap kategori dengan proporsi frekuensi kemunculannya di dataset.

**Alasan:**  
Model machine learning lebih mudah memproses data numerik. Frequency encoding membantu mempertahankan informasi distribusi kategori tanpa membuat fitur menjadi sangat banyak (seperti pada one-hot encoding). Metode ini juga membantu mengurangi risiko overfitting terutama pada fitur dengan banyak kategori.

### Perhitungan Fitur Tambahan: Harga per Meter Persegi

Fitur baru `Harga per m2 (estimasi)` dihitung dengan membagi `Harga (Rp)` dengan `Luas (m2)`.

**Alasan:**  
Harga per meter persegi adalah indikator harga properti yang umum digunakan dan membantu model memahami harga relatif apartemen berdasarkan luasnya.

---

## Visualisasi Data

Setelah proses data preparation selesai, dilakukan visualisasi untuk mengeksplorasi sebaran data dan mendapatkan insight awal dari fitur-fitur numerik maupun kategorikal.

### Visualisasi Distribusi Fitur Numerik

![Distribusi Numerik](https://github.com/user-attachments/assets/05766487-6ac8-4959-9bca-ca74f7280b1d)

**Insight:**

- **Luas (m²):** Mayoritas apartemen memiliki luas di bawah 200 m². Distribusi sangat tidak merata, dengan beberapa unit memiliki luas ekstrem hingga lebih dari 8.000 m².
- **Kamar Tidur (n):** Jumlah kamar tidur paling umum adalah 2, diikuti oleh 1 dan 3 kamar. Jumlah unit dengan lebih dari 4 kamar sangat sedikit.
- **Harga per m² (estimasi):** Didominasi oleh harga per m² di bawah 300 juta rupiah. Terdapat nilai ekstrem hingga lebih dari 2,5 miliar per m².
- **Harga (Rp):** Sebagian besar apartemen memiliki harga total di bawah 1 miliar rupiah. Distribusi sangat tidak merata, dengan harga tertinggi mencapai hampir 30 miliar rupiah.

### Visualisasi Distribusi Fitur Kategorikal

![Distribusi Kategorikal](https://github.com/user-attachments/assets/46ab09ee-1f28-4b3a-b662-bde745552a73)

**Insight:**

- **Kelurahan:** Kelurahan Cipinang Besar Selatan dan Karet Kuningan mendominasi jumlah listing. Terjadi konsentrasi data pada beberapa kelurahan saja.
- **Kecamatan:** Setiabudi merupakan kecamatan dengan jumlah listing apartemen terbanyak. Dominasi Jakarta Selatan terlihat dari frekuensi tinggi kecamatan seperti Pancoran dan Kebayoran Baru.
- **Kota:** Jakarta Selatan menjadi kota dengan listing terbanyak, hampir dua kali lipat dari Jakarta Utara. Jakarta Barat tercatat memiliki jumlah listing paling sedikit.
- **Hak Guna:** Sertifikat Hak Milik (SHM) merupakan tipe hak guna yang paling umum. Tipe seperti Girik sangat jarang muncul dalam dataset.

### Korelasi antar Fitur Numerik

![Heatmap Korelasi](https://github.com/user-attachments/assets/fa0a72ef-f568-4815-8d6a-5d3be8d50740)

**Insight:**

- **Harga per m²** memiliki korelasi tertinggi terhadap harga total (0.72), menjadikannya faktor penentu utama.
- **Jumlah kamar tidur** menunjukkan korelasi sedang (0.41) terhadap harga.
- **Luas bangunan** memiliki korelasi rendah (0.21), menunjukkan bahwa luas saja tidak cukup kuat menjelaskan harga tanpa mempertimbangkan lokasi atau harga per meter persegi.

### Fitur & Target

Pada tahap ini, dilakukan pemilihan fitur-fitur yang dianggap relevan untuk memprediksi harga apartemen (target). Fitur yang digunakan meliputi:

- **Luas (m2)**: Luas bangunan dalam meter persegi.
- **Kamar Tidur (n)**: Jumlah kamar tidur.
- **Harga per m2 (estimasi)**: Estimasi harga per meter persegi, dihitung dari harga total dibagi luas bangunan.
- **Kelurahan_Freq**: Frequency encoding kategori kelurahan.
- **Kecamatan_Freq**: Frequency encoding kategori kecamatan.
- **Kota_Freq**: Frequency encoding kategori kota.
- **Hak Guna_Freq**: Frequency encoding status hak guna tanah.

Variabel target yang ingin diprediksi adalah:

- **Harga (Rp)**: Harga total apartemen dalam satuan Rupiah.

```python
features = [
    'Luas (m2)', 'Kamar Tidur (n)', 'Harga per m2 (estimasi)',
    'Kelurahan_Freq', 'Kecamatan_Freq', 'Kota_Freq', 'Hak Guna_Freq'
]
target = 'Harga (Rp)'

X = df[features]
y = df[target]
```

---

### Data Splitting

Setelah fitur (`X`) dan target (`y`) dipilih, dataset dibagi menjadi dua subset yaitu data latih dan data uji dengan proporsi 80:20 menggunakan fungsi `train_test_split` dari library `sklearn.model_selection`. Parameter `random_state=42` digunakan agar pembagian data bersifat reproducible.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Modeling

### Tujuan Pemodelan

- Membangun model machine learning yang dapat memprediksi harga apartemen secara akurat berdasarkan fitur-fitur input.
- Membandingkan beberapa algoritma regresi untuk memilih model terbaik.
- Melakukan tuning parameter untuk meningkatkan performa model terpilih.

### Algoritma yang Digunakan

Tiga algoritma regresi yang dipilih untuk proses pemodelan adalah:

1. **Decision Tree Regressor**  
   Algoritma ini membangun model berbentuk pohon keputusan berdasarkan fitur yang paling mengurangi impurity pada setiap split.

2. **Random Forest Regressor**  
   Merupakan ensemble dari banyak pohon keputusan, bekerja dengan melakukan bootstrap sampling dan agregasi (bagging) untuk meningkatkan generalisasi.

3. **Gradient Boosting Regressor**  
   Teknik boosting yang membangun model secara iteratif, di mana model baru memperbaiki kesalahan dari model sebelumnya. Memiliki performa sangat baik dalam berbagai kasus regresi.

### Proses Pemodelan

1. **Pembagian Data**  
   Dataset dibagi menjadi **data latih (80%)** dan **data uji (20%)** menggunakan `train_test_split` untuk menghindari overfitting dan mengukur generalisasi model.

2. **Baseline Model**  
   Masing-masing algoritma dilatih terlebih dahulu menggunakan parameter default. Hasil baseline ini digunakan sebagai acuan untuk proses tuning berikutnya.

3. **Hyperparameter Tuning**  
   Model kemudian dioptimasi menggunakan **RandomizedSearchCV** dengan 5-fold cross-validation untuk mencari kombinasi parameter terbaik berdasarkan performa terhadap data validasi.

   - **Decision Tree Regressor**
     - `max_depth`: 5 – 50
     - `min_samples_split`: 2 – 10
     - `min_samples_leaf`: 1 – 10

   - **Random Forest Regressor**
     - `n_estimators`: 100 – 1000
     - `max_depth`: 10 – 100
     - `min_samples_split`: 2 – 10
     - `min_samples_leaf`: 1 – 10
     - `bootstrap`: True / False

   - **Gradient Boosting Regressor**
     - `n_estimators`: 100 – 1000
     - `learning_rate`: 0.01 – 0.2
     - `max_depth`: 3 – 10
     - `subsample`: 0.5 – 1.0
     - `min_samples_split`: 2 – 10
     - `min_samples_leaf`: 1 – 10

### Kelebihan dan Kekurangan Algoritma

| Algoritma           | Kelebihan                              | Kekurangan                            |
|---------------------|---------------------------------------|-------------------------------------|
| Decision Tree       | Interpretasi mudah, cepat dilatih     | Overfitting pada data kecil, tidak stabil |
| Random Forest       | Mengurangi overfitting, hasil stabil  | Lebih lambat saat inferensi, tidak sebaik Gradient Boosting |
| Gradient Boosting   | Akurasi tinggi, cocok untuk data kompleks | Butuh waktu pelatihan lebih lama, sensitif terhadap noise dan outlier |

---

## Evaluation

### Metrik Evaluasi yang Digunakan

Dalam konteks regresi, metrik evaluasi yang digunakan untuk mengukur performa model adalah:

1. **MAE (Mean Absolute Error)**  
   Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi. MAE memberikan gambaran seberapa besar rata-rata kesalahan dalam satuan nilai target.

2. **MSE (Mean Squared Error)**  
   Mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. MSE memberikan gambaran seberapa besar kesalahan prediksi secara keseluruhan, dengan memberikan bobot lebih besar pada kesalahan yang besar karena menggunakan kuadrat.

3. **RMSE (Root Mean Squared Error)**  
   Mengukur akar dari rata-rata kuadrat error. RMSE memberikan penalti lebih besar pada error yang tinggi dan memiliki satuan yang sama dengan target.

4. **R² (R-squared / Coefficient of Determination)**  
   Menunjukkan seberapa besar variasi dalam data target yang dapat dijelaskan oleh fitur input. Nilai R² berkisar antara 0 hingga 1, dimana nilai yang lebih dekat ke 1 menunjukkan model yang lebih baik.

### Hasil Evaluasi Model Setelah Tuning

| Model                  | MAE (Rp)         | MSE (Rp²)                        | RMSE (Rp)        | R² Score  |
|------------------------|------------------|---------------------------------|------------------|-----------|
| Gradient Boosting       | **582,910,206**  | 8,871,530,371,283,419,136       | **2,978,511,435** | **0.9858** |
| Random Forest          | 721,409,372      | 17,042,014,454,308,581,376      | 4,128,197,482    | 0.9727    |
| Decision Tree          | 1,077,433,723    | 19,970,412,670,443,679,744      | 4,468,826,767    | 0.9680    |

### Interpretasi Hasil

- **MAE** sebesar sekitar Rp 582 juta pada model Gradient Boosting berarti rata-rata kesalahan prediksi harga apartemen relatif kecil untuk pasar properti Jakarta.
- **RMSE** sekitar Rp 2,97 miliar menunjukkan bahwa sebagian besar prediksi memiliki error dalam kisaran yang masih wajar.
- **R² sebesar 0.9858** menunjukkan bahwa hampir 98,6% variasi harga apartemen dapat dijelaskan oleh model Gradient Boosting, menandakan kualitas prediksi yang sangat baik.
- Hasil tuning menunjukkan perbaikan performa yang signifikan terutama pada model Gradient Boosting dibandingkan dengan parameter default.

Model Gradient Boosting Regressor dipilih sebagai model terbaik untuk aplikasi prediksi harga apartemen Jakarta karena akurasinya yang superior dan kestabilan hasil prediksi.

### Perbandingan MAE dan R² Sebelum dan Sesudah Tuning

| Model            | MAE Default (Rp)  | MAE Tuned (Rp)    | R² Default | R² Tuned  |
|------------------|-------------------|-------------------|------------|-----------|
| Random Forest    | 734,432,949       | 721,409,371       | 0.971488   | 0.972695  |
| Decision Tree    | 1,161,969,248     | 1,077,433,722     | 0.962453   | 0.968003  |
| Gradient Boosting| 1,234,944,817     | 582,910,206       | 0.975676   | 0.985786  |

### Best Hyperparameters Setelah Tuning

- **Random Forest**  
  `{'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`

- **Decision Tree**  
  `{'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2}`

- **Gradient Boosting**  
  `{'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300, 'subsample': 0.8}`

### Contoh Prediksi Harga Apartemen

| Index | Harga Asli (Rp)       | Prediksi Random Forest (Rp) | Prediksi Decision Tree (Rp) | Prediksi Gradient Boosting (Rp) |
|-------|-----------------------|-----------------------------|-----------------------------|----------------------------------|
| 1975  | 14,000,000,000        | 14,016,670,000              | 14,000,000,000              | 14,026,820,000                   |
| 363   | 16,000,000,000        | 16,343,330,000              | 16,000,000,000              | 15,910,250,000                   |
| 284   | 44,000,000,000        | 44,433,330,000              | 45,000,000,000              | 44,584,340,000                   |
| 168   | 270,000,000           | 267,393,300                 | 270,000,000                 | 257,826,400                      |
| 100   | 12,000,000,000        | 14,150,000,000              | 14,000,000,000              | 13,177,390,000                   |

### Kesimpulan Model Prediksi Harga Apartemen

Berdasarkan hasil evaluasi setelah tuning, berikut kesimpulan dari tiga model yang diuji:

- **Gradient Boosting Regressor** menunjukkan performa terbaik dengan MAE sebesar Rp 582 juta, RMSE sekitar Rp 2,97 miliar, dan R² sebesar 0.9858. Model ini sangat akurat dan stabil dalam memprediksi harga apartemen, sehingga dipilih sebagai model terbaik.

- **Random Forest Regressor** juga memberikan hasil yang baik dengan MAE Rp 721 juta dan R² sebesar 0.9727. Meskipun lebih baik dari Decision Tree, model ini masih sedikit kalah dari Gradient Boosting dalam hal akurasi.

- **Decision Tree Regressor** memiliki performa paling rendah di antara ketiganya dengan MAE Rp 1,07 miliar dan R² sebesar 0.9680. Meskipun sederhana dan cepat, model ini kurang akurat untuk kasus prediksi harga apartemen di Jakarta.

Secara keseluruhan, **Gradient Boosting** adalah pilihan optimal untuk model prediktif dalam proyek ini.

---

### Referensi

- Badan Pusat Statistik (BPS) Provinsi DKI Jakarta. (2024). *Statistik Harga Properti Residensial di DKI Jakarta*.
- Susanto, A., Pratama, R., & Lestari, D. (2023). *Prediksi Harga Properti Menggunakan Algoritma Machine Learning*. Jurnal Teknologi Informasi dan Ilmu Komputer, 10(2), 115–123.

---
