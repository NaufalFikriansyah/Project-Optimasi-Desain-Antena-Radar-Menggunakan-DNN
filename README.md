# Project Optimasi Desain Antena Radar Menggunakan Deep Neural Network (DNN)

## 📋 Deskripsi Project

Project ini mengimplementasikan optimasi desain antena mikrostrip radar menggunakan pendekatan **Deep Neural Network (DNN) sebagai surrogate model** yang dikombinasikan dengan **Genetic Algorithm (GA)** untuk mencari parameter desain optimal. Metode ini memungkinkan optimasi yang lebih cepat dibandingkan simulasi elektromagnetik langsung karena menggunakan model prediksi yang sudah dilatih.

## 🎯 Tujuan

1. Membangun model DNN surrogate yang dapat memprediksi parameter dimensi antena (L, W, h) berdasarkan karakteristik elektromagnetik (er, VSWR, Bandwidth, Frekuensi Resonansi, S11, Gain, Efisiensi)
2. Mengoptimalkan desain antena menggunakan Genetic Algorithm dengan memanfaatkan model DNN sebagai evaluator fitness
3. Mencari kombinasi parameter optimal untuk mendapatkan performa antena terbaik (gain maksimum, S11 minimum, efisiensi tinggi)

## 🏗️ Arsitektur Project

### Struktur Direktori

```
Project-Optimasi-Desain-Antena-Radar-Menggunakan-DNN/
│
├── 📓 Notebooks/
│   ├── antenna_opt_GA.ipynb          # Script utama: Training DNN + Optimasi GA
│   ├── prediksi_vs_simulasi_plot.ipynb  # Validasi model: Plot prediksi vs simulasi
│   └── optimal_antenna_slide.ipynb      # Visualisasi desain antena optimal
│
├── 📊 Dataset/
│   ├── dataset_patch_rectangular_10GHz_2000rows.csv    # Dataset utama (2000 samples)
│   ├── dataset_patch_rectangular_mixed_DNN.csv        # Dataset campuran
│   ├── dataset_patch_antenna_FR4_200rows.csv
│   ├── dataset_patch_antenna_rogers_200rows.csv
│   ├── dataset_patch_antenna_microstrip_500rows.csv
│   ├── dataset_antenna_radar10GHz_FR4_100.csv
│   ├── test_dataset_rectangular_circular_10GHz.csv
│   ├── data uji1_antena_radar.csv
│   └── data uji2_antena_radar.csv
│
├── 🤖 Model/
│   ├── model_baru.h5                 # Model DNN (mixed dataset)
│   └── model_baru_10ghz.h5           # Model DNN (10GHz dataset)
│
├── 📁 Results/
│   ├── ga_results/                   # Hasil optimasi GA (mixed)
│   │   ├── ga_best_solution2.csv
│   │   ├── ga_top10_solutions.csv
│   │   ├── ga_fitness_evolution2.png
│   │   └── top_L_vs_W2.png
│   │
│   ├── ga_results_10ghz/             # Hasil optimasi GA (10GHz)
│   │   ├── ga_best_solution2.csv
│   │   ├── ga_top10_solutions.csv
│   │   ├── ga_fitness_evolution2.png
│   │   └── top_L_vs_W2.png
│   │
│   ├── data uji1_model baru/         # Hasil validasi data uji 1
│   ├── data uji1_model baru 10ghz/   # Hasil validasi data uji 1 (10GHz)
│   ├── data uji2_model baru/         # Hasil validasi data uji 2
│   ├── data uji2_model baru 10ghz/   # Hasil validasi data uji 2 (10GHz)
│   ├── data test_model baru/         # Hasil validasi test dataset
│   └── data test_model baru 10ghz/   # Hasil validasi test dataset (10GHz)
│
└── 📄 Dokumentasi/
    ├── Metode Penelitian_point 1.docx
    ├── Rencana_(algoritma) penelitian.docx
    ├── Script untuk menghasilan Desain optimal antena.docx
    └── Desain optimal contoh.docx
```

## 🔬 Metodologi

### 1. Data Collection & Preprocessing

**Dataset:**
- Dataset utama terdiri dari 2000 sampel untuk frekuensi 10 GHz
- Fitur input (7 fitur): 
  - `er` (Epsilon_r / Permitivitas relatif)
  - `VSWR` (Voltage Standing Wave Ratio)
  - `Bandwidth_MHz` (Lebar pita)
  - `Resonant_freq_Hz` (Frekuensi resonansi)
  - `S11_min_dB` (Return loss minimum)
  - `Peak_gain_dBi` (Gain puncak)
  - `Efficiency_pct` (Efisiensi persentase)

- Target output (3 fitur):
  - `L_mm` (Panjang patch dalam mm)
  - `W_mm` (Lebar patch dalam mm)
  - `h_mm` (Tebal substrat dalam mm)

**Preprocessing:**
- Normalisasi menggunakan `MinMaxScaler` untuk input dan output
- Split data: 80% training, 20% testing
- Validasi: 10% dari training data

### 2. Deep Neural Network (Surrogate Model)

**Arsitektur Model:**
```
Input Layer:  7 features (karakteristik elektromagnetik)
    ↓
Dense Layer:  128 units, ReLU activation
    ↓
Dropout:      0.15 (regularisasi)
    ↓
Dense Layer:  64 units, ReLU activation
    ↓
Dense Layer:  32 units, ReLU activation
    ↓
Output Layer: 3 units, Linear activation (L, W, h)
```

**Hyperparameter:**
- Optimizer: Adam (learning rate = 1e-3)
- Loss function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Batch size: 8
- Epochs: 100 (dengan early stopping, patience=30)
- Callbacks: EarlyStopping, ModelCheckpoint

**Fungsi Model:**
Model DNN berfungsi sebagai **surrogate model** yang memprediksi dimensi antena berdasarkan karakteristik elektromagnetik yang diinginkan. Ini memungkinkan evaluasi cepat tanpa perlu simulasi elektromagnetik yang memakan waktu.

### 3. Genetic Algorithm (Optimasi)

**Parameter Optimasi:**
- `L_patch_mm`: 5.8 - 7.0 mm
- `W_patch_mm`: 8.5 - 9.8 mm
- `inset_x_mm`: 0.5 - 3.5 mm
- `feed_width_mm`: 2.0 - 4.0 mm

**Parameter GA:**
- Population size: 40 individu
- Generations: 60 generasi
- Crossover probability: 0.9
- Mutation probability: 0.2
- Tournament selection: k=3

**Operators:**
- **Selection**: Tournament selection
- **Crossover**: Arithmetic crossover
- **Mutation**: Gaussian mutation dengan boundary enforcement

**Fitness Function:**
Fitness dihitung berdasarkan perbedaan antara parameter yang dioptimasi dengan prediksi model DNN, dengan tujuan meminimalkan perbedaan tersebut.

### 4. Validasi & Evaluasi

**Validasi Model:**
- Plot perbandingan prediksi vs simulasi untuk berbagai dataset uji
- Perhitungan metrik akurasi (MAE, MSE)
- Visualisasi scatter plot untuk setiap parameter output

**Hasil Optimasi:**
- Top 10 solusi terbaik
- Best solution dengan detail parameter
- Grafik evolusi fitness (best & mean per generasi)
- Visualisasi distribusi parameter solusi terbaik

## 📝 Script Utama

### 1. `antenna_opt_GA.ipynb`

**Fungsi Utama:**
- Load dan prepare dataset
- Build dan train model DNN surrogate
- Implementasi Genetic Algorithm
- Optimasi parameter antena
- Visualisasi hasil optimasi

**Fungsi-fungsi Penting:**

```python
# Load dataset
load_and_prepare(csv_path)

# Build model DNN
build_surrogate_model(input_dim, output_dim, units, dropout, lr)

# Train model
train_surrogate(df, model_path, retrain)

# GA operators
sample_individual(bounds)
tournament_selection(pop, fitnesses, k)
crossover(parent1, parent2, cr)
mutation(ind, bounds, mu)

# Fitness evaluation
fitness_from_surrogate(individual, model, scaler_X, scaler_y, df_train)

# Run GA
run_ga(model, scaler_X, scaler_y, bounds, df_train, pop_size, generations, ...)

# Visualisasi
visualize_ga(best_history, mean_history, final_infos_sorted)
```

**Output:**
- Model DNN tersimpan (.h5)
- File CSV dengan top 10 solusi
- File CSV dengan best solution
- Grafik evolusi fitness
- Scatter plot distribusi parameter

### 2. `prediksi_vs_simulasi_plot.ipynb`

**Fungsi:**
- Load model DNN yang sudah dilatih
- Prediksi untuk dataset uji
- Perbandingan prediksi vs nilai simulasi aktual
- Visualisasi scatter plot untuk validasi akurasi

**Output:**
- Scatter plot prediksi vs simulasi untuk L, W, h
- File CSV perbandingan hasil
- Metrik akurasi (MAE, RMSE)

### 3. `optimal_antenna_slide.ipynb`

**Fungsi:**
- Visualisasi 2D layout antena mikrostrip (top view)
- Menampilkan parameter desain optimal
- Generate gambar PNG untuk dokumentasi

**Output:**
- File PNG dengan layout antena dan parameter

## 🚀 Cara Menggunakan

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas matplotlib numpy
```

### Langkah-langkah

#### 1. Training Model DNN

```python
# Buka antenna_opt_GA.ipynb
# Jalankan semua cell

# Atau ubah konfigurasi dataset:
DATA_CSV = 'dataset_patch_rectangular_10GHz_2000rows.csv'
MODEL_SAVE = "model_baru_10ghz.h5"
```

#### 2. Optimasi dengan GA

```python
# Setelah model dilatih, jalankan optimasi GA
# Parameter GA dapat disesuaikan:
pop_size = 40
generations = 60
cx_prob = 0.9
mut_prob = 0.2
```

#### 3. Validasi Model

```python
# Buka prediksi_vs_simulasi_plot.ipynb
# Set path dataset uji dan model
# Jalankan untuk melihat akurasi prediksi
```

#### 4. Visualisasi Desain Optimal

```python
# Buka optimal_antenna_slide.ipynb
# Update parameter dengan hasil optimasi
# Jalankan untuk generate gambar layout antena
```

## 📊 Hasil & Output

### Model Performance
- Model DNN berhasil dilatih dengan loss validation yang rendah
- Early stopping mencegah overfitting
- Model dapat memprediksi dimensi antena dengan akurasi yang baik

### Optimasi GA
- Menghasilkan top 10 solusi dengan fitness terbaik
- Evolusi fitness menunjukkan konvergensi
- Parameter optimal ditemukan dalam rentang yang ditentukan

### Validasi
- Prediksi model menunjukkan korelasi tinggi dengan nilai simulasi
- Error relatif kecil untuk parameter L, W, dan h
- Model dapat digunakan sebagai surrogate yang reliable

## 📚 Dokumentasi Tambahan

Project ini dilengkapi dengan dokumentasi dalam format Word (.docx):

1. **Metode Penelitian_point 1.docx** - Penjelasan metodologi penelitian
2. **Rencana_(algoritma) penelitian.docx** - Rencana algoritma yang digunakan
3. **Script untuk menghasilan Desain optimal antena.docx** - Panduan penggunaan script
4. **Desain optimal contoh.docx** - Contoh desain optimal yang dihasilkan

## 🔧 Konfigurasi

### Dataset Configuration

```python
# Untuk dataset mixed
DATA_CSV = 'dataset_patch_rectangular_mixed_DNN.csv'
MODEL_SAVE = "model_baru.h5"
RESULTS_DIR = "ga_results"

# Untuk dataset 10GHz
DATA_CSV = 'dataset_patch_rectangular_10GHz_2000rows.csv'
MODEL_SAVE = "model_baru_10ghz.h5"
RESULTS_DIR = "ga_results_10ghz"
```

### GA Bounds Configuration

```python
BOUNDS = {
    'L_patch_mm': (5.8, 7.0),      # Rentang panjang patch
    'W_patch_mm': (8.5, 9.8),     # Rentang lebar patch
    'inset_x_mm': (0.5, 3.5),     # Rentang kedalaman inset
    'feed_width_mm': (2.0, 4.0)   # Rentang lebar feed
}
```

## 📈 Metrik Evaluasi

### Model DNN
- **Loss (MSE)**: Mean Squared Error pada validation set
- **MAE**: Mean Absolute Error untuk setiap output (L, W, h)
- **R² Score**: Koefisien determinasi (jika dihitung)

### GA Optimization
- **Best Fitness**: Fitness terbaik per generasi
- **Mean Fitness**: Rata-rata fitness populasi per generasi
- **Convergence**: Jumlah generasi hingga konvergen

### Validation
- **Prediction Error**: Perbedaan prediksi vs simulasi
- **Relative Error**: Error relatif dalam persentase
- **Correlation**: Korelasi antara prediksi dan simulasi

## 🎓 Penjelasan Konsep

### Surrogate Model
Model DNN berfungsi sebagai **surrogate model** yang menggantikan simulasi elektromagnetik yang mahal secara komputasi. Dengan model ini, evaluasi fitness dalam GA dapat dilakukan dengan sangat cepat (mikrodetik) dibandingkan simulasi penuh (menit hingga jam).

### Inverse Design
Pendekatan ini menggunakan **inverse design** dimana:
- Input: Karakteristik elektromagnetik yang diinginkan
- Output: Dimensi fisik antena yang menghasilkan karakteristik tersebut

### Multi-objective Optimization
Optimasi dilakukan untuk mencapai beberapa tujuan sekaligus:
- Gain maksimum
- S11 minimum (impedance matching baik)
- Efisiensi tinggi

## ⚠️ Catatan Penting

1. **Reproducibility**: Gunakan `RND_SEED = 42` untuk hasil yang dapat direproduksi
2. **Model Loading**: Model yang sudah dilatih dapat dimuat tanpa perlu retrain
3. **Dataset**: Pastikan dataset memiliki kolom yang sesuai dengan `SURROGATE_INPUT_NAMES` dan `TARGET_NAMES`
4. **Memory**: Training dengan dataset besar (2000+ rows) memerlukan memori yang cukup
5. **GPU**: Training dapat dipercepat dengan GPU jika tersedia

## 🔄 Workflow Lengkap

```
1. Data Collection
   ↓
2. Data Preprocessing & Normalization
   ↓
3. Train DNN Surrogate Model
   ↓
4. Validate Model (prediksi vs simulasi)
   ↓
5. Setup Genetic Algorithm
   ↓
6. Run GA Optimization (menggunakan DNN sebagai evaluator)
   ↓
7. Analyze Results (top solutions, fitness evolution)
   ↓
8. Visualize Optimal Design
   ↓
9. Verification (simulasi ulang dengan parameter optimal)
```

## 📞 Kontak & Support

Untuk pertanyaan atau masalah terkait project ini, silakan buka issue atau hubungi pengembang.

## 📄 License

Project ini dibuat untuk keperluan penelitian dan akademik.

---

**Dibuat dengan ❤️ untuk optimasi desain antena radar yang lebih efisien**

