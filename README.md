# KGAT Movie Recommender HetRec 2011

Implementasi **Knowledge Graph Attention Network (KGAT)** untuk sistem rekomendasi film berbasis *graph mining* pada dataset **HetRec 2011 (MovieLens-2k)**.

Proyek ini dibuat untuk mata kuliah Data Mining, Program Studi S1 Sains Data, Universitas Negeri Surabaya (Kelas 2024B).

---

## Latar Belakang

Sistem rekomendasi konvensional berbasis *Collaborative Filtering* rentan terhadap masalah **cold start** dan **data sparsity**. Proyek ini memanfaatkan **knowledge graph** yang menghubungkan user, film, genre, sutradara, aktor, tag, dan negara produksi. Dengan mekanisme **attention**, KGAT mempelajari relasi tingkat tinggi (*high-order connectivity*) antar entitas sehingga menghasilkan rekomendasi yang lebih akurat dan dapat dijelaskan lewat jalur relasi pada graph.

## Dataset

**HetRec 2011 MovieLens-2k** ([GroupLens](https://grouplens.org/datasets/hetrec-2011/)), yaitu data MovieLens yang diperkaya informasi dari IMDb dan Rotten Tomatoes.

| Komponen | Nilai |
|---|---|
| Jumlah user | 2.113 |
| Jumlah film | 10.109 |
| Total rating | 855.598 |
| Genre | 20 |
| Sutradara | 4.060 |
| Aktor | 95.321 |
| Sparsity | ≈ 95,99% |

> Dataset **tidak disertakan** di repo ini. Unduh dari GroupLens lalu letakkan file `.dat`-nya di folder data (lihat bagian *Cara Menjalankan*).

## Alur Penelitian

```
Data Collecting → Preprocessing → Knowledge Graph Construction
→ KGAT Implementation → Training → Evaluation → Recommendation
```

### Preprocessing
- **Filtering cold-start** (3 iterasi): minimal 20 rating per user dan 5 rating per film
- **Binarisasi rating**: rating ≥ 3,5 dianggap interaksi positif (*implicit feedback*)
- **Encoding entitas** ke ID numerik unik dengan mekanisme offset per tipe entitas
- **Split data berbasis timestamp** per user: 80% train / 10% validation / 10% test (mencegah *data leakage*)
- **Negative sampling** untuk training BPR

### Knowledge Graph
- **29.410 node** (user, movie, genre, director, actor, tag, country)
- **6 tipe relasi**: `rates`, `has_genre`, `directed_by`, `acted_in`, `has_tag`, `produced_in`
- Setiap relasi diaugmentasi dengan relasi inversnya, sehingga total **12 tipe relasi** dan **1.056.206 triplet**
- Graph terhubung penuh (1 komponen terhubung), dengan distribusi degree berpola *power law*

## Model

1. **Pre-training TransR** (100 epoch) untuk inisialisasi embedding entitas dan relasi
2. **KGAT** dengan attentive embedding propagation dan loss BPR

| Hyperparameter | Nilai |
|---|---|
| Embedding dimension | 64 |
| Jumlah layer | 3 |
| Dropout | 0,3 |
| Learning rate | 0,001 |
| Weight decay | 1e-5 |
| Regularization weight | 1e-4 |
| Batch size | 1.024 |
| Max epoch / patience | 100 / 10 evaluation cycles |
| Jumlah parameter | 1.883.584 |

## Hasil Evaluasi (Test Set)

| K | Precision | Recall | NDCG | Hit Ratio |
|---|---|---|---|---|
| 5 | 0,0957 | 0,0587 | 0,3357 | 0,4787 |
| 10 | 0,0637 | 0,0755 | 0,3860 | 0,6367 |
| 20 | 0,0392 | 0,0879 | 0,4228 | 0,7841 |

Pada K=20, sekitar **78,41%** pengguna mendapatkan minimal satu film relevan dalam daftar rekomendasi. Recall@20 terbaik pada validation adalah **0,1314**.

## Struktur Repo

```
.
├── movie-recommendation-kgat-graph-mining_source code.ipynb   # Notebook utama (EDA sampai rekomendasi)
├── docs/
│   └── laporan_kelompok_12.pdf     # Laporan lengkap
├── data/                           # Letakkan file .dat HetRec 2011 di sini
└── README.md
```

Isi notebook:
1. Load Data
2. Exploratory Data Analysis
3. Preprocessing
4. Knowledge Graph Construction
5. Graph Structural Analysis
6. Feature Engineering
7. KG Embedding (TransR Pre-training)
8. Model KGAT
9. Fungsi Evaluasi
10. Training KGAT
11. Evaluasi Final
12. Contoh Rekomendasi

## Cara Menjalankan

1. **Clone repo**
```bash
   git clone https://github.com/<username>/kgat-movie-recommender-hetrec2011.git
   cd kgat-movie-recommender-hetrec2011
```

2. **Install dependensi**
```bash
   pip install torch numpy pandas scikit-learn networkx matplotlib jupyter
```

3. **Unduh dataset** HetRec 2011 MovieLens dari [GroupLens](https://grouplens.org/datasets/hetrec-2011/), ekstrak, lalu letakkan file `.dat` di folder `data/`.

4. **Jalankan notebook**
```bash
   jupyter notebook kgat-movie-recommender-hetrec2011_source code.ipynb
```
   Sesuaikan path data pada bagian *Load Data* dengan lokasi file di komputermu. GPU (CUDA) disarankan untuk mempercepat training.

## Temuan Utama

- Relasi `rates` mendominasi graph (**79,3%** dari total triplet), sedangkan `directed_by` dan `produced_in` sangat sedikit.
- Nilai Recall@20 validation cenderung stagnan di kisaran 0,1289–0,1314, artinya model mendekati batas konvergensi dengan konfigurasi saat ini.

## Saran Pengembangan

- Menerapkan *relation-aware sampling* atau pembobotan relasi agar metadata (sutradara, negara) lebih termanfaatkan
- Eksperimen dengan embedding dimension yang lebih besar, jumlah layer berbeda, atau *learning rate scheduling*
- Menambah entitas baru ke graph (penulis skenario, penghargaan, tahun rilis)

## Tim (Kelompok 12)

| NIM | Nama |
|---|---|
| 24031554132 | Moh. Rasya Al Khalifi |
| 24031554040 | Bima Setia Sugiharto |
| 24031554091 | Muhammad Geralldo Agatha Saputra |
| 24031554026 | Najmu Tsaqib Arsalan |

**Dosen Pengampu:** Ulfa Siti Nuraini, S.Stat., M.Stat.
Program Studi S1 Sains Data, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Surabaya

## Referensi

- Wang, X., He, X., Cao, Y., Liu, M., & Chua, T.-S. (2019). *KGAT: Knowledge Graph Attention Network for Recommendation*. KDD 2019.
- Cantador, I., Brusilovsky, P., & Kuflik, T. (2011). *2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)*. RecSys 2011.
- Wu, Z. (2024). *An efficient recommendation model based on KGAT-AX*. arXiv:2409.15315.

## Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset HetRec 2011 mengikuti ketentuan lisensi dari [GroupLens](https://grouplens.org/datasets/hetrec-2011/).
