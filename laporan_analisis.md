# Laporan Analisis — Deteksi Transaksi Fraud E-Commerce
**PT Data Analytics Ritel | Associate Data Scientist**

---

## 1. Ringkasan Profil Data (Unit 3)
- **Jumlah baris**: 23,634
- **Jumlah kolom awal**: 16
- **Kolom target**: `Is Fraudulent` (0 = legit, 1 = fraud)
- **Distribusi kelas**: Legit 22,412 (94.8%) | Fraud 1,222 (5.2%)
- **Tipe data**: 6 numerik, 5 kategorik, 3 teks

## 2. Metodologi Pembersihan Data (Unit 4, 6)
| Jenis | Deteksi | Penanganan |
|---|---|---|
| Duplikat Transaction ID | 0 baris | Dihapus |
| Outlier Transaction Amount | 1339 baris (IQR) | Winsorize ke upper bound 636.71 |
| Outlier Customer Age | 152 baris (<10 / >80) | Ditandai NaN, diimputasi |
| Missing value numerik | Per kolom | Median imputation |
| Missing value kategorik | -- | Diisi "Unknown" |
| **Sisa missing value** | **0** | -- |

## 3. Feature Engineering (Unit 7)
- **Date features**: `Txn_Month`, `Txn_DayOfWeek`, `Txn_IsWeekend`, `Txn_DayOfMonth`
- **Interaction**: `Amount_Per_Qty`, `Addr_Match` (1 jika Shipping == Billing)
- **User aggregates**: `User_Txn_Count`, `User_Avg_Amount`, `User_Avg_Age`, `Amt_UserRatio`
- **Encoding**: Frequency (`Location_Freq`), One-hot (Payment Method, Product Category, Device Used)
- **Fitur akhir**: 25 kolom fitur

## 4. Pemilihan Model (Unit 8)
Kelas imbalanced ditangani dengan `class_weight='balanced'` (LR, RF) dan `scale_pos_weight` (LightGBM).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6947 | 0.1074 | 0.6721 | 0.1852 | 0.7643 |
| Random Forest | 0.9442 | 0.3936 | 0.1516 | 0.2189 | 0.7913 |
| **LightGBM** | **0.879** | **0.2123** | **0.4959** | **0.2973** | **0.7854** |

**Model terbaik**: LightGBM — dipilih berdasarkan F1-Score.

## 5. Evaluasi Model (Unit 9)
- **F1-Score** diprioritaskan karena dataset imbalanced
- **PR-AUC** (0.2872) menunjukkan performa kelas minoritas
- **ROC-AUC** (0.7854) — membedakan fraud vs legit
- Top fitur: `User_Txn_Count`, `Amt_UserRatio`, `User_Avg_Amount`

## 6. Rekomendasi Bisnis
1. Flag transaksi `Addr_Match = 0` + `Amt_UserRatio > 2` sebagai high-risk
2. Naikkan threshold jika prioritaskan presisi (kurangi false alarm)
3. Deploy model sebagai API real-time scoring
4. Retraining bulanan dengan data terbaru
5. Review manual untuk user dengan `User_Txn_Count` sangat tinggi

---
*Laporan dihasilkan otomatis oleh pipeline.py*
*Tanggal: 2026-09-20 13:21*
