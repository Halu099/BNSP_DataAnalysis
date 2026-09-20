# Presentasi Teknis — Deteksi Transaksi Fraud E-Commerce
**PT Data Analytics Ritel | Associate Data Scientist**

---

## Slide 1: Judul & Perkenalan
- **Judul**: Deteksi Transaksi Penipuan (Fraud) pada E-Commerce
- **Skenario**: Associate Data Scientist di PT Data Analytics Ritel
- **Tujuan**: Membangun model klasifikasi untuk mengidentifikasi transaksi fraud
- **Dataset**: Fraudulent E-Commerce Transactions (Kaggle) — 23.634 transaksi, 16 kolom

---

## Slide 2: Profil Data (Unit 3)
- **23.634 baris** × 16 kolom
- **Kolom target**: `Is Fraudulent` (0 = legit, 1 = fraud)
- **Distribusi kelas**: 94.83% legit vs 5.17% fraud (imbalanced)
- **Fitur**: Transaction Amount, Payment Method, Product Category, Customer Age, Device Used, Shipping/Billing Address, dll.
- **Insight awal**: Fraud lebih tinggi pada transaksi bernilai besar (korelasi 0.275)

---

## Slide 3: Objek Data (Unit 5)
- **1 baris = 1 transaksi e-commerce**
- Setiap transaksi memiliki: data pelanggan, data produk, data pembayaran, data lokasi
- Fitur `Transaction ID` dan `Customer ID` bersifat unik (tidak cocok untuk model)
- Fitur `Shipping Address` dan `Billing Address` bisa menjadi sinyal fraud

---

## Slide 4: Pembersihan Data (Unit 4, 6)
| Masalah | Jumlah | Penanganan |
|---|---|---|
| Duplikat Transaction ID | 0 | Tidak ada duplikat |
| Outlier Transaction Amount | 1.339 (5.67%) | Winsorize (IQR method) |
| Outlier Customer Age | 152 (<10 atau >80) | Imputasi median |
| Missing value (Age) | 152 | Median imputation |

**Hasil**: 0 null value setelah pembersihan

---

## Slide 5: Feature Engineering (Unit 7)
| Kategori | Fitur Baru | Justifikasi |
|---|---|---|
| Date | `Txn_Month`, `Txn_DayOfWeek`, `Txn_IsWeekend` | Pola fraud berbeda di weekend |
| Interaction | `Amount_Per_Qty`, `Addr_Match` | Shipping ≠ Billing = sinyal fraud |
| User Agg | `User_Txn_Count`, `User_Avg_Amount`, `Amt_UserRatio` | Transaksi anomali vs rata-rata user |
| Encoding | One-hot (Payment, Category, Device) | Kategori → numerik untuk model |
| Frequency | `Location_Freq` | Lokasi dengan volume tinggi |

**Total**: 25 fitur akhir (dari 16 kolom awal)

---

## Slide 6: Pemilihan Model (Unit 8)
- **3 model diuji**: Logistic Regression, Random Forest, LightGBM
- **Penanganan imbalanced**: `class_weight='balanced'` (LR, RF) + `scale_pos_weight` (LightGBM)
- **Train/test split**: 80/20 stratified (18.907 train / 4.727 test)
- **Evaluasi fokus**: F1-Score (prioritas untuk kelas minoritas)

---

## Slide 7: Perbandingan Model (Unit 9)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 69.5% | 10.7% | 67.2% | 18.5% | 0.764 |
| Random Forest | 94.4% | 39.4% | 15.2% | 21.9% | 0.791 |
| **LightGBM** | **87.9%** | **21.2%** | **49.6%** | **29.7%** | **0.785** |

**LightGBM dipilih**: F1-Score tertinggi (0.297) + balance precision-recall terbaik

---

## Slide 8: Analisis Model Terbaik (LightGBM)
- **Confusion Matrix**: 4.035 TN, 448 FP, 123 FN, 121 TP
- **Recall 49.6%**: Model menangkap ~setengah kasus fraud
- **Precision 21.2%**: Dari semua yang diprediksi fraud, 21.2% benar fraud
- **Tradeoff**: Untuk bisnis, lebih baik false positive daripada miss fraud (recall lebih penting)
- **ROC-AUC 0.785**: Kemampuan diskriminasi cukup baik

---

## Slide 9: Feature Importance & Sinyal Fraud
**Top Sinyal Fraud**:
1. `User_Txn_Count` — Frekuensi transaksi per user
2. `Amt_UserRatio` — Rasio amount vs rata-rata user
3. `Transaction Amount` — Nilai transaksi tinggi
4. `Addr_Match` — Shipping ≠ Billing address
5. `Account Age Days` — Akun baru lebih rentan fraud

**Rekomendasi fitur untuk production**:
- Real-time scoring dengan threshold yang bisa diatur
- Flag transaksi: `Addr_Match=0` + `Amt_UserRatio > 2` = high risk

---

## Slide 10: Kesimpulan & Rekomendasi Bisnis
**Kesimpulan**:
- Model LightGBM mampu mendeteksi fraud dengan F1-Score 0.297
- Dataset imbalanced (5.17% fraud) membuat precision rendah — wajar untuk fraud detection

**Rekomendasi**:
1. **Threshold tuning**: Naikkan threshold → lebih presisi, turunkan → lebih recall
2. **Deploy sebagai API**: Real-time scoring untuk setiap transaksi baru
3. **Monitoring berkala**: Retraining bulanan dengan data terbaru
4. **Review manual**: User dengan `User_Txn_Count` sangat tinggi perlu review
5. **Hybrid approach**: Model + rule-based (addr mismatch + high amount = auto-flag)

---
*Presentasi ini mendemonstrasikan pemahaman seluruh unit kompetensi (1-9)*
