# Analisis Biaya Klaim Asuransi Kesehatan dengan Google BigQuery

Proyek individu mata kuliah Big Data Analytics (2025), Universitas Multimedia Nusantara.

**Tools:** Google BigQuery · SQL · BigQuery ML

## Konteks
Studi kasus perusahaan asuransi jiwa dan kesehatan yang ingin memahami variasi biaya klaim antar wilayah dan penyedia layanan, sebagai dasar evaluasi risiko dan perencanaan premi.

**Data:** BigQuery public dataset `bigquery-public-data.cms_medicare.inpatient_charges_2015` (201.876 baris klaim rawat inap).

## Yang Dikerjakan
1. **Data ingestion dan staging:** menyalin dataset publik ke tabel staging.
2. **Data quality:** cek missing value, duplikasi provider–diagnosis, dan rentang nilai numerik.
3. **Agregasi dan EDA:** biaya klaim per negara bagian, per diagnosis (DRG), dan per provider.
4. **Feature engineering** untuk pemodelan.
5. **Pemodelan dengan BigQuery ML:** Linear Regression sebagai baseline dan **Boosted Tree Regressor** sebagai model utama untuk memprediksi biaya klaim, dengan train/test split 80/20.
6. **Evaluasi dan interpretasi:** membandingkan kedua model di test set, membuat tabel prediksi, serta menganalisis feature importance dan global explain.

## File
- `sql/queries.sql` – seluruh query SQL yang digunakan
