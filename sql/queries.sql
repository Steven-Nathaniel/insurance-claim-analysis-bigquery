-- A) Setup dataset kerja
CREATE SCHEMA IF NOT EXISTS `is429-bda-steven-nathaniel.garuda_life_uas`
OPTIONS(location="US");

-- B) Staging table (copy dari public dataset)
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015` AS
SELECT
  *
FROM `bigquery-public-data.cms_medicare.inpatient_charges_2015`;

-- C) Q1 — Data Understanding & Data Quality
-- C1. Cek sample & jumlah baris
SELECT * FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015` LIMIT 20;

SELECT COUNT(*) AS n_rows
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`;
-- C2. Missing values
SELECT
  COUNT(*) AS n_rows,
  COUNTIF(drg_definition IS NULL) AS miss_drg_definition,
  COUNTIF(provider_id IS NULL) AS miss_provider_id,
  COUNTIF(provider_state IS NULL) AS miss_provider_state,
  COUNTIF(total_discharges IS NULL) AS miss_total_discharges,
  COUNTIF(average_covered_charges IS NULL) AS miss_avg_covered_charges,
  COUNTIF(average_total_payments IS NULL) AS miss_avg_total_payments,
  COUNTIF(average_medicare_payments IS NULL) AS miss_avg_medicare_payments
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`;
-- C3. Duplicate check (provider_id + drg_definition)
SELECT
  provider_id,
  drg_definition,
  COUNT(*) AS c
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
GROUP BY provider_id, drg_definition
HAVING c > 1
ORDER BY c DESC;
-- C4. Range check numerik
SELECT
  MIN(total_discharges) AS min_total_discharges,
  MAX(total_discharges) AS max_total_discharges,
  MIN(average_covered_charges) AS min_avg_covered_charges,
  MAX(average_covered_charges) AS max_avg_covered_charges,
  MIN(average_total_payments) AS min_avg_total_payments,
  MAX(average_total_payments) AS max_avg_total_payments,
  MIN(average_medicare_payments) AS min_avg_medicare_payments,
  MAX(average_medicare_payments) AS max_avg_medicare_payments
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`;

-- D) Q1 — EDA (untuk screenshot & narasi)
-- D1. Top 20 state by average claim cost
SELECT
  provider_state,
  AVG(average_total_payments) AS avg_claim_cost,
  AVG(average_covered_charges) AS avg_covered_charges,
  SUM(total_discharges) AS total_discharges
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
GROUP BY provider_state
ORDER BY avg_claim_cost DESC
LIMIT 20;
-- D2. Top 20 DRG by average claim cost
SELECT
  drg_definition,
  AVG(average_total_payments) AS avg_claim_cost,
  AVG(average_covered_charges) AS avg_covered_charges,
  SUM(total_discharges) AS total_discharges
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
GROUP BY drg_definition
ORDER BY avg_claim_cost DESC
LIMIT 20;
-- D3. Top 20 provider by average claim cost
SELECT
  provider_id,
  provider_name,
  provider_city,
  provider_state,
  AVG(average_total_payments) AS avg_claim_cost,
  SUM(total_discharges) AS total_discharges
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
GROUP BY provider_id, provider_name, provider_city, provider_state
ORDER BY avg_claim_cost DESC
LIMIT 20;
-- D4. Korelasi
SELECT
  CORR(average_covered_charges, average_total_payments) AS corr_covered_vs_total,
  CORR(total_discharges, average_total_payments) AS corr_discharges_vs_total
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`;
-- D5. IQR bounds (outlier reference)
WITH q AS (
  SELECT APPROX_QUANTILES(average_total_payments, 100) AS qs
  FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
),
bounds AS (
  SELECT
    qs[OFFSET(25)] AS q1,
    qs[OFFSET(75)] AS q3
  FROM q
)
SELECT
  q1, q3,
  (q3 - q1) AS iqr,
  (q1 - 1.5*(q3 - q1)) AS lower_bound,
  (q3 + 1.5*(q3 - q1)) AS upper_bound
FROM bounds;

-- E) Q1 — Data Preparation + Feature Engineering (table untuk ML)
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.ml_features` AS
SELECT
  average_total_payments AS label_avg_total_payments,

  total_discharges,
  average_covered_charges,
  average_medicare_payments,

  SAFE_DIVIDE(average_covered_charges, average_total_payments) AS charge_to_payment_ratio,
  SAFE_DIVIDE(average_medicare_payments, average_total_payments) AS medicare_share,

  provider_state,
  provider_city,

  SPLIT(drg_definition, ' - ')[OFFSET(0)] AS drg_group
FROM `is429-bda-steven-nathaniel.garuda_life_uas.stg_inpatient_2015`
WHERE
  average_total_payments IS NOT NULL
  AND average_covered_charges IS NOT NULL
  AND average_medicare_payments IS NOT NULL
  AND total_discharges IS NOT NULL
  AND average_total_payments > 0
  AND average_covered_charges > 0
  AND total_discharges > 0;

-- F) Train/Test split (repeatable)
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.ml_features_split` AS
SELECT
  *,
  MOD(ABS(FARM_FINGERPRINT(
    CONCAT(provider_state, '|', provider_city, '|', drg_group, '|', CAST(total_discharges AS STRING))
  )), 100) AS split_bucket
FROM `is429-bda-steven-nathaniel.garuda_life_uas.ml_features`;
--F
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.train` AS
SELECT * EXCEPT(split_bucket)
FROM `is429-bda-steven-nathaniel.garuda_life_uas.ml_features_split`
WHERE split_bucket < 80;

CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.test` AS
SELECT * EXCEPT(split_bucket)
FROM `is429-bda-steven-nathaniel.garuda_life_uas.ml_features_split`
WHERE split_bucket >= 80;

-- G) Q2 — Build Model (BigQuery ML)
-- G1. Baseline: Linear Regression
CREATE OR REPLACE MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_linear_reg`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['label_avg_total_payments']
) AS
SELECT
  label_avg_total_payments,
  total_discharges,
  average_covered_charges,
  average_medicare_payments,
  charge_to_payment_ratio,
  medicare_share,
  provider_state,
  provider_city,
  drg_group
FROM `is429-bda-steven-nathaniel.garuda_life_uas.train`;
-- G2. Main model: Boosted Tree Regressor
CREATE OR REPLACE MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['label_avg_total_payments'],
  max_iterations = 50,
  learn_rate = 0.1,
  max_tree_depth = 6,
  subsample = 0.8
) AS
SELECT
  label_avg_total_payments,
  total_discharges,
  average_covered_charges,
  average_medicare_payments,
  charge_to_payment_ratio,
  medicare_share,
  provider_state,
  provider_city,
  drg_group
FROM `is429-bda-steven-nathaniel.garuda_life_uas.train`;

-- H) Q3 — Evaluate Models (test set)
SELECT 'LINEAR_REG' AS model_name, *
FROM ML.EVALUATE(
  MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_linear_reg`,
  (SELECT
     label_avg_total_payments,
     total_discharges,
     average_covered_charges,
     average_medicare_payments,
     charge_to_payment_ratio,
     medicare_share,
     provider_state,
     provider_city,
     drg_group
   FROM `is429-bda-steven-nathaniel.garuda_life_uas.test`)
);

SELECT 'BOOSTED_TREE' AS model_name, *
FROM ML.EVALUATE(
  MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`,
  (SELECT
     label_avg_total_payments,
     total_discharges,
     average_covered_charges,
     average_medicare_payments,
     charge_to_payment_ratio,
     medicare_share,
     provider_state,
     provider_city,
     drg_group
   FROM `is429-bda-steven-nathaniel.garuda_life_uas.test`)
);

-- I) Q3 — Predictions table (buat visualisasi)
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.test_with_id` AS
SELECT
  ROW_NUMBER() OVER() AS row_id,
  *
FROM `is429-bda-steven-nathaniel.garuda_life_uas.test`;
-- I2. Simpan predictions
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.predictions_test` AS
SELECT
  t.*,
  p.predicted_label_avg_total_payments AS predicted_avg_total_payments
FROM `is429-bda-steven-nathaniel.garuda_life_uas.test_with_id` t
JOIN ML.PREDICT(
  MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`,
  (SELECT
     row_id,
     label_avg_total_payments,
     total_discharges,
     average_covered_charges,
     average_medicare_payments,
     charge_to_payment_ratio,
     medicare_share,
     provider_state,
     provider_city,
     drg_group
   FROM `is429-bda-steven-nathaniel.garuda_life_uas.test_with_id`)
) p
USING(row_id);

-- J) Q3 — Visualization queries (untuk chart/screenshot)
-- J1. Actual vs Predicted
SELECT
  label_avg_total_payments AS actual,
  predicted_avg_total_payments AS predicted,
  provider_state,
  drg_group
FROM `is429-bda-steven-nathaniel.garuda_life_uas.predictions_test`;
-- J2. Error per state
SELECT
  provider_state,
  AVG(ABS(predicted_avg_total_payments - label_avg_total_payments)) AS avg_abs_error,
  AVG(predicted_avg_total_payments - label_avg_total_payments) AS avg_bias,
  COUNT(*) AS n
FROM `is429-bda-steven-nathaniel.garuda_life_uas.predictions_test`
GROUP BY provider_state
ORDER BY avg_abs_error DESC;
-- J3. Top DRG predicted cost
SELECT
  drg_group,
  AVG(predicted_avg_total_payments) AS avg_pred_cost,
  AVG(label_avg_total_payments) AS avg_actual_cost,
  COUNT(*) AS n
FROM `is429-bda-steven-nathaniel.garuda_life_uas.predictions_test`
GROUP BY drg_group
ORDER BY avg_pred_cost DESC
LIMIT 20;

-- K) Explainable AI / Interpretability
-- K1. Feature importance
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`)
ORDER BY importance DESC;
-- K2. Global explain
SELECT *
FROM ML.GLOBAL_EXPLAIN(MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`);

-- L) Tabel perbandingan model
CREATE OR REPLACE TABLE `is429-bda-steven-nathaniel.garuda_life_uas.model_comparison` AS
SELECT 'LINEAR_REG' AS model_name, *
FROM ML.EVALUATE(
  MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_linear_reg`,
  (SELECT
     label_avg_total_payments,
     total_discharges,
     average_covered_charges,
     average_medicare_payments,
     charge_to_payment_ratio,
     medicare_share,
     provider_state,
     provider_city,
     drg_group
   FROM `is429-bda-steven-nathaniel.garuda_life_uas.test`)
)
UNION ALL
SELECT 'BOOSTED_TREE' AS model_name, *
FROM ML.EVALUATE(
  MODEL `is429-bda-steven-nathaniel.garuda_life_uas.m_boosted_tree`,
  (SELECT
     label_avg_total_payments,
     total_discharges,
     average_covered_charges,
     average_medicare_payments,
     charge_to_payment_ratio,
     medicare_share,
     provider_state,
     provider_city,
     drg_group
   FROM `is429-bda-steven-nathaniel.garuda_life_uas.test`)
);
