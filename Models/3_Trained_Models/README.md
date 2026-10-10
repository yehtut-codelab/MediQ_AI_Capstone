# Trained models

**App developer: see `5_Documentation/APP_INTEGRATION.md` for exactly which files the MediQ AI app should load
(waiting time: `deployment_global_model/`; queue depth: the LSTM in `per_category_models/`) and what inputs it needs.**

## What each model does, roughly

- **Waiting-time model (LightGBM):** predicts how long a patient will wait when they join a queue (off by about 10.6 minutes on average).
- **Likely-range models (P10 / P50 / P90):** give a low, middle and high estimate instead of one number.
- **Breach classifier:** flags patients likely to wait longer than the clinical limit.
- **LSTM:** forecasts how many patients will be waiting in a service over the next hour (off by 1.65 patients on average).
- **HMM** (teammate): labels how congested a service is right now (low / medium / high).
- **Petri-net service times** (teammate): average service time per service, used by the resource-planning simulator.

More detail, inputs and accuracy: `5_Documentation/APP_INTEGRATION.md`.

## `per_category_models/`: the models reported in the report

Despite the folder name, this holds **all** submitted models from the pipeline
(`2_Source_Code/model_pipeline/scripts/02_train_lgbm.py`, `03_train_lstm.py`, `04_train_hmm.py`),
trained on 2 Oct 2023 - 30 May 2025 and tested on 31 Jul - 30 Sep 2025.

| Model | Files | Test result |
|---|---|---|
| Waiting time, LightGBM (one per service category) | `lgbm_{category}.pkl` | mean MAE 9.02 min |
| Likely range (P10 / P50 / P90) | `lgbm_{category}_p10/_p50/_p90.pkl`, `quantile_calibration.json` | see `7_Validation/quantile_calibration_results.json` |
| Breach classifier | `breach_classifier.pkl` | AUC 0.885, F1 0.602 |
| Room encoding (model input) | `room_encoder.pkl` | fitted on training data only |
| LightGBM settings and results | `lgbm_metadata.json` | |
| **Queue depth, LSTM** | **`lstm_queue_forecast.pt`** (PyTorch weights), **`lstm_queue_scaler.pkl`**, **`lstm_metadata.json`** | MAE 1.65 patients vs 2.84 persistence |
| Queue states, HMM (teammate) | `clinic_hmms.pkl`, `hmm_metadata.json` | accuracy 79.9-93.2% |
| Petri net service times (teammate) | `stpn_service_params.json` | |

Loading the LSTM: the network class is `QueueLSTM` in `03_train_lstm.py`
(`model.load_state_dict(torch.load("lstm_queue_forecast.pt"))`); `7_Validation/validate_models.py` shows a full example.

## `deployment_global_model/`

One LightGBM model for all categories (category as an input), used by
`2_Source_Code/deployment/scripts/predictor.py`. Same accuracy as the per-category models (pooled MAE 10.62 min).

## `extended_through_may2026/`

LightGBM and LSTM retrained on data to January 2026 with the room mapping (`8_Extended_Data_Study`).
A deployment candidate, **not** the models reported in the report. No HMM.

## Not here

`2_Source_Code/mediq_ai_app/Models/` holds the MediQ AI app's own earlier models (XGBoost waiting time, LSTM, HMM)
from the legacy pipeline. They are not the submitted models.
