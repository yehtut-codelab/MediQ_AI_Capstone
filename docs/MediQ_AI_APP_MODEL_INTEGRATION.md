# MediQ AI — Integration of the Trained Models into the Application

**Date:** 2026-10-11
**Scope:** how the three trained models (LightGBM wait time, HMM congestion, LSTM queue
forecast) are loaded, fed and used by the backend, and where their outputs appear in the app.
**Related:** `MediQ_AI_APP_SPECIFICATION.md` (§6), `MediQ_AI_APP_PATHWAY_TRIAGE.md`,
`Models/3_Trained_Models/README.md` (the modelling team's hand-over note)

## 1. Summary

The app does not train anything. It loads model files from `Models/`, rebuilds each
model's inputs from a replayed historical dataset (there is no live clinic feed), and
serves the predictions through three surfaces: the arrival simulator, patient
registration, and the supervisor multi-agent analysis.

| Model | Answers | Output in the app |
|---|---|---|
| LightGBM (point, P10/P50/P90, breach classifier) | How long will this patient wait at this station? | Estimate in minutes, likely range, chance of exceeding the station's limit |
| Gaussian HMM, one per service category | How congested is this station right now? | Low / Medium / High |
| LSTM | How long will the queue be over the next hour? | Queue depth for the next 4 × 15 minutes, and a rising / stable / falling trend |

Two files hold the integration: `backend/app/services/models_service.py` (loading and
prediction) and `backend/app/services/livestate.py` (building the inputs).

```
Clean_Dataset/cleaned_events.csv ──▶ livestate.py ──▶ model inputs ──▶ models_service.py ──▶ routes.py / supervisor_graph.py ──▶ pages
        (replayed clinic state)      (as of a timestamp)                (Models/*.pkl, *.pt)
```

## 2. Model files

| File (under `Models/`) | Used for | Loaded by |
|---|---|---|
| `3_Trained_Models/deployment_global_model/lgbm_global.pkl` | Wait-time point estimate | `_wait_models()` |
| `…/lgbm_global_p10.pkl`, `_p50.pkl`, `_p90.pkl` | Likely range | `_wait_models()` |
| `…/quantile_calibration.json` | Per-category shift applied to each quantile | `_wait_models()` |
| `…/breach_classifier.pkl` | Chance the wait exceeds the station's limit | `_wait_models()` |
| `…/deployment_metadata.json` | Feature list, category and clinic encodings, breach limits | `wait_meta()` |
| `lgbm_app_profile.json` | Typical values for inputs the app lacks; accuracy as the app calls the model | `wait_profile()` |
| `clinic_hmms.pkl`, `hmm_metadata.json` | Congestion state | `_hmms()` |
| `lstm_queue_forecast.pt`, `lstm_queue_scaler.pkl`, `lstm_queue_metadata.json` | Queue forecast | `_lstm()` |
| `waiting_time_xgb.pkl`, `waiting_time_metadata.json` | Legacy XGBoost wait model — not used by the app, kept for `scripts/audit_wait_model.py` | `_xgb()` |

`room_encoder.pkl` is in the deployment folder but the app does not load it, because the
app never knows a new patient's room (§4.1).

**Loading.** Every loader is wrapped in `lru_cache`, so a model is read from disk the
first time it is needed and kept in memory afterwards. Models are not loaded at startup;
the first request that needs them pays the load time. Paths are fixed relative to the
repository (`Models/`), not configurable through `.env`. The model files are committed;
the two other folders in the hand-over (`per_category_models/`,
`extended_through_may2026/`) are gitignored and not used.

**Libraries.** `lightgbm` and `joblib` for the wait models, `hmmlearn` for the HMMs,
`torch` for the LSTM. The LSTM file holds weights only, so `models_service.py` rebuilds
the network (`_QueueLSTM`: two LSTM layers of 64 and 32 units, dropout 0.2, a linear
layer to 4 outputs) from the sizes in the metadata.

## 3. Where the inputs come from — the replayed clinic

The models need the state of the clinic at a moment in time: who is waiting, how long
recent patients waited. With no live feed, `livestate.py` reconstructs that state from
`Clean_Dataset/cleaned_events.csv` (405,600 station visits, 2024-10-01 → 2025-09-30),
read once and cached.

| Function | Returns | Used by |
|---|---|---|
| `queue_depth(clinic, category, at)` | Patients whose wait has started and whose service has not | Display, bottleneck and resource agents |
| `recent_waits(clinic, category, at)` | Waits of up to 40 patients who started service in the last 3 hours | HMM |
| `lstm_bucket_features(clinic, category, at)` | The last 8 fifteen-minute buckets, 8 features each | LSTM |
| `wait_model_features(clinic, category, at, …)` | The 17 numeric inputs of the wait model | LightGBM |
| `analog_timestamp(at)` | An in-range timestamp to replay | All entry points |

**Dates outside the dataset.** A request for a date the dataset does not cover (today,
for example) is not rejected. `analog_timestamp` maps it to the nearest date inside the
range with the same weekday and time of day, and the response carries
`is_analog_replay` and a `replay_note` so the page can label the result as a pattern
replay. Demo "golden samples" instead pin an exact recorded moment with `replay_at`.

## 4. Integration, model by model

### 4.1 Wait time — LightGBM

**Call:** `models_service.predict_wait(category, livestate.wait_model_features(...))`

**Inputs.** The model takes the service category plus 17 numeric features. The app
derives most of them from the replayed clinic, and fills the rest with typical values:

| Input | Source in the app |
|---|---|
| `service_cat` / `cat_encoded` | The station's category; an unknown category becomes `other` |
| `clinic_encoded`, `hour_of_day`, `day_of_week`, `month` | The request |
| `days_to_next_holiday`, `days_since_last_holiday` | Singapore holiday list in `livestate.py` |
| `queue_depth`, `queue_depth_lag1`, `queue_drain_rate` | Patients waiting in this category now and 15 minutes ago |
| `cross_cat_load`, `consultation_queue_at_arrival` | Patients waiting in all categories, and in consultation |
| `appointment_density_next30` | Appointments in the 30 minutes after this patient's |
| `appointment_lag_clipped` | Arrival minus appointment, clipped to −180…+60 min; a typical value when no appointment is given |
| `visit_position` | The request if given, otherwise a typical value |
| `room_number_encoded`, `room_queue_depth`, `prior_visit_count` | **Always** a typical value — the app has no room or visit history for a new patient |

The typical values are per clinic and category, with a per-category fallback, and come
from `lgbm_app_profile.json` (`models_service.wait_defaults`).

The queue counts for the model are taken over **both clinics pooled**, because that is
how the training pipeline built them. The queue depth shown on the page is per clinic.
The two numbers are different on purpose.

**Outputs.**

- `estimate_min` — the point model predicts log1p(minutes); the app applies expm1 and
  floors at 0.
- `p10_min`, `p50_min`, `p90_min` — each quantile model's prediction plus its
  per-category calibration shift, floored at 0, then sorted.
- `breach_limit_min`, `breach_probability`, `breach_flag` — the limit is 20 min for
  visual acuity, 45 for consultation, 60 for visual field test and 30 otherwise; the flag
  is raised at a probability of 0.4 or more.

The point model and the quantile models are separate models, so the point estimate can
fall outside P10–P90. The registration route widens the displayed range to include it.

### 4.2 Congestion — HMM

**Call:** `models_service.congestion_state(category, livestate.recent_waits(...))`

One three-state Gaussian HMM per service category (nine in total). The app passes the
recent waits as the observation sequence, decodes the most likely state sequence, and
reports the last state as Low, Medium or High together with that state's mean wait. If
no patient started service in the last three hours, or the category has no model, the
state is `unknown`.

### 4.3 Queue forecast — LSTM

**Call:** `models_service.forecast_queue_depth(livestate.lstm_bucket_features(...))`

The input is 8 buckets × 8 features (queue depth, arrivals, mean wait, hour, weekday,
month, category code, weekend flag). The app scales it with the saved MinMax scaler,
runs the network, and converts the 4 outputs back to patients using the scaler's range
for the queue-depth column. If the last two hours had no activity the forecast is empty
and the trend is `no-data`.

**Trend.** The first forecast bucket is compared with the current depth: more than 15%
higher is `rising`, more than 15% lower is `falling`, otherwise `stable`.

## 5. Where the models are used

### 5.1 Shared pipeline — `_run_queue_models` (`api/routes.py`)

Runs all three models for one clinic, category and timestamp and returns one result.
Both endpoints below call it, so they always agree.

| Endpoint | Page | What the models contribute |
|---|---|---|
| `POST /api/v1/queue/simulate` | `/` Arrival simulator | The whole response: estimate, range, breach, congestion state, forecast, trend |
| `POST /api/v1/patients/register` | `/register` New patient entry | The headline wait estimate and the `model_*` fields on the ticket |
| `GET /api/v1/models/wait-time` | `/register` (model card) | No prediction — the model's inputs, training data and measured accuracy |

**Registration.** The service type entered at registration is mapped to a model
category (unmapped types become `other`). The LightGBM estimate is the headline; the
explanation text states the estimate, the range, the typical error and, when RAG found
similar cases, their median for comparison. If the models fail, the ticket falls back to
the RAG estimate and carries `model_unavailable_reason`.

### 5.2 Supervisor multi-agent graph (`agents/supervisor_graph.py`)

`POST /api/v1/agent/analyze` (and `/stream`) → `/agent` AI Agent Console.

| Agent | Model | How the output is used |
|---|---|---|
| Waiting Time Analysis | LightGBM, plus a RAG median from Qdrant | Estimate, range, breach, typical error |
| Bottleneck Detection | HMM, for every category in the clinic | A station is flagged when its state is High, or Medium with 10 or more waiting |
| Queue Depth Forecasting | LSTM | Next-hour forecast and trend |
| Resource Planning | None — queue depths only | Options for stations with 5 or more waiting |
| Predictive Analytics | None — combines the others | LightGBM estimate and range × 1.15 if the queue is rising, × 0.85 if falling |

The supervisor then combines the model outputs in code:

- **Integrate** — the consensus wait is the mean of the LightGBM estimate and the RAG median.
- **Validate** — confidence starts at 1.0 and loses points when LightGBM and RAG differ
  by more than half the consensus, when the HMM says High while the forecast is falling
  and the wait is short, or when there is no RAG evidence or no forecast data.

No LLM produces or alters a number. The LLM only phrases the final report.

### 5.3 Not model-driven

The station overview (`/stations`) shows historical statistics from the same replayed
dataset, not predictions. The SOP Q&A agent and the pathway and triage logic use no
trained model from `Models/`. The cross-encoder reranker used by RAG is a separate
pretrained model downloaded from Hugging Face, not one of the project's trained models.

## 6. Accuracy as integrated

The wait model was audited the way the app calls it (`scripts/audit_lgbm_global.py`,
results stored in `lgbm_app_profile.json` and shown on the model card). Training data:
649,488 events, 2023-10-02 → 2025-05-30. Test window: 73,369 events, 2025-07-31 →
2025-09-30.

| How the model is called | MAE (min) | RMSE | R² |
|---|---:|---:|---:|
| With all inputs as trained | 11.40 | 19.04 | 0.508 |
| As the app calls it, appointment time given | 14.78 | 24.04 | 0.216 |
| As the app calls it, no appointment time | 15.60 | 25.44 | 0.122 |
| Baseline: station median | 15.88 | 25.40 | 0.124 |
| Baseline: station, weekday and hour median | 15.65 | 25.08 | 0.146 |

- **The app gives up about 3.4 minutes of accuracy** because it scores a patient with
  only what is known at registration: the arrival hour rather than the service-start
  hour, and no room, visit history or visit position. With an appointment time the model is about 0.9 min
  better than the best simple baseline; without one it is level with it.
- **Likely range:** 65.3% of test waits fall inside P10–P90 (the nominal figure is 80%);
  15.6% fall below and 19.1% above. Median width 26.3 min.
- **Breach flag:** AUC 0.756, precision 0.37, recall 0.61, with 20.7% of test waits
  actually breaching.
- The "typical error" shown beside an estimate is the MAE from the second or third row,
  chosen by whether an appointment time was entered (`models_service.wait_error_min`).

The HMM and LSTM have not been audited inside the app in the same way. The agent
console shows the LSTM's validation MAE from its own metadata (2.11 patients).

## 7. Known gaps

1. **The LSTM and HMM in `Models/` are the earlier models, not the submitted ones.**
   The hand-over note says the files directly under `Models/` come from the legacy
   pipeline and that the submitted LSTM is the one in `per_category_models/`. The app
   still loads the earlier `lstm_queue_forecast.pt` and `clinic_hmms.pkl`. The submitted
   LSTM takes 9 features (it adds `is_holiday`) where the app builds 8, and its metadata
   records a corrected target (true point-in-time queue depth rather than the arrival
   count). Switching therefore needs a change to `lstm_bucket_features` as well as the
   new weights, scaler and metadata — not only a file swap.
2. **Missing inputs cost accuracy** (§6). Capturing the room and the visit position at
   registration would close most of the gap.
3. **Replay, not live.** Every prediction describes a recorded day. Predictions for
   dates outside 2024-10-01 → 2025-09-30 come from an analog day.
4. **The model data and the RAG data cover different periods.** The models and the
   replay end in September 2025; the Qdrant evidence starts in October 2025. The
   LightGBM-versus-RAG agreement check compares estimates from different periods.
5. **Fixed holiday list.** The holiday features use a list ending 2025-12-25. Past that
   date `days_to_next_holiday` is 999. This does not affect the app today, because
   out-of-range dates are replayed from inside the dataset.
6. **No automated tests** cover the loaders or the feature builders (`backend/tests/` is
   empty). A check that `wait_model_features` returns exactly the features listed in
   `deployment_metadata.json` would catch a mismatched model file early.
7. **`torch.load(..., weights_only=False)`** and the pickle and joblib loaders execute
   code from the files they read. This is acceptable for the team's own committed files
   and should not be pointed at files from elsewhere.

## 8. Replacing a model

- **Wait model, same features:** replace the files in
  `Models/3_Trained_Models/deployment_global_model/`, then run
  `python -m scripts.audit_lgbm_global` from `backend/` to regenerate
  `lgbm_app_profile.json` (it needs `models_codes/data/cleaned_events.csv`, which is
  local-only). Restart the backend — loaded models are cached for the life of the process.
- **Wait model, different features:** also update `livestate.wait_model_features` so it
  returns every feature named in the new `deployment_metadata.json`.
- **LSTM:** the feature order in `lstm_bucket_features` must match
  `lstm_queue_metadata.json`, and the layer sizes in the metadata must match the weights.
- **HMM:** `clinic_hmms.pkl` must be a dict keyed by category, each entry holding
  `model`, `state_map` and `means`.
