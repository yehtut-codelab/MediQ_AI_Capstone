from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ClinicalContext(BaseModel):
    """Free-text presentation used for semantic matching against historical cases."""
    diagnosis: str | None = Field(None, max_length=500,
                                  examples=["Primary open-angle glaucoma, both eyes"])
    medical_history: str | None = Field(None, max_length=1000,
                                        examples=["Diabetic, cataract surgery 2023"])
    current_issue: str | None = Field(None, max_length=1000,
                                      examples=["Blurred vision and eye pressure since last week"])


class ArrivalRequest(ClinicalContext):
    clinic: str = Field(..., examples=["TTSH Eye Centre", "Clinic 1A"])
    service_type: str = Field(..., examples=["Consultation", "OCT", "VA"])
    arrival_datetime: datetime
    appointment_datetime: datetime | None = None


class SimilarEvent(BaseModel):
    score: float
    rerank_score: float | None = None
    clinic: str
    service_type: str
    service_point: str
    wait_min: float
    wait_start_iso: str


class EstimateResponse(BaseModel):
    estimated_wait_min: float
    range_min: tuple[float, float]
    confidence: Literal["high", "medium", "low"]
    evidence_count: int
    filters_widened: int
    similar_events_sample: list[SimilarEvent]
    explanation: str
    status_band: Literal["green", "amber", "red"]


class SamplePatient(BaseModel):
    patient_id: int
    clinic: str
    service_type: str
    service_point: str
    arrival_iso: str
    actual_wait_min: float


class PatientRegistration(ClinicalContext):
    patient_id: int | None = Field(None, description="Auto-generated when omitted")
    display_name: str | None = Field(None, max_length=80,
                                     description="Demo only — kept in memory, never persisted")
    clinic: str = Field(..., examples=["TTSH Eye Centre", "Clinic 1A"])
    service_type: str = Field(..., examples=["Consultation", "OCT", "VA"])
    arrival_datetime: datetime
    appointment_datetime: datetime | None = None


class PathwayStep(BaseModel):
    station: str
    eta_offset_min: float
    expected_wait_min: float
    expected_service_min: float
    note: str | None = None


class ForecastBucket(BaseModel):
    minutes_ahead: int
    queue_depth: float


class RegisteredPatient(BaseModel):
    patient_id: int
    display_name: str | None
    clinic: str
    service_type: str
    diagnosis: str | None = None
    medical_history: str | None = None
    current_issue: str | None = None
    arrival_datetime: datetime
    appointment_datetime: datetime | None
    queue_position: int
    priority: int = 3
    priority_label: str = "P3 — Routine"
    triage_reasons: list[str] = []
    pathway_label: str | None = None
    pathway: list[PathwayStep] = []
    total_visit_min: float | None = None
    estimated_wait_min: float
    range_min: tuple[float, float]
    status_band: Literal["green", "amber", "red"]
    explanation: str
    registered_at: datetime

    # RAG comparison shown beside the model prediction on a fresh registration
    # (similar historical cases). Not persisted, so absent from GET /patients.
    rag_estimate_min: float | None = None
    rag_range_min: tuple[float, float] | None = None
    rag_evidence_n: int | None = None

    # model-based queue state (XGBoost / HMM / LSTM), reconstructed from the
    # trained-model dataset. When the arrival time falls outside that
    # dataset's range, clinic state is replayed from the nearest date with
    # the same weekday/time-of-day instead (see livestate.analog_timestamp) —
    # model_is_analog_replay + model_replay_note explain that when it happens.
    # model_unavailable_reason is reserved for genuine prediction failures.
    model_category: str | None = None
    model_estimate_min: float | None = None
    model_mae_min: float | None = None
    model_queue_depth: int | None = None
    model_congestion_state: Literal["Low", "Medium", "High", "unknown"] | None = None
    model_congestion_mean_wait_min: float | None = None
    model_forecast: list[ForecastBucket] = []
    model_forecast_trend: Literal["rising", "falling", "stable", "no-data"] | None = None
    model_replayed_at: datetime | None = None
    model_is_analog_replay: bool = False
    model_replay_note: str | None = None
    model_unavailable_reason: str | None = None


class DocumentOut(BaseModel):
    id: str
    name: str
    ext: str
    size_bytes: int
    status: Literal["uploaded", "processing", "ingested", "failed"]
    chunk_count: int = 0
    error_message: str | None = None
    uploaded_at: datetime
    ingested_at: datetime | None = None


class StationStats(BaseModel):
    service_type: str
    count: int
    median_wait_min: float
    p75_wait_min: float
    p90_wait_min: float


class QueueSimulationRequest(BaseModel):
    """Model-based queue simulation — reconstructs clinic state from the
    trained-model dataset (Clean_Dataset), distinct from the RAG corpus.
    `as_of` outside the dataset's range is replayed from the nearest date
    with the same weekday/time-of-day (see livestate.analog_timestamp)
    rather than rejected."""
    clinic: str = Field(..., examples=["TTSH Eye Centre", "Clinic 1A"])
    category: str = Field(..., examples=["consultation", "visual_acuity", "diagnostic_scan"])
    as_of: datetime = Field(..., description="Any timestamp — mapped to an analog if out of range")
    visit_position: int | None = Field(None, description="Defaults to the category's typical median")
    visit_length: int | None = Field(None, description="Defaults to the category's typical median")
    appointment_lag_min: float = 0.0


class QueueSimulationResponse(BaseModel):
    clinic: str
    category: str
    as_of: datetime
    replayed_at: datetime
    is_analog_replay: bool
    replay_note: str | None
    queue_depth: int
    model_estimate_min: float
    model_mae_min: float
    status_band: Literal["green", "amber", "red"]
    congestion_state: Literal["Low", "Medium", "High", "unknown"]
    congestion_mean_wait_min: float | None
    congestion_observations: int
    forecast: list[ForecastBucket]
    forecast_trend: Literal["rising", "falling", "stable", "no-data"]
