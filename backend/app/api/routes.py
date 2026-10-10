import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from app.agents.graph import wait_estimate_graph
from app.agents.qa_graph import qa_graph, suggest_followups
from app.agents.qa_personas import PERSONAS, Length, Register, catalogue, resolve_style
from app.agents.supervisor_graph import supervisor_graph
from app.config import settings
from app.schemas import (
    ArrivalRequest,
    DocumentOut,
    EstimateResponse,
    ForecastBucket,
    GoldenSample,
    PatientRegistration,
    QueueSimulationRequest,
    QueueSimulationResponse,
    RegisteredPatient,
    SamplePatient,
    SimilarEvent,
)
from app.services import document_registry, livestate, models_service, qa_feedback, registry, stations, topics
from app.services.golden_samples import GOLDEN_SAMPLES
from app.services.pathway import build_pathway, classify
from app.services.qdrant_service import collection_status, get_client
from app.services.sop_ingest import SUPPORTED_EXTENSIONS, ingest_file
from app.services.sop_service import delete_document, ensure_sop_collection
from app.services.triage import triage

router = APIRouter(prefix="/api/v1")

SGT_BANDS = [(30, "green"), (60, "amber")]  # else red — matches digital-twin thresholds


def band(wait_min: float) -> str:
    for limit, name in SGT_BANDS:
        if wait_min < limit:
            return name
    return "red"


# RAG service_type (Qdrant label) -> trained-model category (Clean_Dataset label).
# The two datasets were built from different exports with different label
# schemes; "other" is a legitimate trained-model bucket, so unmapped service
# types still get a model prediction rather than none at all.
SERVICE_TYPE_TO_CATEGORY: dict[str, str] = {
    "Consultation": "consultation",
    "OCT": "diagnostic_scan",
    "Diagnostic": "diagnostic_scan",
    "VA": "visual_acuity",
    "HVF": "visual_field_test",
    "Biometry": "biometry",
    "Treatment": "treatment",
    "Orthoptic": "other",
    "Financial Con": "other",
    "PAT": "other",
    "Pre Consultation Test": "other",
}


def _run_queue_models(clinic: str, category: str, at: datetime, *,
                      visit_position: int | None = None,
                      appointment_lag_min: float | None = None) -> dict:
    """LightGBM wait estimate + HMM congestion state + LSTM queue-depth
    forecast for one clinic/category at a tz-naive timestamp inside
    livestate.dataset_range(). Shared by /queue/simulate and /patients/register
    so both surfaces run the exact same model pipeline. `appointment_lag_min`
    is arrival minus appointment; None means no appointment was given."""
    depth = livestate.queue_depth(clinic, category, at)
    wait = models_service.predict_wait(category, livestate.wait_model_features(
        clinic, category, at,
        appointment_lag_min=appointment_lag_min, visit_position=visit_position,
    ))
    model_wait = wait["estimate_min"]
    mae = models_service.wait_error_min(has_appointment=appointment_lag_min is not None)

    waits = livestate.recent_waits(clinic, category, at)
    congestion = models_service.congestion_state(category, waits)

    feats = livestate.lstm_bucket_features(clinic, category, at)
    bucket_min = models_service.lstm_meta()["bucket_minutes"]
    if feats is None:
        forecast_vals: list[float] = []
        trend = "no-data"
    else:
        forecast_vals = models_service.forecast_queue_depth(feats)
        current = float(feats[-1][0])
        if not forecast_vals:
            trend = "no-data"
        elif forecast_vals[0] > current * 1.15:
            trend = "rising"
        elif forecast_vals[0] < current * 0.85:
            trend = "falling"
        else:
            trend = "stable"

    return {
        "queue_depth": depth,
        "model_estimate_min": round(model_wait, 1),
        "model_mae_min": mae,
        "model_range_min": (round(wait["p10_min"], 1), round(wait["p90_min"], 1)),
        "breach_limit_min": wait["breach_limit_min"],
        "breach_probability": round(wait["breach_probability"], 2),
        "breach_flag": wait["breach_flag"],
        "congestion_state": congestion["state"],
        "congestion_mean_wait_min": congestion.get("state_mean_wait"),
        "congestion_observations": congestion.get("n_observations", 0),
        "forecast": [
            ForecastBucket(minutes_ahead=bucket_min * (i + 1), queue_depth=v)
            for i, v in enumerate(forecast_vals)
        ],
        "forecast_trend": trend,
    }


@router.post("/arrivals/estimate", response_model=EstimateResponse)
def estimate_arrival(req: ArrivalRequest) -> EstimateResponse:
    local = req.replay_at or req.arrival_datetime
    result = wait_estimate_graph.invoke({
        "clinic": req.clinic,
        "service_type": req.service_type,
        "hour": local.hour,
        "day_of_week": local.weekday(),
        "diagnosis": req.diagnosis,
        "medical_history": req.medical_history,
        "current_issue": req.current_issue,
    })
    if not result.get("hits"):
        raise HTTPException(404, "No historical evidence found — has ingestion run?")

    stats = result["stats"]
    return EstimateResponse(
        estimated_wait_min=stats["median"],
        range_min=(stats["p10"], stats["p90"]),
        confidence=result["confidence"],
        evidence_count=stats["n"],
        filters_widened=result["widenings"],
        similar_events_sample=[
            SimilarEvent(**{k: h.get(k) for k in SimilarEvent.model_fields})
            for h in result["hits"][:10]
        ],
        explanation=result["explanation"],
        status_band=band(stats["median"]),
    )


@router.get("/arrivals/samples", response_model=list[SamplePatient])
def sample_arrivals(n: int = 5) -> list[SamplePatient]:
    n = max(1, min(n, 20))
    points = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        limit=n,
        with_payload=True,
    ).points
    if not points:
        raise HTTPException(404, "No events in collection — has ingestion run?")
    return [
        SamplePatient(
            patient_id=p.payload["patient_id"],
            clinic=p.payload["clinic"],
            service_type=p.payload["service_type"],
            service_point=p.payload["service_point"],
            arrival_iso=p.payload["wait_start_iso"],
            actual_wait_min=p.payload["wait_min"],
        )
        for p in points
    ]


@router.get("/samples/golden", response_model=list[GoldenSample])
def golden_samples() -> list[GoldenSample]:
    """Curated demo patients for the Arrival Simulator and New Patient Entry."""
    return GOLDEN_SAMPLES


def _resolve_replay(requested: datetime, replay_at: datetime | None) -> tuple[datetime, bool, str | None]:
    """(moment to replay clinic state from, whether it differs from the
    request, note explaining it). A golden sample pins the moment; otherwise
    an out-of-range request falls back to its nearest analog."""
    if replay_at is not None:
        pinned = replay_at.replace(tzinfo=None)
        lo, hi = livestate.dataset_range()
        if not lo <= pinned <= hi:
            raise HTTPException(422, f"replay_at must be within {lo.date()} … {hi.date()}")
        return pinned, True, (
            f"Golden sample: clinic state is replayed from "
            f"{pinned.strftime('%a %Y-%m-%d %H:%M')}, a recorded moment in the historical dataset."
        )
    replayed_at, is_analog = livestate.analog_timestamp(requested)
    return replayed_at, is_analog, _replay_note(requested, replayed_at) if is_analog else None


def _replay_note(requested: datetime, replayed_at: datetime) -> str:
    lo, hi = livestate.dataset_range()
    return (
        f"{requested.date()} is outside the trained-model dataset range "
        f"({lo.date()} … {hi.date()}); showing a pattern replay from "
        f"{replayed_at.strftime('%a %Y-%m-%d %H:%M')} — the nearest date with "
        f"the same weekday and time-of-day."
    )


@router.post("/queue/simulate", response_model=QueueSimulationResponse)
def simulate_queue(req: QueueSimulationRequest) -> QueueSimulationResponse:
    """Model-based queue simulation: LightGBM wait estimate + HMM congestion
    state + LSTM queue-depth forecast, driven by the trained models in
    Models/ rather than RAG retrieval. Clinic state is replayed from the
    trained-model dataset (Clean_Dataset); a timestamp outside that dataset's
    range is mapped to the nearest analog (same weekday/time-of-day) instead
    of being rejected — see livestate.analog_timestamp."""
    if req.category not in livestate.CATEGORIES:
        raise HTTPException(422, f"category must be one of {livestate.CATEGORIES}")

    requested = req.as_of.replace(tzinfo=None)
    replayed_at, is_analog, replay_note = _resolve_replay(requested, req.replay_at)

    m = _run_queue_models(
        req.clinic, req.category, replayed_at,
        visit_position=req.visit_position,
        appointment_lag_min=req.appointment_lag_min,
    )
    return QueueSimulationResponse(
        clinic=req.clinic,
        category=req.category,
        as_of=requested,
        replayed_at=replayed_at,
        is_analog_replay=is_analog,
        replay_note=replay_note,
        queue_depth=m["queue_depth"],
        model_estimate_min=m["model_estimate_min"],
        model_mae_min=m["model_mae_min"],
        model_range_min=m["model_range_min"],
        breach_limit_min=m["breach_limit_min"],
        breach_probability=m["breach_probability"],
        breach_flag=m["breach_flag"],
        status_band=band(m["model_estimate_min"]),
        congestion_state=m["congestion_state"],
        congestion_mean_wait_min=m["congestion_mean_wait_min"],
        congestion_observations=m["congestion_observations"],
        forecast=m["forecast"],
        forecast_trend=m["forecast_trend"],
    )


@router.post("/patients/register", response_model=RegisteredPatient)
def register_patient(req: PatientRegistration) -> RegisteredPatient:
    local = req.arrival_datetime
    # a golden sample searches similar cases by its pinned moment's hour and weekday
    rag_at = req.replay_at or local
    result = wait_estimate_graph.invoke({
        "clinic": req.clinic,
        "service_type": req.service_type,
        "hour": rag_at.hour,
        "day_of_week": rag_at.weekday(),
        "diagnosis": req.diagnosis,
        "medical_history": req.medical_history,
        "current_issue": req.current_issue,
    })
    # RAG evidence is supporting context only; the headline estimate comes
    # from the trained model, so a service type with no evidence still registers.
    stats = result["stats"] if result.get("hits") else None
    queue_position = registry.next_queue_position(
        req.clinic, req.service_type, local.date().isoformat()
    )
    triage_result = triage(req.diagnosis, req.medical_history, req.current_issue)
    archetype = classify(req.diagnosis, req.medical_history, req.current_issue,
                         priority=triage_result.priority)
    pathway = build_pathway(archetype)

    # Model-based queue state (LightGBM/HMM/LSTM), same pipeline as
    # /queue/simulate. A timestamp outside the trained-model dataset's range
    # is replayed from the nearest same-weekday/time-of-day analog instead of
    # being skipped — the RAG estimate above always applies regardless, since
    # it only reads hour-of-day/weekday off the timestamp, not the actual
    # calendar date.
    naive_arrival = local.replace(tzinfo=None) if local.tzinfo else local
    model_category = SERVICE_TYPE_TO_CATEGORY.get(req.service_type, "other")
    replayed_at, is_analog, replay_note = _resolve_replay(naive_arrival, req.replay_at)
    # The model's appointment input: minutes the patient arrived after (+) or
    # before (-) their appointment, when one was entered.
    appointment_lag = None
    if req.appointment_datetime is not None:
        appointment_lag = (naive_arrival - req.appointment_datetime.replace(tzinfo=None)).total_seconds() / 60
    model_fields: dict = {}
    try:
        m = _run_queue_models(req.clinic, model_category, replayed_at,
                              appointment_lag_min=appointment_lag)
        model_fields = {
            "model_category": model_category,
            "model_estimate_min": m["model_estimate_min"],
            "model_mae_min": m["model_mae_min"],
            "model_range_min": m["model_range_min"],
            "model_breach_limit_min": m["breach_limit_min"],
            "model_breach_probability": m["breach_probability"],
            "model_breach_flag": m["breach_flag"],
            "model_queue_depth": m["queue_depth"],
            "model_congestion_state": m["congestion_state"],
            "model_congestion_mean_wait_min": m["congestion_mean_wait_min"],
            "model_forecast": m["forecast"],
            "model_forecast_trend": m["forecast_trend"],
            "model_replayed_at": replayed_at,
            "model_is_analog_replay": is_analog,
            "model_replay_note": replay_note,
        }
    except Exception:
        model_fields = {"model_unavailable_reason": "Model prediction failed unexpectedly."}

    # Headline estimate: the LightGBM prediction, with the P10-P90 range from
    # its quantile models. RAG is the fallback only if the model could not run.
    model_est = model_fields.get("model_estimate_min")
    if model_est is not None:
        mae = model_fields["model_mae_min"]
        estimate = model_est
        low, high = model_fields["model_range_min"]
        # the point model and the quantile models are separate, so keep the
        # headline inside the range shown beside it
        estimate_range = (min(low, round(model_est, 1)), max(high, round(model_est, 1)))
        explanation = (
            f"Predicted by the LightGBM wait-time model for {model_category.replace('_', ' ')} "
            f"at {req.clinic} on {replayed_at:%A} at {replayed_at:%H:%M}, with "
            f"{model_fields['model_queue_depth']} patient(s) already waiting: about "
            f"{model_est:.0f} minutes, likely between {estimate_range[0]:.0f} and "
            f"{estimate_range[1]:.0f} minutes (typical error ±{mae:.0f} minutes)."
        )
        if appointment_lag is None:
            explanation += " No appointment time was entered, so a typical one was assumed."
        if stats:
            explanation += (
                f" For comparison, the {stats['n']} most similar historical cases had a "
                f"median wait of {stats['median']:.0f} minutes."
            )
    elif stats:
        estimate = stats["median"]
        estimate_range = (stats["p10"], stats["p90"])
        explanation = result["explanation"]
    else:
        raise HTTPException(404, "No model prediction or historical evidence available.")

    patient = RegisteredPatient(
        patient_id=req.patient_id or random.randint(300_000_000, 399_999_999),
        display_name=req.display_name,
        clinic=req.clinic,
        service_type=req.service_type,
        diagnosis=req.diagnosis,
        medical_history=req.medical_history,
        current_issue=req.current_issue,
        arrival_datetime=req.arrival_datetime,
        appointment_datetime=req.appointment_datetime,
        queue_position=queue_position,
        priority=triage_result.priority,
        priority_label=triage_result.label,
        triage_reasons=triage_result.reasons,
        pathway_label=pathway["label"],
        pathway=pathway["steps"],
        total_visit_min=pathway["total_visit_min"],
        estimated_wait_min=estimate,
        range_min=estimate_range,
        status_band=band(estimate),
        explanation=explanation,
        registered_at=datetime.now(timezone.utc),
        rag_estimate_min=stats["median"] if stats else None,
        rag_range_min=(stats["p10"], stats["p90"]) if stats else None,
        rag_evidence_n=stats["n"] if stats else None,
        **model_fields,
    )
    registry.add(patient)
    return patient


def _today_start() -> datetime:
    """Start of the server-local day — the cut-off for "registered today"."""
    return datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)


@router.get("/patients", response_model=list[RegisteredPatient])
def list_patients() -> list[RegisteredPatient]:
    """Patients registered today (server-local day), newest first."""
    return registry.list_all(since=_today_start())


@router.get("/stations/overview")
def stations_overview() -> dict:
    """Per-station historical stats + today's registered counts, grouped by clinic."""
    overview = stations.station_overview(get_client())
    counts = registry.counts_since(_today_start())
    return {
        clinic: [
            {**s, "registered_today": counts.get((clinic, s["service_type"]), 0)}
            for s in station_list
        ]
        for clinic, station_list in overview.items()
    }


@router.get("/stations/weekly")
def stations_weekly() -> dict:
    """Per-station historical wait pattern by day of week + today's registered
    counts. Stations here are trained-model categories, so registrations are
    rolled up from their service type."""
    weekly = stations.weekly_profile()
    counts: dict[tuple[str, str], int] = {}
    for (clinic, service_type), n in registry.counts_since(_today_start()).items():
        key = (clinic, SERVICE_TYPE_TO_CATEGORY.get(service_type, "other"))
        counts[key] = counts.get(key, 0) + n
    return {
        **weekly,
        "clinics": {
            clinic: [
                {**s, "registered_today": counts.get((clinic, s["category"]), 0)}
                for s in station_list
            ]
            for clinic, station_list in weekly["clinics"].items()
        },
    }


class AgentAnalyzeRequest(BaseModel):
    clinic: str = Field(..., examples=["TTSH Eye Centre", "Clinic 1A"])
    category: str = Field(..., examples=["consultation", "visual_acuity", "diagnostic_scan"])
    as_of: datetime = Field(..., description="Timestamp within the cleaned dataset range")
    objective: str = "Analyse bottlenecks and suggest resource optimization"


@router.post("/agent/analyze")
def agent_analyze(req: AgentAnalyzeRequest) -> dict:
    """Run the full supervisor multi-agent analysis (Figure 2 architecture).
    A requested `as_of` outside the trained-model dataset's range is replayed
    from the nearest same-weekday/time-of-day analog instead of being
    rejected — see livestate.analog_timestamp."""
    requested = req.as_of.replace(tzinfo=None)
    replayed_at, is_analog = livestate.analog_timestamp(requested)
    result = supervisor_graph.invoke({
        "clinic": req.clinic,
        "category": req.category,
        "as_of": replayed_at,
        "objective": req.objective,
    })
    report = result["report"]
    report["requested_as_of"] = requested.isoformat()
    report["is_analog_replay"] = is_analog
    report["replay_note"] = _replay_note(requested, replayed_at) if is_analog else None
    return report


# Process-window labels for the supervisor graph's nodes: (who acts, what it does).
AGENT_STEP_LABELS = {
    "supervisor": ("Supervisor", "Parse the objective and delegate"),
    "waiting_time_agent": ("Specialist", "Waiting Time Analysis — LightGBM model and RAG evidence"),
    "bottleneck_agent": ("Specialist", "Bottleneck Detection — HMM congestion states"),
    "queue_depth_forecast_agent": ("Specialist", "Queue Depth Forecasting — LSTM next-hour forecast"),
    "resource_planning_agent": ("Specialist", "Resource Planning — allocation options"),
    "clarify": ("Supervisor", "Ask the user to clarify the objective"),
    "supervisor_review": ("Supervisor", "Collect the specialists' results"),
    "predictive_agent": ("Specialist", "Predictive Analytics — projected wait and risk"),
    "supervisor_integrate": ("Supervisor", "Integrate the findings"),
    "supervisor_validate": ("Supervisor", "Validate the result against the objective"),
    "decision_support": ("Decision support", "Rank recommendations"),
    "reporting": ("Reporting", "Write the report summary"),
}


def _agent_step_details(node: str, update: dict) -> list[str]:
    """What a finished supervisor-graph node produced, as short lines."""
    if update.get("trace") and "report" not in update:
        return [f"{t['action']}: {t['detail']}" for t in update["trace"]]
    f = update.get("findings") or {}
    if wt := f.get("waiting_time"):
        rag = (f"RAG median {wt['rag_median_min']} min from {wt['rag_evidence_n']} similar events"
               if wt["rag_evidence_n"] else "no similar RAG events")
        return [f"LightGBM estimate {wt['model_estimate_min']} min; {rag}; "
                f"{wt['current_queue_depth']} in queue"]
    if bn := f.get("bottlenecks"):
        return [f"{len(bn['stations'])} stations checked, {len(bn['flagged'])} flagged as bottlenecks"]
    if fc := f.get("forecast"):
        depths = ", ".join(str(v) for v in fc["next_hour_queue_depth"]) or "no data"
        return [f"Queue trend {fc['trend']}; next hour {depths}"]
    if rp := f.get("resource_planning"):
        return [f"{len(rp['options'])} resource option(s) proposed"]
    if pa := f.get("predictive_analysis"):
        lo, hi = pa["predicted_wait_range_min"]
        risk = f"; risk {pa['risk_label']}" if pa["risk_label"] else ""
        return [f"Projected wait {pa['predicted_wait_min']} min (range {lo:.0f}–{hi:.0f}); "
                f"outlook {pa['outlook']}{risk}"]
    if "decisions" in update:
        return [f"{len(update['decisions'])} recommendation(s) ranked"]
    if node == "reporting":
        return [f"{len(update['report']['alerts'])} alert(s) raised; report ready"]
    return []


@router.post("/agent/analyze/stream")
def agent_analyze_stream(req: AgentAnalyzeRequest) -> StreamingResponse:
    """Same analysis as /agent/analyze, streamed as Server-Sent Events so the
    console can show each step while it runs. Events (JSON in `data:`):
      {"type": "step_start", node, actor, label}
      {"type": "step_done", node, elapsed_s, details: [...]}
      {"type": "report", "report": {...}}   — final, same shape as /agent/analyze
      {"type": "error", "message": "..."}"""
    requested = req.as_of.replace(tzinfo=None)
    replayed_at, is_analog = livestate.analog_timestamp(requested)
    state = {"clinic": req.clinic, "category": req.category, "as_of": replayed_at,
             "objective": req.objective}

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def events():
        started: dict[str, float] = {}
        report = None
        try:
            for mode, data in supervisor_graph.stream(state, stream_mode=["tasks", "updates"]):
                if mode == "tasks":
                    if "result" not in data:  # a node starting; its finish arrives as an update
                        node = data["name"]
                        started[node] = time.perf_counter()
                        actor, label = AGENT_STEP_LABELS.get(node, ("Agent", node))
                        yield sse({"type": "step_start", "node": node, "actor": actor, "label": label})
                    continue
                for node, update in data.items():
                    if not isinstance(update, dict):
                        continue
                    report = update.get("report", report)
                    yield sse({
                        "type": "step_done", "node": node,
                        "elapsed_s": round(time.perf_counter() - started.get(node, time.perf_counter()), 1),
                        "details": _agent_step_details(node, update),
                    })
            report["requested_as_of"] = requested.isoformat()
            report["is_analog_replay"] = is_analog
            report["replay_note"] = _replay_note(requested, replayed_at) if is_analog else None
            yield sse({"type": "report", "report": report})
        except Exception as e:  # surface to the client; the HTTP status is already 200
            yield sse({"type": "error", "message": str(e)[:300]})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/models/wait-time")
def wait_time_model_card() -> dict:
    """Model card for the LightGBM wait-time model: the data it learned from,
    its inputs, and its hold-out accuracy as this app calls it (see
    scripts/audit_lgbm_global.py), next to the simple baselines."""
    meta = models_service.wait_meta()
    profile = models_service.wait_profile()
    return {
        "name": "LightGBM wait-time model",
        "algorithm": "Gradient-boosted decision trees (LightGBM)",
        "features": ["service_category", *meta["features"]],
        # inputs the app has no value for and fills with the station's typical value
        "defaulted_features": ["room_number_encoded", "room_queue_depth", "prior_visit_count",
                               "visit_position"],
        "train": profile["train"],
        "test": profile["test"],
        "evaluation": profile["evaluation"],
        "range": profile["range"],
        "breach": {**profile["breach"], "limits_min": meta["breach_thresholds_min"]},
        "baselines": profile["baselines"],
    }


@router.get("/agent/dataset-range")
def agent_dataset_range() -> dict:
    lo, hi = livestate.dataset_range()
    return {"min": lo.isoformat(), "max": hi.isoformat(),
            "categories": livestate.CATEGORIES}


class QARequest(BaseModel):
    # Any non-blank text: short greetings like "hi" are normal chat input.
    question: str = Field(..., max_length=1000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Please type a question.")
        return v
    # All optional: unset persona -> QA_DEFAULT_PERSONA; unset style -> persona's defaults.
    persona: str | None = None
    length: Length | None = None
    technical_level: Register | None = None

    def graph_input(self) -> dict:
        """Initial graph state with persona/style resolved (unknown persona -> 422)."""
        from langchain_core.messages import HumanMessage

        if self.persona is not None and self.persona not in PERSONAS:
            raise HTTPException(422, f"Unknown persona '{self.persona}'. "
                                     f"Options: {', '.join(PERSONAS)}")
        persona, length, register = resolve_style(
            self.persona or settings.qa_default_persona, self.length, self.technical_level)
        return {"messages": [HumanMessage(self.question)], "citations": [],
                "persona": persona.id, "length": length, "register": register}


def _style_info(state: dict) -> dict:
    return {"persona": state["persona"], "persona_name": PERSONAS[state["persona"]].name,
            "length": state["length"], "technical_level": state["register"]}


class QAFeedback(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message_id: str = Field(..., min_length=1, max_length=100)
    # None clears a previous rating (the user un-clicked the thumb).
    rating: Literal["up", "down"] | None
    reason: str | None = Field(None, max_length=200)
    question: str = Field("", max_length=2000)
    answer: str = Field("", max_length=20000)
    persona: str | None = None


@router.post("/qa/feedback", status_code=204)
def qa_feedback_submit(fb: QAFeedback) -> None:
    """Thumbs up/down on an answer (one rating per answer; re-rating overwrites)."""
    if fb.rating is None:
        qa_feedback.clear(fb.session_id, fb.message_id)
    else:
        qa_feedback.upsert(fb.session_id, fb.message_id, fb.rating, fb.reason,
                           fb.question, fb.answer, fb.persona)


@router.get("/qa/feedback/summary")
def qa_feedback_summary() -> dict:
    """Totals and per-persona counts, plus recent thumbs-down reasons."""
    return qa_feedback.summary()


@router.get("/qa/personas")
def qa_personas() -> dict:
    """Personas and answer-style options for the Q&A page's pickers."""
    return catalogue(settings.qa_default_persona)


@router.post("/qa/ask")
def qa_ask(req: QARequest) -> dict:
    """SOP & Healthcare Q&A: a tool-calling LangGraph agent searches the
    `mediq_sop_docs` Qdrant collection and answers grounded in what it finds.
    Unlike the other graphs, there is no template fallback here — open-ended
    reasoning over retrieved text requires an LLM."""
    if not settings.openai_api_key:
        raise HTTPException(503, "SOP Q&A requires OPENAI_API_KEY to be configured.")

    from langchain_core.messages import AIMessage, ToolMessage

    state = req.graph_input()
    result = qa_graph.invoke(state)
    # .text flattens Responses-API content blocks into a plain string
    answer = str(result["messages"][-1].text)
    passages = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    covered, sources = _coverage(result.get("citations", []))

    return {
        "status": "ok",
        "question": req.question,
        **_style_info(state),
        "answer": answer,
        "covered": covered,
        "sources": sources,
        "thinking": _qa_thinking(m for m in result["messages"] if isinstance(m, AIMessage)),
        # Not covered: the UI offers the knowledge-base topics instead.
        "followups": suggest_followups(req.question, answer, passages, state["persona"]) if covered else [],
    }


@router.post("/qa/ask/stream")
def qa_ask_stream(req: QARequest) -> StreamingResponse:
    """Same agent as /qa/ask, streamed as Server-Sent Events so the answer renders
    as it is generated. Events (JSON in `data:`):
      {"type": "thinking", "step": {kind, text}}  — a search or reasoning summary
      {"type": "token", "text": "..."}            — answer text delta
      {"type": "start", persona, persona_name, length, technical_level} — resolved style, first
      {"type": "done", answer, covered, sources, thinking} — final, authoritative result
      {"type": "followups", "questions": [...]}   — suggested next questions, after done
      {"type": "error", "message": "..."}"""
    if not settings.openai_api_key:
        raise HTTPException(503, "SOP Q&A requires OPENAI_API_KEY to be configured.")

    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

    state = req.graph_input()  # before streaming starts, so a bad persona is a real 422

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def events():
        citations: list[dict] = []
        thinking: list[dict] = []
        passages: list[str] = []
        answer = ""
        try:
            yield sse({"type": "start", **_style_info(state)})
            for mode, data in qa_graph.stream(state, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    chunk, meta = data
                    if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk):
                        text = str(chunk.text)
                        if text:
                            yield sse({"type": "token", "text": text})
                    continue
                # "updates": {node: update}; ToolNode's Command updates arrive as a list
                for node, update in data.items():
                    for upd in update if isinstance(update, list) else [update]:
                        if not isinstance(upd, dict):
                            continue
                        citations += upd.get("citations", [])
                        passages += [str(m.content) for m in upd.get("messages", [])
                                     if isinstance(m, ToolMessage)]
                        ai_msgs = [m for m in upd.get("messages", []) if isinstance(m, AIMessage)]
                        for step in _qa_thinking(ai_msgs):
                            thinking.append(step)
                            yield sse({"type": "thinking", "step": step})
                        if node == "agent" and ai_msgs and not ai_msgs[-1].tool_calls:
                            answer = str(ai_msgs[-1].text)
            covered, sources = _coverage(citations)
            yield sse({"type": "done", "answer": answer, "covered": covered,
                       "sources": sources, "thinking": thinking})
            # After "done" so the answer never waits on suggestions. Not covered:
            # the UI offers the knowledge-base topics instead.
            followups = suggest_followups(req.question, answer, passages, state["persona"]) if covered else []
            if followups:
                yield sse({"type": "followups", "questions": followups})
        except Exception as e:  # surface to the client; the HTTP status is already 200
            yield sse({"type": "error", "message": str(e)[:300]})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _coverage(citations: list[dict]) -> tuple[bool, list[dict]]:
    """Whether the knowledge base covers the question: the best reranked passage
    across every search must clear QA_COVERED_MIN_SCORE. Not covered -> no
    sources, since the passages retrieved are only the least-bad matches."""
    best = max((c.get("score", float("-inf")) for c in citations), default=float("-inf"))
    covered = best >= settings.qa_covered_min_score
    return covered, _dedupe_sources(citations) if covered else []


def _dedupe_sources(citations: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    sources = []
    for c in citations:
        key = (c["title"], c["snippet"])
        if key not in seen:
            seen.add(key)
            sources.append(c)
    return sources


def _qa_thinking(ai_messages) -> list[dict]:
    """The agent's visible trail, in order: reasoning summaries (only emitted at
    high reasoning effort) and each knowledge-base search it ran."""
    steps: list[dict] = []
    for msg in ai_messages:
        blocks = msg.content if isinstance(msg.content, list) else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                for part in block.get("summary") or []:
                    if part.get("text", "").strip():
                        steps.append({"kind": "reasoning", "text": part["text"].strip()})
        for call in msg.tool_calls:
            query = call.get("args", {}).get("query", "")
            steps.append({"kind": "search", "text": f'Searched SOP knowledge base: "{query}"'})
    return steps


def _document_out(row) -> DocumentOut:
    return DocumentOut(
        id=row["id"], name=row["name"], ext=row["ext"], size_bytes=row["size_bytes"],
        status=row["status"], chunk_count=row["chunk_count"] or 0,
        error_message=row["error_message"], uploaded_at=row["uploaded_at"],
        ingested_at=row["ingested_at"],
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    """Documents in the SOP knowledge base, newest first — powers the document
    management page (upload/status/remove)."""
    return [_document_out(r) for r in document_registry.list_all()]


@router.post("/documents/upload", response_model=DocumentOut, status_code=201)
def upload_document(file: UploadFile = File(...)) -> DocumentOut:
    """Upload one SOP document (PDF/DOCX/TXT) and ingest it immediately into
    the `mediq_sop_docs` Qdrant collection."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: pdf, docx, txt")

    settings.sop_docs_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid.uuid4())
    dest = settings.sop_docs_dir / f"{document_id}{ext}"
    contents = file.file.read()
    dest.write_bytes(contents)

    document_registry.add(document_id, file.filename or dest.name, str(dest.resolve()),
                          ext, len(contents), status="processing")

    try:
        client = get_client()
        ensure_sop_collection(client)
        n = ingest_file(client, document_id, Path(file.filename or dest.name).stem, dest)
        document_registry.update_status(document_id, "ingested", chunk_count=n)
    except Exception as exc:
        document_registry.update_status(document_id, "failed", error_message=str(exc))
    else:
        topics.ensure_topics()  # one LLM call unless it duplicates an existing document

    return _document_out(document_registry.get(document_id))


@router.post("/documents/{document_id}/reingest", response_model=DocumentOut)
def reingest_document(document_id: str) -> DocumentOut:
    """Retry ingestion for a document stuck in `failed` (or refresh an existing one)."""
    row = document_registry.get(document_id)
    if not row:
        raise HTTPException(404, "Document not found")

    path = Path(row["file_path"])
    if not path.exists():
        document_registry.update_status(document_id, "failed",
                                        error_message="Source file no longer on disk.")
        raise HTTPException(404, "Source file no longer on disk — re-upload instead.")

    try:
        client = get_client()
        ensure_sop_collection(client)
        delete_document(client, document_id)
        n = ingest_file(client, document_id, path.stem, path)
        document_registry.update_status(document_id, "ingested", chunk_count=n)
    except Exception as exc:
        document_registry.update_status(document_id, "failed", error_message=str(exc))
    else:
        document_registry.clear_topic(document_id)
        topics.ensure_topics()

    return _document_out(document_registry.get(document_id))


def _remove_document(document_id: str) -> None:
    row = document_registry.get(document_id)
    if not row:
        raise HTTPException(404, "Document not found")
    delete_document(get_client(), document_id)
    document_registry.delete(document_id)
    # Bulk ingestion can register one file under several rows: only delete the
    # file once no remaining row points at it.
    path = Path(row["file_path"])
    still_used = any(Path(r["file_path"]) == path for r in document_registry.list_all())
    if path.exists() and not still_used:
        path.unlink()


@router.delete("/documents/{document_id}", status_code=204)
def delete_document_endpoint(document_id: str) -> None:
    """Remove a document's chunks from Qdrant, its file on disk, and its registry row."""
    _remove_document(document_id)


@router.get("/qa/topics")
def qa_topics() -> dict:
    """What the SOP assistant can answer: one topic per distinct document
    (identical uploads grouped), with a summary and example questions."""
    return {"topics": topics.catalogue()}


def _topic_or_404(topic_id: str) -> dict:
    topic = next((t for t in topics.catalogue() if t["id"] == topic_id), None)
    if not topic:
        raise HTTPException(404, "Topic not found")
    return topic


@router.post("/documents/topics/{topic_id}/regenerate")
def regenerate_topic(topic_id: str) -> dict:
    """Write a fresh topic card for one group from its document text."""
    _topic_or_404(topic_id)
    topics.ensure_topics(regenerate_group=topic_id)
    return _topic_or_404(topic_id)


@router.post("/documents/topics/{topic_id}/merge")
def merge_duplicates(topic_id: str) -> dict:
    """Keep one document of a duplicate group (the human-named copy, else the
    oldest) and remove the identical copies from Qdrant, disk and the registry."""
    _topic_or_404(topic_id)
    keep, remove = topics.keep_document(topic_id)
    for document_id in remove:
        _remove_document(document_id)
    return {"kept": keep, "removed": remove}


@router.get("/collection/status")
def qdrant_status() -> dict:
    return collection_status(get_client())
