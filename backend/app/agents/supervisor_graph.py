"""Supervisor/Orchestrator multi-agent graph — implements the Agentic AI Framework
from MediQ_AI_Agents_Architecture.png (Figure 2).

   supervisor ──────────── delegates ───┬─ waiting_time_agent ────────┐
   (parses the objective,               ├─ bottleneck_agent ──────────┤
    decomposes it into tasks)           ├─ queue_depth_forecast_agent ┤
                                        └─ resource_planning_agent ───┤
   supervisor_review ◄──────────── results return ────────────────────┘
   (checks who reported, decides
    the second delegation) ─────────────── predictive_agent ──┐
   supervisor_integrate ◄─────────────────────────────────────┘
   (fuses the findings)
   supervisor_validate
   (checks the result against the objective)
   decision_support ─ reporting ─ END

The supervisor is one agent that acts in four phases. It reads the user's
objective and delegates only to the specialists that objective needs (see
plan_for_objective); they run in parallel and report back to it. It then
decides whether Predictive Analytics can run on what came back, integrates all
findings, and validates the result against the objective. If it cannot tell
what the objective asks for, it delegates to nobody and returns a clarifying
question instead (see clarify). Each phase appends to `trace`, the record of
what the supervisor did. Trained models: LightGBM
(waiting time), per-category HMMs (bottlenecks), LSTM (queue forecast). All
numbers are computed in code; the optional LLM only phrases the report.
"""

import operator
import statistics
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.services import livestate, models_service
from app.services.embedder import embed_one
from app.services.preprocess import DAY_NAMES, daypart
from app.services.qdrant_service import build_filter, get_client, search_similar

# cleaned-dataset categories → raw Qdrant service_type labels (for RAG evidence)
CATEGORY_TO_SERVICE_TYPE = {
    "consultation": "Consultation",
    "visual_acuity": "VA",
    "diagnostic_scan": "OCT",
    "visual_field_test": "HVF",
    "biometry": "Biometry",
    "treatment": "Treatment",
}


SPECIALISTS = {
    "waiting_time_agent": "Waiting Time Analysis",
    "bottleneck_agent": "Bottleneck Detection",
    "queue_depth_forecast_agent": "Queue Depth Forecasting",
    "resource_planning_agent": "Resource Planning",
}
PREDICTIVE = "predictive_agent"
PREDICTIVE_LABEL = "Predictive Analytics"
# predictive_agent projects the wait from the LightGBM estimate and the LSTM trend
PREDICTIVE_NEEDS = ("waiting_time_agent", "queue_depth_forecast_agent")

# Words in an objective that call for each specialist. The console offers three
# preset objectives, but any free-text objective is decomposed the same way.
INTENT_KEYWORDS = {
    "waiting_time_agent": ("wait", "delay", "how long"),
    "bottleneck_agent": ("bottleneck", "congest", "constraint", "busy", "crowd"),
    "queue_depth_forecast_agent": ("forecast", "queue", "trend", "next hour", "outlook"),
    "resource_planning_agent": ("resource", "staff", "allocat", "capacity", "optimi"),
}
FULL_KEYWORDS = ("full", "overall", "review", "everything", "all agents")

DEFAULT_OBJECTIVE = "Analyse bottlenecks and suggest resource optimization"

# Objectives the supervisor knows how to decompose; offered as the choices when
# it has to ask the user what they meant. The ids match the console's presets.
PRESET_OBJECTIVES = [
    {"id": "bottlenecks", "title": "Bottlenecks & resources", "objective": DEFAULT_OBJECTIVE},
    {"id": "waiting", "title": "Waiting time & forecast",
     "objective": "Estimate waiting time and forecast the queue"},
    {"id": "full", "title": "Full clinic review", "objective": "Full clinic performance review"},
]


def plan_for_objective(objective: str) -> dict:
    """Decompose the objective into the specialists to delegate to."""
    text = (objective or "").lower()
    matched = {
        agent: next((k for k in words if k in text), None)
        for agent, words in INTENT_KEYWORDS.items()
    }
    full = next((k for k in FULL_KEYWORDS if k in text), None)
    recognised = bool(full) or any(matched.values())
    if full:
        chosen = list(SPECIALISTS)
    else:                       # nothing matched → nobody is chosen, the supervisor asks
        chosen = [a for a in SPECIALISTS if matched[a]]
    use_predictive = all(a in chosen for a in PREDICTIVE_NEEDS)

    tasks = []
    for agent, label in SPECIALISTS.items():
        if agent in chosen:
            why = (f'objective mentions "{matched[agent]}"' if matched[agent] and not full
                   else f'objective asks for a "{full}" analysis')
        else:
            why = "not needed for this objective" if recognised else "waiting for clarification"
        tasks.append({"agent": agent, "label": label, "delegated": agent in chosen, "reason": why})
    tasks.append({
        "agent": PREDICTIVE, "label": PREDICTIVE_LABEL, "delegated": use_predictive,
        "reason": ("combines the waiting-time and queue-forecast findings" if use_predictive
                   else "needs both the waiting-time and queue-forecast findings" if recognised
                   else "waiting for clarification"),
    })
    return {
        "objective": objective,
        "recognised": recognised,
        "needs_clarification": not recognised,
        "specialists": chosen,
        "predictive": use_predictive,
        "tasks": tasks,
    }


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class SupervisorState(TypedDict, total=False):
    # user request
    clinic: str
    category: str
    as_of: datetime
    objective: str
    # supervisor's decomposition of the objective
    plan: dict
    # patient context for the wait estimate
    visit_position: int | None
    appointment_lag_min: float | None
    # specialist outputs merge here (parallel fan-in)
    findings: Annotated[dict[str, Any], _merge]
    # what the supervisor did, in order (each phase appends)
    trace: Annotated[list[dict], operator.add]
    # supervisor's integration and validation of the findings
    fused: dict
    critique: dict
    goal: dict
    decisions: list[dict]
    report: dict


FINDING_KEY = {
    "waiting_time_agent": "waiting_time",
    "bottleneck_agent": "bottlenecks",
    "queue_depth_forecast_agent": "forecast",
    "resource_planning_agent": "resource_planning",
    PREDICTIVE: "predictive_analysis",
}


def _step(phase: str, action: str, detail: str) -> dict:
    return {"phase": phase, "action": action, "detail": detail}


# ── Supervisor, phase 1: parse the objective, decompose it, delegate ─────

def supervisor(state: SupervisorState) -> SupervisorState:
    # no objective at all → the default; a blank or unclear one → ask the user
    objective = DEFAULT_OBJECTIVE if state.get("objective") is None else state["objective"].strip()
    plan = plan_for_objective(objective)
    names = [SPECIALISTS[a] for a in plan["specialists"]]
    if plan["needs_clarification"]:
        second = _step("Clarify", "Asked the user to clarify the objective",
                       "it names nothing the specialist agents can analyse, so none was delegated to")
    else:
        second = _step("Delegate", f"Delegated to {len(names)} specialist agent(s)", ", ".join(names))
    # None lets the wait model fall back to the station's typical value
    return {
        "plan": plan,
        "visit_position": state.get("visit_position"),
        "appointment_lag_min": state.get("appointment_lag_min"),
        "findings": {},
        "trace": [
            _step("Plan", "Parsed the objective",
                  f'"{objective}"' if objective else "(no objective given)"),
            second,
        ],
    }


def delegate(state: SupervisorState) -> list[str]:
    """Conditional fan-out: only the specialists the plan calls for — or the
    clarification step when the supervisor could not decompose the objective."""
    if state["plan"]["needs_clarification"]:
        return ["clarify"]
    return state["plan"]["specialists"]


# ── Supervisor: feedback & clarification ─────────────────────────────────

def clarify(state: SupervisorState) -> SupervisorState:
    """Ends the run with a question for the user instead of a report. Nothing
    was analysed, so there are no findings, recommendations or alerts."""
    plan = state["plan"]
    asked = f'"{plan["objective"]}"' if plan["objective"] else "an empty request"
    question = (f"I could not tell what {asked} is asking me to analyse. "
                "Which of these do you want?")
    return {"report": {
        "summary": question,
        "as_of": state["as_of"].isoformat(),
        "clinic": state["clinic"],
        "category": state["category"],
        "plan": plan,
        "trace": state["trace"],
        "clarification": {
            "question": question,
            "options": PRESET_OBJECTIVES,
            "hint": "You can also rephrase the request using words such as wait, bottleneck, "
                    "queue forecast, staffing or resources.",
        },
        "goal": None,
        "consensus_wait_min": None,
        "confidence": None,
        "findings": {},
        "recommendations": [],
        "alerts": [],
    }}


# ── Supervisor, phase 2: collect the specialists' results, delegate again ──

def supervisor_review(state: SupervisorState) -> SupervisorState:
    """The delegated specialists report back here. The supervisor records who
    reported, then decides the second delegation: Predictive Analytics runs
    only if the findings it builds on actually came back."""
    plan, f = state["plan"], state["findings"]
    asked = plan["specialists"]
    reported = [a for a in asked if FINDING_KEY[a] in f]
    silent = [SPECIALISTS[a] for a in asked if a not in reported]
    trace = [_step(
        "Monitor", f"Received results from {len(reported)} of {len(asked)} specialist agent(s)",
        ", ".join(SPECIALISTS[a] for a in reported) + (f"; no result from {', '.join(silent)}" if silent else ""),
    )]

    have_inputs = all(a in reported for a in PREDICTIVE_NEEDS)
    go = plan["predictive"] and have_inputs
    if plan["predictive"] and not have_inputs:      # planned, but an input is missing
        tasks = [
            {**t, "delegated": False, "reason": "its inputs did not come back"} if t["agent"] == PREDICTIVE else t
            for t in plan["tasks"]
        ]
        plan = {**plan, "predictive": False, "tasks": tasks}
    trace.append(_step(
        "Delegate",
        f"Delegated to {PREDICTIVE_LABEL}" if go else f"Did not delegate to {PREDICTIVE_LABEL}",
        next(t["reason"] for t in plan["tasks"] if t["agent"] == PREDICTIVE),
    ))
    return {"plan": plan, "trace": trace}


def delegate_predictive(state: SupervisorState) -> str:
    return PREDICTIVE if state["plan"]["predictive"] else "supervisor_integrate"


# ── Specialist 1: Waiting Time Analysis Agent (LightGBM + RAG evidence) ──

def waiting_time_agent(state: SupervisorState) -> SupervisorState:
    at, clinic, cat = state["as_of"], state["clinic"], state["category"]
    depth = livestate.queue_depth(clinic, cat, at)

    lag = state["appointment_lag_min"]
    wait = models_service.predict_wait(cat, livestate.wait_model_features(
        clinic, cat, at, appointment_lag_min=lag, visit_position=state["visit_position"],
    ))

    rag = {"median": None, "n": 0}
    service_type = CATEGORY_TO_SERVICE_TYPE.get(cat)
    if service_type:
        query = (f"{service_type} at {clinic}, on {DAY_NAMES[at.weekday()]} "
                 f"at {at.hour:02d}:00, {daypart(at.hour)} weekday.")
        hits = search_similar(
            get_client(), embed_one(query),
            build_filter(clinic=clinic, service_type=service_type,
                         hour=at.hour, day_of_week=at.weekday()),
            limit=50,
        )
        if hits:
            rag = {"median": round(statistics.median(h["wait_min"] for h in hits), 1),
                   "n": len(hits)}

    return {"findings": {"waiting_time": {
        "model_estimate_min": round(wait["estimate_min"], 1),
        "model_mae_min": models_service.wait_error_min(has_appointment=lag is not None),
        "model_range_min": [round(wait["p10_min"], 1), round(wait["p90_min"], 1)],
        "breach_limit_min": wait["breach_limit_min"],
        "breach_probability": round(wait["breach_probability"], 2),
        "breach_flag": wait["breach_flag"],
        "rag_median_min": rag["median"],
        "rag_evidence_n": rag["n"],
        "current_queue_depth": depth,
    }}}


# ── Specialist 2: Bottleneck Detection Agent (HMM congestion states) ─────

def bottleneck_agent(state: SupervisorState) -> SupervisorState:
    at, clinic = state["as_of"], state["clinic"]
    depths = livestate.queue_depths_all(clinic, at)
    stations = []
    for cat, depth in depths.items():
        waits = livestate.recent_waits(clinic, cat, at)
        hmm = models_service.congestion_state(cat, waits)
        stations.append({**hmm, "queue_depth": depth})
    bottlenecks = [s for s in stations
                   if s["state"] == "High" or (s["state"] == "Medium" and s["queue_depth"] >= 10)]
    return {"findings": {"bottlenecks": {
        "stations": stations,
        "flagged": bottlenecks,
    }}}


# ── Specialist 3: Queue Depth Forecasting Agent (LSTM queue forecast) ────

def queue_depth_forecast_agent(state: SupervisorState) -> SupervisorState:
    at, clinic, cat = state["as_of"], state["clinic"], state["category"]
    feats = livestate.lstm_bucket_features(clinic, cat, at)
    if feats is None:
        forecast, trend = [], "no-data"
    else:
        forecast = models_service.forecast_queue_depth(feats)
        current = float(feats[-1][0])
        trend = ("rising" if forecast and forecast[0] > current * 1.15 else
                 "falling" if forecast and forecast[0] < current * 0.85 else "stable")
    return {"findings": {"forecast": {
        "category": cat,
        "next_hour_queue_depth": forecast,   # 4 × 15-min buckets
        "trend": trend,
        "model_mae": models_service.lstm_meta()["evaluation"]["val_MAE"],
    }}}


# ── Specialist 4: Resource Planning Agent (allocation options) ───────────

def resource_planning_agent(state: SupervisorState) -> SupervisorState:
    at, clinic = state["as_of"], state["clinic"]
    depths = livestate.queue_depths_all(clinic, at)
    total = sum(depths.values()) or 1
    options = []
    for cat, depth in sorted(depths.items(), key=lambda kv: -kv[1]):
        if depth < 5:
            continue
        share = depth / total
        options.append({
            "station": cat,
            "queue_depth": depth,
            "load_share_pct": round(share * 100, 1),
            "option": (
                f"Open an additional {cat.replace('_', ' ')} room/counter"
                if share > 0.35 else
                f"Reassign one floating staff member to {cat.replace('_', ' ')}"
            ),
        })
    return {"findings": {"resource_planning": {"options": options[:4]}}}


# ── Specialist 5: Predictive Analytics Agent (synthesizes specialists 1-4) ─

def predictive_agent(state: SupervisorState) -> SupervisorState:
    """Second-round specialist: the supervisor delegates to it only once the
    two findings it depends on (waiting time + queue forecast) have come back
    (see supervisor_review). Does not call a trained model itself — it analyzes
    their combined findings to produce a trend-adjusted wait projection, a wait
    range taken from the wait model's own quantiles and a near-term outlook. The overall
    operational risk score also needs the bottleneck and resource-planning
    findings, so it is only produced when those specialists ran too."""
    f = state["findings"]
    wt, qf = f["waiting_time"], f["forecast"]
    bn, rp = f.get("bottlenecks"), f.get("resource_planning")

    # Predictive modeling: trend-adjust the LightGBM point estimate using the LSTM
    # queue-depth trend, so a rising/falling queue shifts the expected wait.
    adjust = {"rising": 1.15, "falling": 0.85, "stable": 1.0, "no-data": 1.0}[qf["trend"]]
    predicted_wait_min = round(wt["model_estimate_min"] * adjust, 1)

    # Uncertainty estimation, in minutes throughout: the range is the wait
    # model's own P10–P90 for this patient, shifted by the same trend factor,
    # and the typical error is that model's hold-out MAE. The LSTM's error is
    # measured in patients, not minutes, so it cannot be added to either; the
    # forecast contributes only through the trend factor.
    low, high = (round(v * adjust, 1) for v in wt["model_range_min"])
    uncertainty_min = round(wt["model_mae_min"], 1)

    # Performance forecast: near-term outlook from the queue trend, sharpened by
    # the bottleneck signal when it is available.
    if bn is not None:
        high_risk_stations = len(bn["flagged"])
        outlook = (
            "deteriorating" if qf["trend"] == "rising" and high_risk_stations > 0 else
            "improving" if qf["trend"] == "falling" and high_risk_stations == 0 else
            "stable"
        )
    else:
        outlook = {"rising": "deteriorating", "falling": "improving"}.get(qf["trend"], "stable")

    # Overall operational risk score (0-1): weighted blend of forecast trend,
    # bottleneck pressure, and how much resource reallocation is already indicated.
    risk_score = risk_label = None
    if bn is not None and rp is not None:
        risk_score = round(min(1.0,
            0.4 * (1.0 if qf["trend"] == "rising" else 0.0) +
            0.4 * min(len(bn["flagged"]) / 3, 1.0) +
            0.2 * min(len(rp["options"]) / 4, 1.0)
        ), 2)
        risk_label = "high" if risk_score >= 0.66 else "medium" if risk_score >= 0.33 else "low"

    return {"findings": {"predictive_analysis": {
        "predicted_wait_min": predicted_wait_min,
        "predicted_wait_range_min": [min(low, predicted_wait_min), max(high, predicted_wait_min)],
        "uncertainty_min": uncertainty_min,
        "outlook": outlook,
        "risk_score": risk_score,
        "risk_label": risk_label,
    }}}


# ── Supervisor, phase 3: integrate the findings (result fusion) ──────────

def supervisor_integrate(state: SupervisorState) -> SupervisorState:
    f = state["findings"]
    fused: dict = {}
    if wt := f.get("waiting_time"):
        estimates = [wt["model_estimate_min"]]
        if wt["rag_median_min"] is not None:
            estimates.append(wt["rag_median_min"])
        fused.update({
            "consensus_wait_min": round(statistics.mean(estimates), 1),
            "estimate_spread_min": round(max(estimates) - min(estimates), 1),
            "sources": {"lightgbm": wt["model_estimate_min"], "rag": wt["rag_median_min"]},
        })
    if bn := f.get("bottlenecks"):
        fused["bottleneck_count"] = len(bn["flagged"])
    if qf := f.get("forecast"):
        fused["queue_trend"] = qf["trend"]
    if rp := f.get("resource_planning"):
        fused["resource_option_count"] = len(rp["options"])
    if pa := f.get("predictive_analysis"):
        fused.update({k: pa[k] for k in
                      ("predicted_wait_min", "predicted_wait_range_min", "outlook",
                       "risk_score", "risk_label")})

    parts = []
    if "consensus_wait_min" in fused:
        parts.append(f"consensus wait {fused['consensus_wait_min']:.0f} min")
    if "queue_trend" in fused:
        parts.append(f"queue {fused['queue_trend']}")
    if "bottleneck_count" in fused:
        parts.append(f"{fused['bottleneck_count']} bottleneck(s)")
    if "resource_option_count" in fused:
        parts.append(f"{fused['resource_option_count']} resource option(s)")
    if "outlook" in fused:
        parts.append(f"outlook {fused['outlook']}")
    return {"fused": fused, "trace": [_step(
        "Integrate", f"Integrated the findings of {len(f)} agent(s)", ", ".join(parts) or "nothing to integrate",
    )]}


# ── Supervisor, phase 4: validate the result against the objective ───────

def _goal_criteria(state: SupervisorState) -> list[dict]:
    """What each delegated agent had to deliver for the objective to be met."""
    f = state["findings"]
    criteria = []
    for task in state["plan"]["tasks"]:
        if not task["delegated"]:
            continue
        agent, finding = task["agent"], f.get(FINDING_KEY[task["agent"]])
        if agent == "waiting_time_agent":
            name, met = "Waiting time estimated", finding is not None
            detail = f"{finding['model_estimate_min']} min from the model" if met else "no estimate"
        elif agent == "bottleneck_agent":
            assessed = [s for s in (finding or {}).get("stations", []) if s["state"] != "unknown"]
            name, met = "Stations assessed for bottlenecks", bool(assessed)
            detail = (f"{len(assessed)} station(s) assessed, {len(finding['flagged'])} flagged" if met
                      else "no station had enough recent activity to assess")
        elif agent == "queue_depth_forecast_agent":
            name, met = "Queue forecast produced", bool(finding) and finding["trend"] != "no-data"
            detail = f"next hour trend: {finding['trend']}" if met else "no recent activity to forecast from"
        elif agent == "resource_planning_agent":
            name, met = "Resource options evaluated", finding is not None
            detail = (f"{len(finding['options'])} option(s) proposed" if met and finding["options"]
                      else "queues are too short to need reallocation" if met else "no evaluation")
        else:
            name, met = "Outlook projected", finding is not None
            detail = f"outlook {finding['outlook']}" if met else "no projection"
        criteria.append({"criterion": name, "agent": task["label"], "met": met, "detail": detail})
    return criteria


def supervisor_validate(state: SupervisorState) -> SupervisorState:
    """Goal validation (did the run deliver what the objective asked for?) and
    cross-model consistency checks, which set the confidence."""
    fused, f, plan = state["fused"], state["findings"], state["plan"]
    checks, score = [], 1.0

    criteria = _goal_criteria(state)
    n_met = sum(c["met"] for c in criteria)
    goal = {"achieved": n_met == len(criteria), "met": n_met, "total": len(criteria),
            "objective": plan["objective"], "criteria": criteria}

    # every specialist the supervisor delegated to must have reported back
    delegated = [t for t in plan["tasks"] if t["delegated"]]
    missing = [t["label"] for t in delegated if FINDING_KEY[t["agent"]] not in f]
    checks.append({"check": "delegated_agents_reported", "passed": not missing,
                   "detail": (f"{len(delegated)} of {len(delegated)} reported" if not missing
                              else "missing: " + ", ".join(missing))})
    score -= 0.3 if missing else 0.0

    wt, qf, bn = f.get("waiting_time"), f.get("forecast"), f.get("bottlenecks")
    if wt:    wt, qf, bn = f.get("waiting_time"), f.get("forecast"), f.get("bottlenecks")
    if wt:
        if wt["rag_median_min"] is not None:
            rel_gap = fused["estimate_spread_min"] / max(fused["consensus_wait_min"], 1.0)
            agree = rel_gap <= 0.5
            checks.append({"check": "lightgbm_vs_rag_agreement", "passed": agree,
                           "detail": f"spread {fused['estimate_spread_min']}m "
                                     f"({rel_gap:.0%} of consensus)"})
            score -= 0.0 if agree else 0.3
        else:
            checks.append({"check": "rag_evidence", "passed": False,
                           "detail": "no RAG evidence for this category"})
            score -= 0.15

    if wt and qf and bn:
        hmm_state = next((s["state"] for s in bn["stations"]
                          if s["category"] == state["category"]), "unknown")
        consistent = not (hmm_state == "High" and qf["trend"] == "falling"
                          and fused["consensus_wait_min"] < 15)
        checks.append({"check": "hmm_forecast_consistency", "passed": consistent,
                       "detail": f"HMM={hmm_state}, trend={qf['trend']}"})
        score -= 0.0 if consistent else 0.2

    if qf and qf["trend"] == "no-data":
        checks.append({"check": "forecast_data", "passed": False,
                       "detail": "no recent activity to forecast from"})
        score -= 0.15

    label = "high" if score >= 0.85 else "medium" if score >= 0.6 else "low"
    failed = [c["check"] for c in checks if not c["passed"]]
    return {
        "goal": goal,
        "critique": {
            "checks": checks,
            "confidence": round(max(0.2, score), 2),
            "confidence_label": label,
        },
        "trace": [_step(
            "Validate",
            "Goal achieved" if goal["achieved"] else "Goal only partly achieved",
            f"{n_met} of {len(criteria)} criteria met; confidence {label}"
            + (f" (failed check: {', '.join(failed)})" if failed else ""),
        )],
    }


# ── Decision Support Module ──────────────────────────────────────────────

def decision_support(state: SupervisorState) -> SupervisorState:
    f = state["findings"]
    decisions = []
    for b in f.get("bottlenecks", {}).get("flagged", []):
        decisions.append({
            "priority": 1 if b["state"] == "High" else 2,
            "recommendation": f"Bottleneck at {b['category'].replace('_', ' ')}: "
                              f"{b['queue_depth']} waiting, HMM state {b['state']} "
                              f"(typical {b['state_mean_wait']}m). Escalate staffing.",
            "impact": "reduce station wait toward its Low-state mean",
        })
    for opt in f.get("resource_planning", {}).get("options", []):
        decisions.append({
            "priority": 2 if opt["load_share_pct"] > 35 else 3,
            "recommendation": opt["option"],
            "impact": f"station holds {opt['load_share_pct']}% of clinic load "
                      f"({opt['queue_depth']} waiting)",
        })
    if f.get("forecast", {}).get("trend") == "rising":
        decisions.append({
            "priority": 1,
            "recommendation": f"Queue for {state['category'].replace('_', ' ')} is "
                              "forecast to rise within the next hour — pre-empt with "
                              "additional capacity now.",
            "impact": "avoid projected queue growth",
        })
    pa = f.get("predictive_analysis")
    if pa and pa["risk_label"] == "high":
        decisions.append({
            "priority": 1,
            "recommendation": "Predictive analysis flags high overall operational risk "
                              f"({pa['risk_score']}) — outlook is {pa['outlook']}. Review "
                              "staffing and station allocation across the clinic now.",
            "impact": f"projected wait range {pa['predicted_wait_range_min'][0]:.0f}"
                      f"–{pa['predicted_wait_range_min'][1]:.0f} min if unaddressed",
        })
    decisions.sort(key=lambda d: d["priority"])
    return {"decisions": decisions[:6]}


# ── Reporting & Visualization ────────────────────────────────────────────

def reporting(state: SupervisorState) -> SupervisorState:
    fused, crit, f = state["fused"], state["critique"], state["findings"]
    at = state["as_of"]
    station = f"{state['category'].replace('_', ' ')} at {state['clinic']}"
    parts = [f"As of {at:%A %H:%M}, for {station}:"]
    if "consensus_wait_min" in fused:
        src = fused["sources"]
        parts.append(
            f"expected wait is ~{fused['consensus_wait_min']:.0f} min (LightGBM {src['lightgbm']}m"
            + (f", historical RAG median {src['rag']}m" if src["rag"] else "") + ")."
        )
    if "queue_trend" in fused:
        parts.append(f"Queue trend: {fused['queue_trend']}.")
    if "bottleneck_count" in fused:
        parts.append(f"{fused['bottleneck_count']} station(s) flagged as bottlenecks.")
    if "resource_option_count" in fused:
        parts.append(f"{fused['resource_option_count']} resource option(s) proposed.")
    parts.append(f"Confidence: {crit['confidence_label']} ({crit['confidence']}).")
    if "outlook" in fused:
        risk = (f" (risk {fused['risk_label']}, score {fused['risk_score']})"
                if fused["risk_label"] else "")
        parts.append(
            f"Predictive outlook: {fused['outlook']}{risk} — projected wait range next cycle "
            f"{fused['predicted_wait_range_min'][0]:.0f}–{fused['predicted_wait_range_min'][1]:.0f} min."
        )
    summary = " ".join(parts)

    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=settings.llm_model, max_tokens=400,
                              api_key=settings.openai_api_key)
            summary = llm.invoke(
                "Rewrite this clinic operations summary for a nurse manager. Keep every "
                f"number exactly as given, two short paragraphs max:\n\n{summary}"
            ).content
        except Exception:
            pass  # numeric summary stands on its own (NFR-4)

    alerts = [
        {"level": "critical" if d["priority"] == 1 else "warning",
         "message": d["recommendation"]}
        for d in state["decisions"] if d["priority"] <= 2
    ]
    return {"report": {
        "summary": summary,
        "as_of": at.isoformat(),
        "clinic": state["clinic"],
        "category": state["category"],
        "plan": state["plan"],
        "trace": state["trace"],
        "goal": state["goal"],
        "consensus_wait_min": fused.get("consensus_wait_min"),
        "confidence": crit,
        "findings": f,
        "recommendations": state["decisions"],
        "alerts": alerts,
    }}


def build_supervisor_graph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("waiting_time_agent", waiting_time_agent)
    g.add_node("bottleneck_agent", bottleneck_agent)
    g.add_node("queue_depth_forecast_agent", queue_depth_forecast_agent)
    g.add_node("resource_planning_agent", resource_planning_agent)
    g.add_node("clarify", clarify)
    g.add_node("supervisor_review", supervisor_review)
    g.add_node("predictive_agent", predictive_agent)
    g.add_node("supervisor_integrate", supervisor_integrate)
    g.add_node("supervisor_validate", supervisor_validate)
    g.add_node("decision_support", decision_support)
    g.add_node("reporting", reporting)

    g.set_entry_point("supervisor")
    # round 1: the supervisor delegates only to the specialists its plan calls
    # for; they run in parallel and report back to it
    g.add_conditional_edges("supervisor", delegate, [*SPECIALISTS, "clarify"])
    g.add_edge("clarify", END)
    for specialist in SPECIALISTS:
        g.add_edge(specialist, "supervisor_review")
    # round 2: Predictive Analytics, if its inputs came back
    g.add_conditional_edges("supervisor_review", delegate_predictive,
                            [PREDICTIVE, "supervisor_integrate"])
    g.add_edge(PREDICTIVE, "supervisor_integrate")
    g.add_edge("supervisor_integrate", "supervisor_validate")
    g.add_edge("supervisor_validate", "decision_support")
    g.add_edge("decision_support", "reporting")
    g.add_edge("reporting", END)
    return g.compile()


supervisor_graph = build_supervisor_graph()
