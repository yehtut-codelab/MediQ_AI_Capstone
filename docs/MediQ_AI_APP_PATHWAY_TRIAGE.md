# Implementation Note — Patient Pathway Prediction & Triage

**Date:** 2026-07-08 · **Last reviewed against the code:** 2026-10-11
**Scope:** Registration flow (`POST /api/v1/patients/register`), register page (`/register`)
**Related:** `docs/MediQ_AI_APP_SPECIFICATION.md`, `docs/MediQ_AI_APP_FRAMEWORK_DECISION.md` (ADR-001)

## 1. Motivation

The registration flow previously produced only a single-station wait estimate from
historical RAG retrieval. From a clinical-workflow point of view this misses two things
the presentation (diagnosis, history, current issue) actually determines:

- **Pathway** — an eye-centre visit is a *sequence* of stations (e.g. VA → OCT →
  Consultation), not one queue. Total visit time is the sum of a journey, and the
  journey depends on why the patient came.
- **Priority / triage** — sight-threatening presentations (sudden vision loss, chemical
  injury, retinal-detachment symptoms) must jump the queue entirely; wait estimation is
  secondary for them.

This note documents the implementation of both, grounded in observed TTSH data rather
than invented rules wherever possible.

## 2. Empirical pathway mining

**Script:** `backend/scripts/mine_pathways.py` → **Output:** `backend/data/pathways.json`

Each row of the TTSH export is one station visit with `Anonymized_ID` and timestamps.
Grouping by `(patient, day)` and ordering by wait start yields real observed journeys
(consecutive duplicate stations collapsed).

Results over the full export (227,573 clean events):

- **90,367 patient-day journeys**, 1,857 distinct sequences
- Top sequences:

| Share | Journeys | Sequence |
|------:|---------:|----------|
| 18.5% | 16,722 | Pre Consultation Test → Consultation |
| 13.6% | 12,245 | VA → Consultation |
|  8.7% |  7,846 | Pre Consultation Test → Diagnostic → Consultation |
|  8.5% |  7,653 | Consultation only |
|  2.8% |  2,509 | VA → OCT → Consultation |
|  2.6% |  2,315 | Pre Consultation Test → Treatment |
|  1.5% |  1,326 | EYE-Laser → VA → Consultation |
|  1.3% |  1,208 | REFRACTION → Consultation |

The JSON also stores per-station statistics (median/p75 wait, median contact time, n)
used to build projected timelines.

**Data-driven finding:** glaucoma reviews at TTSH flow through **HRT optic-nerve
imaging (338 journeys)** far more than HVF visual fields (90 journeys). The glaucoma
archetype therefore uses VA → HRT → Consultation — what the clinic actually does, not
the textbook assumption.

Re-run after receiving new data: `python scripts/mine_pathways.py` (~70 s).

`backend/data/pathways.json` is matched by the `data/` rule in `.gitignore`, so it is not
in the repository. On a fresh clone, run the script above before using the registration
flow — `pathway.py` reads the file when it builds a timeline.

## 3. Triage — `backend/app/services/triage.py`

**Design principle: deterministic and auditable. No LLM in the decision path.**
Triage is safety-critical; every priority decision cites the exact red-flag phrases
that triggered it, and the rule table is unit-testable (no automated tests exist yet —
`backend/tests/` is empty).

Regex rules scan diagnosis + history + current issue; the best (lowest) priority wins:

- **P1 — Immediate**: sudden vision loss · chemical injury · ocular trauma / foreign
  body · flashes + floaters / curtain (retinal detachment risk) · acute severe pain
  (angle-closure risk) · post-operative deterioration (endophthalmitis risk)
- **P2 — Urgent**: new-onset floaters · new diplopia · acute red eye / discharge ·
  raised pressure symptoms · halos · recent visual deterioration
- **P3 — Routine**: no red flags (default)

Returns `TriageResult(priority, label, reasons)`.

**Known trade-off:** the rules err toward over-triage. Example: a cataract patient
describing "glare and halos" matches the halos rule (angle-closure signal) and is
bumped to P2. This is deliberate — false escalation is safer than a missed emergency —
and is a good precision-vs-safety-recall discussion point for the evaluation chapter.

## 4. Pathway prediction — `backend/app/services/pathway.py`

Two stages:

1. **Classify** presentation → visit archetype. Ordered keyword rules,
   most-specific-first; **P1 triage overrides everything** to the acute walk-in
   pathway. Default: routine review.
2. **Build timeline** from the archetype's mined sequence + per-station medians:
   cumulative ETA offsets, expected wait and service per station, and a fixed
   **+30 min dilation block** before OCT for retina visits.

Archetypes (all sequences observed in the export, with journey support):

| Archetype | Trigger examples | Sequence | Support |
|---|---|---|---:|
| acute_walkin | P1 triage override | VA → Consultation | 12,245 |
| glaucoma_review | glaucoma, IOP, visual field | VA → HRT → Consultation | 338 |
| retina_imaging_review | retinopathy, diabetic, macula, OCT | VA → OCT → Consultation (+dilation) | 2,509 |
| laser_procedure | laser, YAG, capsulotomy | EYE-Laser → VA → Consultation | 1,326 |
| refraction_visit | refraction, spectacles | REFRACTION → Consultation | 1,208 |
| intravitreal_treatment | injection, anti-VEGF | Pre Consultation Test → Treatment | 2,315 |
| cataract_listing | cataract surgery, biometry | Biometry → PAT → MO Clerking | 263 |
| imaging_workup | scan, angiography | Pre Consultation Test → Diagnostic → Consultation | 7,846 |
| routine_review | (default) | Pre Consultation Test → Consultation | 16,722 |

## 5. Integration

**Registration route** (`app/api/routes.py`): register → triage → classify →
build_pathway → ticket. `RegisteredPatient` gains `priority`, `priority_label`,
`triage_reasons`, `pathway_label`, `pathway` (list of steps), `total_visit_min`.

**Wait estimate on the ticket (changed since this note was written).** The headline
`estimated_wait_min` is now the LightGBM wait-time model's prediction for the first
station, with its P10–P90 range, a breach flag, the HMM congestion state and the LSTM
queue forecast (`model_*` fields). The RAG estimate from similar historical cases is kept
as supporting evidence (`rag_estimate_min`, `rag_range_min`, `rag_evidence_n`) and is the
headline only if the model cannot run. Triage and pathway prediction are unaffected by
this change.

**Persistence** (`app/services/registry.py`): new SQLite columns via the idempotent
`ALTER TABLE ... ADD COLUMN` migration pattern on connect; `triage_reasons` and
`pathway` stored as JSON text. Pre-existing rows read back safely as P3 / empty
pathway.

**Frontend** (`frontend/src/app/register/page.tsx`):

- Registration ticket: colour-coded priority badge (P1 red / P2 amber / P3 green)
  with triage reasons
- Predicted Visit Pathway panel: station stepper with per-step ETA
  (`1. VA starts ~T+0 · wait ~4 min · service ~3 min → …`) and projected total
- "Registered Today" table: sorted in **triage order** (priority, then arrival),
  with priority and pathway columns. Stored `queue_position` remains the
  arrival-order ticket number; ordering is applied at display time.

## 6. Verification performed (2026-07-08)

- Unit sweep of 7 presentations through triage → classify → build_pathway
  (P1 acute, P2 glaucoma, P3 retina/refraction/routine all correct)
- Live: "flashes and new floaters, curtain over vision" → **P1**, both
  retinal-detachment flags cited, acute pathway, ~43 min projected
- Live: "POAG, routine six-month review" → **P3**, VA → HRT → Consultation, ~48 min
- The register page lists the later-arriving P1 ahead of earlier P3s;
  page renders (HTTP 200); legacy rows unaffected. (The API itself,
  `GET /api/v1/patients`, returns today's registrations newest first; the triage
  ordering is applied by the page.)

## 7. Limitations & next steps

- **Keyword classification is shallow.** Structured intake fields (diagnosis codes,
  visit-type dropdown) would beat free text; an LLM could normalise text into symptom
  terms *before* the rules, with rules remaining the final authority.
- **Timelines use unconditional per-station medians** — not conditioned on time of day
  or current queue load. The model-based queue simulation now exists
  (`POST /api/v1/queue/simulate`, and the `model_*` fields on the ticket), but the pathway
  timeline does not use it yet; blending the two is the natural next step.
- **Evaluation pending:** on a held-out month — pathway sequence accuracy (predicted
  vs observed station sequence) and MAE of projected vs actual total visit time.
- **PDPA note:** free-text clinical fields are patient data; a real deployment needs
  consent/anonymisation handling. Demo-only for the capstone.
