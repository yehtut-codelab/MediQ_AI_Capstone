"""Curated demo patients for the Arrival Simulator and New Patient Entry.

Each sample pins the clinic state to `replay_at`, a recorded moment in
Clean_Dataset, so the models return the same output on whatever day it is
presented. The moments were chosen so that every model output is present
(wait estimate and range, limit warning, HMM state, LSTM forecast, RAG
evidence) and so that the five together cover all HMM states, all forecast
trends and all triage priorities. The patients themselves are fictional.

The arrival is shown with today's date at `arrival_time`; `replay_at` keeps
that same time of day so the screen and the replayed moment agree.
"""

from datetime import datetime

from app.schemas import GoldenSample

GOLDEN_SAMPLES: list[GoldenSample] = [
    GoldenSample(
        id="busy-consultation",
        label="Busy consultation",
        story="Clinic 1A late morning, 55 waiting — long wait and limit warning",
        display_name="Mdm Lim Siew Hua",
        clinic="Clinic 1A",
        service_type="Consultation",
        category="consultation",
        arrival_time="11:00",
        appointment_time="11:20",
        replay_at=datetime(2025, 9, 17, 11, 0),
        diagnosis="Primary open-angle glaucoma (POAG), both eyes",
        medical_history="Family history of glaucoma",
        current_issue="Routine glaucoma review, no new symptoms",
    ),
    GoldenSample(
        id="urgent-walk-in",
        label="Urgent retinal symptoms",
        story="Flashes and floaters — triaged P1 ahead of a medium queue",
        display_name="Mr Rajan Pillai",
        clinic="TTSH Eye Centre",
        service_type="Consultation",
        category="consultation",
        arrival_time="10:00",
        appointment_time="10:20",
        replay_at=datetime(2025, 9, 3, 10, 0),
        diagnosis="Retinal detachment",
        medical_history="Previous retinal laser treatment",
        current_issue="New flashes of light and floaters, curtain over vision",
    ),
    GoldenSample(
        id="field-test-congested",
        label="Visual field test, congested",
        story="HVF station in a high-congestion state, queue starting to ease",
        display_name="Mr Tan Boon Keng",
        clinic="TTSH Eye Centre",
        service_type="HVF",
        category="visual_field_test",
        arrival_time="10:00",
        appointment_time="10:20",
        replay_at=datetime(2025, 9, 2, 10, 0),
        diagnosis="Ocular hypertension",
        medical_history="Hypertension, on medication",
        current_issue="Scheduled visual field test for monitoring",
    ),
    GoldenSample(
        id="diabetic-scan",
        label="Diabetic eye scan",
        story="OCT scan with worsening vision — triaged P2, short wait",
        display_name="Mdm Noraini Hassan",
        clinic="TTSH Eye Centre",
        service_type="OCT",
        category="diagnostic_scan",
        arrival_time="10:00",
        appointment_time="10:20",
        replay_at=datetime(2025, 9, 23, 10, 0),
        diagnosis="Diabetic retinopathy",
        medical_history="Diabetic, on insulin",
        current_issue="Vision getting worse over the past month",
    ),
    GoldenSample(
        id="early-morning-check",
        label="Early-morning vision check",
        story="First VA slot of the day — short wait, queue building up",
        display_name="Ms Chua Mei Ling",
        clinic="TTSH Eye Centre",
        service_type="VA",
        category="visual_acuity",
        arrival_time="08:30",
        appointment_time="08:50",
        replay_at=datetime(2025, 9, 12, 8, 30),
        diagnosis="Age-related cataract",
        medical_history="No significant past ocular history",
        current_issue="Gradual blurring when reading, routine check",
    ),
]
