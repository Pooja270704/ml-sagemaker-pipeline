import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOTAL_CSV_ROWS = 500
BASE_CSV_ROWS = 450
CSV_DUPLICATE_ROWS = 50
NULL_BIOMARKER_ROWS = 20
EXTREME_BMI_ROWS = 8

TOTAL_SAFETY_ROWS = 500
BASE_SAFETY_ROWS = 470
SAFETY_DUPLICATE_ROWS = 30

DATA_DIR = Path("data/raw")
CSV_PATH = DATA_DIR / "clinical_trials.csv"
JSON_PATH = DATA_DIR / "clinical_safety_events.json"

random.seed(SEED)
np.random.seed(SEED)

STUDY_DRUG_MAP = {
    "CT-ONC-101": "OncoRelix",
    "CT-CARD-202": "Cardiovex",
    "CT-NEURO-303": "Neuroquel",
    "CT-IMM-404": "Immunara",
}
DOSAGE_BY_DRUG = {
    "OncoRelix": [75, 100, 150, 200],
    "Cardiovex": [25, 50, 75, 100],
    "Neuroquel": [50, 75, 100, 150],
    "Immunara": [50, 100, 150, 200],
}
EVENT_TYPES = ["Headache", "Fever", "Nausea", "Fatigue", "Rash", "Dizziness"]
SEVERITIES = ["Mild", "Moderate", "Severe"]


def _bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 2)


def _random_enrolled_date() -> date:
    end = date.today()
    start = end - timedelta(days=730)
    offset_days = random.randint(0, (end - start).days)
    return start + timedelta(days=offset_days)


def build_clinical_trials() -> pd.DataFrame:
    rows = []
    for i in range(BASE_CSV_ROWS):
        study = random.choice(list(STUDY_DRUG_MAP.keys()))
        drug = STUDY_DRUG_MAP[study]
        patient_id = f"PT-{100000 + i}"
        age = random.randint(18, 85)
        gender = random.choice(["M", "F"])

        # Keep baseline physiological values realistic before injecting outliers.
        if gender == "M":
            height = round(float(np.clip(np.random.normal(175, 7), 155, 200)), 1)
            weight = round(float(np.clip(np.random.normal(82, 13), 50, 150)), 1)
        else:
            height = round(float(np.clip(np.random.normal(163, 7), 145, 195)), 1)
            weight = round(float(np.clip(np.random.normal(69, 12), 40, 140)), 1)
        bmi = _bmi(weight, height)

        biomarker = round(random.uniform(0.2, 8.5), 2)
        dosage = random.choice(DOSAGE_BY_DRUG[drug])
        response = round(float(np.clip(np.random.normal(72 - biomarker * 4, 14), 0, 100)), 2)
        adverse_event = random.choices([0, 1], weights=[0.75, 0.25], k=1)[0]

        rows.append(
            {
                "StudyID": study,
                "PatientID": patient_id,
                "DrugName": drug,
                "Age": age,
                "Gender": gender,
                "WeightKg": weight,
                "HeightCm": height,
                "BMI": bmi,
                "DosageMg": dosage,
                "BiomarkerLevel": biomarker,
                "ResponseScore": response,
                "AdverseEventFlag": adverse_event,
                "EnrolledDate": _random_enrolled_date(),
            }
        )

    df = pd.DataFrame(rows)

    # Inject extreme BMI values to support data validation checks.
    extreme_idx = np.random.choice(df.index, size=EXTREME_BMI_ROWS, replace=False)
    for idx in extreme_idx[: EXTREME_BMI_ROWS // 2]:
        df.at[idx, "WeightKg"] = round(random.uniform(32, 45), 1)
        df.at[idx, "HeightCm"] = round(random.uniform(188, 200), 1)
        df.at[idx, "BMI"] = _bmi(df.at[idx, "WeightKg"], df.at[idx, "HeightCm"])
    for idx in extreme_idx[EXTREME_BMI_ROWS // 2 :]:
        df.at[idx, "WeightKg"] = round(random.uniform(120, 145), 1)
        df.at[idx, "HeightCm"] = round(random.uniform(145, 160), 1)
        df.at[idx, "BMI"] = _bmi(df.at[idx, "WeightKg"], df.at[idx, "HeightCm"])

    duplicate_rows = df.sample(CSV_DUPLICATE_ROWS, random_state=SEED)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    null_idx = np.random.choice(df.index, size=NULL_BIOMARKER_ROWS, replace=False)
    df.loc[null_idx, "BiomarkerLevel"] = np.nan

    assert len(df) == TOTAL_CSV_ROWS
    return df


def build_safety_events(clinical_df: pd.DataFrame) -> pd.DataFrame:
    base_patients = clinical_df.iloc[:BASE_CSV_ROWS][["StudyID", "PatientID", "DrugName"]].reset_index(drop=True)

    rows = []
    seen = set()
    while len(rows) < BASE_SAFETY_ROWS:
        patient_row = base_patients.sample(1).iloc[0]
        event_type = random.choice(EVENT_TYPES)
        severity = random.choices(SEVERITIES, weights=[0.56, 0.31, 0.13], k=1)[0]

        if severity == "Mild":
            duration = random.randint(1, 3)
            hospitalized = False
        elif severity == "Moderate":
            duration = random.randint(2, 7)
            hospitalized = random.choices([False, True], weights=[0.9, 0.1], k=1)[0]
        else:
            duration = random.randint(5, 18)
            hospitalized = random.choices([False, True], weights=[0.45, 0.55], k=1)[0]

        event_row = {
            "StudyID": patient_row["StudyID"],
            "PatientID": patient_row["PatientID"],
            "DrugName": patient_row["DrugName"],
            "EventType": event_type,
            "Severity": severity,
            "Hospitalized": bool(hospitalized),
            "EventDurationDays": duration,
        }

        event_key = tuple(event_row.values())
        if event_key in seen:
            continue
        seen.add(event_key)
        rows.append(event_row)

    df = pd.DataFrame(rows)
    duplicate_events = df.sample(SAFETY_DUPLICATE_ROWS, random_state=SEED)
    df = pd.concat([df, duplicate_events], ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    assert len(df) == TOTAL_SAFETY_ROWS
    return df


def write_outputs(clinical_df: pd.DataFrame, safety_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clinical_df.to_csv(CSV_PATH, index=False)

    with JSON_PATH.open("w", encoding="utf-8") as f:
        for record in safety_df.to_dict(orient="records"):
            f.write(json.dumps(record, default=str) + "\n")


def print_summary(clinical_df: pd.DataFrame, safety_df: pd.DataFrame) -> None:
    csv_duplicate_patient_rows = int(clinical_df.duplicated(subset=["PatientID"]).sum())
    csv_duplicate_study_patient_rows = int(clinical_df.duplicated(subset=["StudyID", "PatientID"]).sum())
    csv_null_biomarker_rows = int(clinical_df["BiomarkerLevel"].isna().sum())
    csv_extreme_bmi_rows = int(((clinical_df["BMI"] < 15) | (clinical_df["BMI"] > 45)).sum())

    safety_duplicate_rows = int(
        safety_df.duplicated(
            subset=[
                "StudyID",
                "PatientID",
                "DrugName",
                "EventType",
                "Severity",
                "Hospitalized",
                "EventDurationDays",
            ]
        ).sum()
    )

    print("=" * 72)
    print("Synthetic datasets generated for SageMaker pipeline")
    print("=" * 72)
    print(f"CSV file:  {CSV_PATH} ({len(clinical_df)} rows)")
    print(f"JSON file: {JSON_PATH} ({len(safety_df)} rows, jsonl)")
    print("-" * 72)
    print(f"CSV duplicate PatientID rows: {csv_duplicate_patient_rows}")
    print(f"CSV duplicate StudyID+PatientID rows: {csv_duplicate_study_patient_rows}")
    print(f"CSV null BiomarkerLevel rows: {csv_null_biomarker_rows}")
    print(f"CSV extreme BMI rows (<15 or >45): {csv_extreme_bmi_rows}")
    print(f"JSON duplicate safety events: {safety_duplicate_rows}")
    print("=" * 72)


def main() -> None:
    clinical_df = build_clinical_trials()
    safety_df = build_safety_events(clinical_df)
    write_outputs(clinical_df, safety_df)
    print_summary(clinical_df, safety_df)


if __name__ == "__main__":
    main()
