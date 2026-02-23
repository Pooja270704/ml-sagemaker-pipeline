from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    # Local raw files (for local testing)
    local_trials_csv: str = "data/raw/clinical_trials.csv"
    local_safety_json: str = "data/raw/clinical_safety_events.json"

    # Default DB table names (RAW STG and structured PSA)
    stg_trials_table: str = "STG_clinical_trials"
    stg_safety_table: str = "STG_clinical_safety_events"
    psa_trials_table: str = "PSA_clinical_trials"
    psa_safety_table: str = "PSA_clinical_safety_events"