"""
Generate a synthetic Hospital Readmission dataset with
structured clinical features + free-text diagnostic notes.
"""
import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N = 5000

# --- Medical text building blocks ---
CONDITIONS = [
    "congestive heart failure", "type 2 diabetes", "hypertension",
    "chronic obstructive pulmonary disease", "pneumonia",
    "acute myocardial infarction", "atrial fibrillation",
    "chronic kidney disease", "urinary tract infection",
    "sepsis", "anemia", "asthma", "hypothyroidism",
    "coronary artery disease", "stroke", "deep vein thrombosis",
    "pulmonary embolism", "liver cirrhosis", "gastroenteritis",
    "cellulitis", "bronchitis", "osteoarthritis"
]

SYMPTOMS = [
    "shortness of breath", "chest pain", "fatigue",
    "dizziness", "nausea", "abdominal pain",
    "swelling in legs", "persistent cough", "fever",
    "weight loss", "confusion", "palpitations",
    "difficulty breathing", "weakness", "back pain",
    "headache", "blurred vision", "frequent urination"
]

PROCEDURES = [
    "blood transfusion administered", "CT scan performed",
    "echocardiogram completed", "dialysis session",
    "surgical debridement", "cardiac catheterization",
    "MRI of the brain", "endoscopy performed",
    "chest X-ray obtained", "ultrasound of abdomen",
    "insulin therapy initiated", "antibiotic therapy started",
    "oxygen supplementation provided", "physical therapy recommended"
]

OUTCOMES = [
    "Patient condition stabilized", "Discharged with follow-up instructions",
    "Referred to specialist for further evaluation",
    "Medication regimen adjusted", "Patient advised lifestyle modifications",
    "Condition improved with treatment", "Monitoring recommended",
    "Patient counseled on medication compliance",
    "Home health services arranged", "Palliative care consultation initiated"
]


def generate_diagnosis_text(readmitted: bool) -> str:
    """Generate a realistic diagnostic note."""
    n_conditions = random.randint(1, 3)
    n_symptoms = random.randint(1, 3)
    n_procedures = random.randint(0, 2)

    conds = random.sample(CONDITIONS, n_conditions)
    symps = random.sample(SYMPTOMS, n_symptoms)
    procs = random.sample(PROCEDURES, n_procedures) if n_procedures > 0 else []
    outcome = random.choice(OUTCOMES)

    parts = []
    parts.append(f"Patient diagnosed with {', '.join(conds)}.")
    parts.append(f"Presenting symptoms include {', '.join(symps)}.")
    if procs:
        parts.append(f"{'. '.join(procs)}.")
    parts.append(outcome + ".")

    # Readmitted patients tend to have more complex notes
    if readmitted and random.random() > 0.4:
        extras = [
            "Previous admission within 30 days noted.",
            "Multiple comorbidities complicate treatment plan.",
            "Patient has history of non-compliance with medication.",
            "High risk for complications identified.",
            "Requires close monitoring post-discharge.",
            "Complex medication interactions observed.",
        ]
        parts.append(random.choice(extras))

    return " ".join(parts)


def generate_dataset(n_samples: int = N):
    """Generate the full synthetic dataset."""
    data = {
        "patient_id": list(range(1, n_samples + 1)),
        "age": np.random.randint(18, 95, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "admission_type": np.random.choice(
            ["Emergency", "Urgent", "Elective"], n_samples, p=[0.5, 0.3, 0.2]
        ),
        "num_medications": np.random.randint(1, 30, n_samples),
        "num_procedures": np.random.randint(0, 6, n_samples),
        "num_diagnoses": np.random.randint(1, 16, n_samples),
        "time_in_hospital": np.random.randint(1, 14, n_samples),
        "num_lab_procedures": np.random.randint(1, 120, n_samples),
        "number_emergency": np.random.randint(0, 5, n_samples),
        "number_inpatient": np.random.randint(0, 10, n_samples),
        "number_outpatient": np.random.randint(0, 20, n_samples),
    }

    # Class imbalance: ~30% readmitted
    readmitted = np.random.choice(
        ["Yes", "No"], n_samples, p=[0.30, 0.70]
    )

    # Generate diagnostic text
    diagnosis_texts = []
    for i in range(n_samples):
        is_readmitted = readmitted[i] == "Yes"
        diagnosis_texts.append(generate_diagnosis_text(is_readmitted))

    data["diagnosis_text"] = diagnosis_texts
    data["readmitted"] = readmitted

    # Introduce some missing values (~2-3%) in numeric columns
    for col in ["age", "num_medications", "num_lab_procedures", "time_in_hospital"]:
        mask = np.random.random(n_samples) < 0.025
        data[col] = np.where(mask, np.nan, data[col])

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    output_path = os.path.join(os.path.dirname(__file__), "data", "hospital_readmission.csv")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Generated dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[INFO] Saved to: {output_path}")
    print(f"[INFO] Class distribution:\n{df['readmitted'].value_counts()}")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}")
