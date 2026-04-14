"""
Prediction Module: Save/load model artifacts and run inference.
"""
import os
import joblib
import numpy as np
import pandas as pd


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def save_model_artifacts(data, output_dir=None):
    """Save all model artifacts for later inference."""
    if output_dir is None:
        output_dir = MODEL_DIR

    os.makedirs(output_dir, exist_ok=True)

    model = data.get("tuned_model", data.get("best_model"))
    model_name = data.get("best_model_name", "Unknown")

    joblib.dump(model, os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(data["scaler"], os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(data["label_encoder"], os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(data["feature_names"], os.path.join(output_dir, "feature_names.joblib"))
    joblib.dump(data["tfidf_vectorizer"], os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(data["svd_model"], os.path.join(output_dir, "svd_model.joblib"))

    print(f"\n[SAVED] Model artifacts saved to '{output_dir}/':")
    print(f"  - best_model.joblib    ({model_name})")
    print(f"  - scaler.joblib")
    print(f"  - label_encoder.joblib")
    print(f"  - feature_names.joblib")
    print(f"  - tfidf_vectorizer.joblib")
    print(f"  - svd_model.joblib")


def load_artifacts(model_dir=None):
    """Load saved model artifacts."""
    if model_dir is None:
        model_dir = MODEL_DIR

    paths = {
        "model": os.path.join(model_dir, "best_model.joblib"),
        "scaler": os.path.join(model_dir, "scaler.joblib"),
        "encoder": os.path.join(model_dir, "label_encoder.joblib"),
        "features": os.path.join(model_dir, "feature_names.joblib"),
        "tfidf": os.path.join(model_dir, "tfidf_vectorizer.joblib"),
        "svd": os.path.join(model_dir, "svd_model.joblib"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact '{name}' not found at {path}. Run `python main.py` first."
            )

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    encoder = joblib.load(paths["encoder"])
    feature_names = joblib.load(paths["features"])
    tfidf = joblib.load(paths["tfidf"])
    svd = joblib.load(paths["svd"])

    return model, scaler, encoder, feature_names, tfidf, svd


def predict_readmission(patient_data, diagnosis_text):
    """Predict readmission risk for a single patient."""
    from src.text_processing import preprocess_text

    model, scaler, encoder, feature_names, tfidf, svd = load_artifacts()

    # Process structured features
    df = pd.DataFrame([patient_data])
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Get structured feature names (without text_svd_ features)
    structured_names = [f for f in feature_names if not f.startswith("text_svd_")]
    for col in structured_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[structured_names]

    # Process text
    processed_text = preprocess_text(diagnosis_text)
    text_tfidf = tfidf.transform([processed_text])
    text_svd = svd.transform(text_tfidf)

    # Combine features
    X_combined = np.hstack([df_encoded.values, text_svd])

    # Scale
    X_scaled = scaler.transform(X_combined)

    # Predict
    prediction = model.predict(X_scaled)[0]
    prediction_label = encoder.inverse_transform([prediction])[0]

    probability = 0.0
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_scaled)[0][1]

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": prediction_label,
        "probability": round(float(probability), 4),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    sample_patient = {
        "age": 72,
        "gender": "Male",
        "admission_type": "Emergency",
        "num_medications": 18,
        "num_procedures": 3,
        "num_diagnoses": 9,
        "time_in_hospital": 8,
        "num_lab_procedures": 65,
        "number_emergency": 2,
        "number_inpatient": 3,
        "number_outpatient": 5,
    }

    sample_text = (
        "Patient diagnosed with congestive heart failure and type 2 diabetes. "
        "Presenting symptoms include shortness of breath and chest pain. "
        "Multiple comorbidities complicate treatment plan. "
        "Previous admission within 30 days noted."
    )

    result = predict_readmission(sample_patient, sample_text)
    print(f"\n{'=' * 40}")
    print(f"  READMISSION PREDICTION")
    print(f"{'=' * 40}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']:.2%}")
    print(f"  Risk Level  : {result['risk_level']}")
