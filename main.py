"""
Hospital Readmission Prediction Pipeline
Combines NLP text analysis with traditional ML on structured clinical data.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import run_preprocessing_pipeline
from src.text_processing import run_text_processing
from src.feature_engineering import run_feature_engineering
from src.model_training import run_model_training
from src.model_evaluation import run_model_evaluation
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.predict import save_model_artifacts


def main():
    start_time = time.time()

    print("\n" + "+" + "=" * 58 + "+")
    print("|  HOSPITAL READMISSION PREDICTION - NLP + ML PIPELINE     |")
    print("+" + "=" * 58 + "+\n")

    data_path = os.path.join(PROJECT_ROOT, "data", "hospital_readmission.csv")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    model_dir = os.path.join(PROJECT_ROOT, "models")

    # Step 1: Data Preprocessing
    data = run_preprocessing_pipeline(data_path)

    # Step 2: NLP Text Processing (TF-IDF + SVD)
    data = run_text_processing(data)

    # Step 3: Feature Engineering (Combine + Scale + SMOTE)
    data = run_feature_engineering(data)

    # Step 4: Model Training
    data = run_model_training(data)

    # Step 5: Model Evaluation
    data = run_model_evaluation(data, output_dir=output_dir)

    # Step 6: Hyperparameter Tuning
    data = run_hyperparameter_tuning(data, output_dir=output_dir)

    # Step 7: Save Model Artifacts
    print("=" * 60)
    print("  STEP 7: SAVING MODEL")
    print("=" * 60)
    save_model_artifacts(data, output_dir=model_dir)

    elapsed = time.time() - start_time
    print("\n" + "+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE                                       |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Total time: {elapsed:.1f} seconds")
    print(f"  Best model: {data['best_model_name']}")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"  Model saved to:   {model_dir}/")
    print(f"\n  Run predictions with:")
    print(f"    python src/predict.py\n")


if __name__ == "__main__":
    main()
