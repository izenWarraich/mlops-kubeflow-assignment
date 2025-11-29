"""
Kubeflow pipeline definition.
"""
import mlflow
from pathlib import Path
import sys
import shutil

# Add src to path to import pipeline_components
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_components import extract_data, preprocess_data, train_model, evaluate_model


def main():
    """
    Main pipeline function that orchestrates the full ML pipeline.
    """
    # Define paths
    raw = "data/raw/boston.csv"
    extracted_dest = "data/processed/boston_raw.csv"
    processed = "data/processed/processed.pkl"
    model_path = "models/model.pkl"

    # Ensure directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # Start MLflow run for the full pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        print("=" * 60)
        print("Starting Full ML Pipeline")
        print("=" * 60)

        # Step 1: Extract data
        print("\n[Step 1/4] Extracting data...")
        # Copy raw data to processed folder if not same file
        extracted_path = extract_data(raw, extracted_dest)
        print(f"✓ Data extracted to: {extracted_path}")
        mlflow.log_param("raw_data_path", raw)

        # Step 2: Preprocess data
        print("\n[Step 2/4] Preprocessing data...")
        processed_path = preprocess_data(extracted_path, processed)
        print(f"✓ Data preprocessed and saved to: {processed_path}")
        mlflow.log_param("processed_data_path", processed)

        # Step 3: Train model
        print("\n[Step 3/4] Training model...")
        model_file_path = train_model(processed_path, model_path, n_estimators=100)
        print(f"✓ Model trained and saved to: {model_file_path}")
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("n_estimators", 100)

        # Step 4: Evaluate model
        print("\n[Step 4/4] Evaluating model...")
        metrics_file_path = evaluate_model(processed_path, model_file_path)
        print(f"✓ Model evaluated. Metrics saved to: {metrics_file_path}")

        # Read and print metrics
        if Path(metrics_file_path).exists():
            with open(metrics_file_path, 'r') as f:
                metrics_content = f.read()
                print("\n" + "=" * 60)
                print("Final Results:")
                print("=" * 60)
                print(metrics_content)

        # Log pipeline completion
        mlflow.log_param("pipeline_status", "completed")

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Raw data: {raw}")
        print(f"  - Extracted raw data: {extracted_path}")
        print(f"  - Processed data: {processed}")
        print(f"  - Model: {model_path}")
        print(f"  - Metrics: {metrics_file_path}")


if __name__ == "__main__":
    main()
