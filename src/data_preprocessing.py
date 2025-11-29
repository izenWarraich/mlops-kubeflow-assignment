"""
Data preprocessing component for the pipeline.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow


def preprocess_data(raw_path, processed_path):
    """
    Preprocess data: load CSV, standardize features, split train/test, and save.
    
    Args:
        raw_path: Path to the raw CSV file
        processed_path: Path where the processed.pkl file will be saved
    
    Returns:
        Path object of the processed file
    """
    # Convert to Path objects
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    
    # Ensure destination directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="preprocessing"):
        # Load CSV using pandas
        df = pd.read_csv(raw_path)
        
        # Separate features and target (assuming 'medv' is the target column)
        # For Boston dataset, target is typically 'medv' or 'MEDV'
        target_col = 'medv' if 'medv' in df.columns else 'MEDV'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Prepare data dictionary to save
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
        
        # Save processed.pkl using joblib
        joblib.dump(processed_data, processed_path)
        
        # Log parameters
        mlflow.log_param("source_file", str(raw_path))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        
        # Log artifacts
        mlflow.log_artifact(str(processed_path))
        
        # Log metrics
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
    
    return processed_path

