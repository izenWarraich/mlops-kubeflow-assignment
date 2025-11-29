"""
Pipeline components for Kubeflow pipeline.
MLflow-based pipeline components for data extraction, preprocessing, training, and evaluation.
"""
import shutil
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow


def extract_data(raw_src_path: str, raw_dest_path: str) -> str:
    """
    Copy a CSV file, log to MLflow, and return the destination path.
    
    Args:
        raw_src_path: Source path of the CSV file to copy
        raw_dest_path: Destination path where the CSV file will be copied
    
    Returns:
        String path of the destination file
    """
    # Convert to Path objects
    src_path = Path(raw_src_path)
    dest_path = Path(raw_dest_path)
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run (nested if already in a run)
    with mlflow.start_run(run_name='data_extraction', nested=True):
        # Log source path
        mlflow.log_param("source_path", str(src_path))
        
        # Copy the file
        shutil.copy(src_path, dest_path)
        
        # Log the copied artifact
        mlflow.log_artifact(str(dest_path))
        
        # Log destination path as parameter
        mlflow.log_param("destination_path", str(dest_path))
    
    return str(dest_path)


def preprocess_data(raw_path: str, processed_path: str) -> str:
    """
    Preprocess data: load CSV, standardize features, split train/test, and save.
    
    Args:
        raw_path: Path to the raw CSV file
        processed_path: Path where the processed.pkl file will be saved
    
    Returns:
        String path of the processed file
    """
    # Convert to Path objects
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    
    # Ensure destination directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run (nested if already in a run)
    with mlflow.start_run(run_name="preprocessing", nested=True):
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
    
    return str(processed_path)


def train_model(processed_path: str, model_path: str, n_estimators: int = 100) -> str:
    """
    Train a RandomForestRegressor model and save it.
    
    Args:
        processed_path: Path to the processed.pkl file
        model_path: Path where the model.pkl file will be saved
        n_estimators: Number of trees in the random forest (default: 100)
    
    Returns:
        String path of the saved model file
    """
    # Convert to Path objects
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    
    # Ensure destination directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run (nested if already in a run)
    with mlflow.start_run(run_name="training", nested=True):
        # Load processed.pkl
        processed_data = joblib.load(processed_path)
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train RandomForestRegressor
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save model.pkl using joblib
        joblib.dump(model, model_path)
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("processed_file", str(processed_path))
        
        # Log metrics
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        
        # Log model artifact
        mlflow.log_artifact(str(model_path))
    
    return str(model_path)


def evaluate_model(processed_path: str, model_path: str) -> str:
    """
    Evaluate the trained model on the test set.
    
    Args:
        processed_path: Path to the processed.pkl file
        model_path: Path to the model.pkl file
    
    Returns:
        String path to the metrics.txt file
    """
    # Convert to Path objects
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    
    # Start MLflow run (nested if already in a run)
    with mlflow.start_run(run_name="evaluation", nested=True):
        # Load processed.pkl and model.pkl
        processed_data = joblib.load(processed_path)
        model = joblib.load(model_path)
        
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate MSE and R2 metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            'mse': mse,
            'r2': r2
        }
        
        # Create metrics.txt file
        metrics_text = f"Model Evaluation Metrics\n"
        metrics_text += f"========================\n"
        metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
        metrics_text += f"R2 Score: {r2:.4f}\n"
        
        # Save metrics.txt to a file for KFP artifact tracking
        metrics_file = model_path.parent / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(metrics_text)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log metrics.txt file as artifact
        mlflow.log_artifact(str(metrics_file))
        
        # Log parameters
        mlflow.log_param("processed_file", str(processed_path))
        mlflow.log_param("model_file", str(model_path))
    
    # Return metrics file path for KFP artifact tracking
    return str(metrics_file)
