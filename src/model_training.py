"""
Model training component.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import mlflow


def train_model(processed_path, model_path, n_estimators=100):
    """
    Train a RandomForestRegressor model and save it.
    
    Args:
        processed_path: Path to the processed.pkl file
        model_path: Path where the model.pkl file will be saved
        n_estimators: Number of trees in the random forest (default: 100)
    
    Returns:
        Path object of the saved model file
    """
    # Convert to Path objects
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    
    # Ensure destination directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="training"):
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
    
    return model_path

