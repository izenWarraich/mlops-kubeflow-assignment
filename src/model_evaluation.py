"""
Model evaluation component for the pipeline.
"""
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import mlflow


def evaluate_model(processed_path, model_path):
    """
    Evaluate the trained model on the test set.
    
    Args:
        processed_path: Path to the processed.pkl file
        model_path: Path to the model.pkl file
    
    Returns:
        Dictionary containing mse and r2 metrics
    """
    # Convert to Path objects
    processed_path = Path(processed_path)
    model_path = Path(model_path)
    
    # Start MLflow run
    with mlflow.start_run(run_name="evaluation"):
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
        
        # Save metrics.txt to a temporary location for logging
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
    
    return metrics

