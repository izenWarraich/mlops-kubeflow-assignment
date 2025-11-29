"""
Data extraction component for the pipeline.
"""
import shutil
from pathlib import Path
import mlflow


def extract_data(raw_src_path, raw_dest_path):
    """
    Copy a CSV file, log to MLflow, and return the destination path.
    
    Args:
        raw_src_path: Source path of the CSV file to copy
        raw_dest_path: Destination path where the CSV file will be copied
    
    Returns:
        Path object of the destination file
    """
    # Convert to Path objects
    src_path = Path(raw_src_path)
    dest_path = Path(raw_dest_path)
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name='data_extraction'):
        # Log source path
        mlflow.log_param("source_path", str(src_path))
        
        # Copy the file
        shutil.copy(src_path, dest_path)
        
        # Log the copied artifact
        mlflow.log_artifact(str(dest_path))
        
        # Log destination path as parameter
        mlflow.log_param("destination_path", str(dest_path))
    
    return dest_path

