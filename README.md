# MLOps Kubeflow Assignment

A complete MLOps pipeline implementation using Kubeflow Pipelines, MLflow, and scikit-learn for training and evaluating a machine learning model on the Boston Housing dataset.

## Project Overview

This project demonstrates an end-to-end machine learning pipeline with the following components:

1. **Data Extraction** - Copies raw data files
2. **Data Preprocessing** - Standardizes features and splits data into train/test sets
3. **Model Training** - Trains a RandomForestRegressor model
4. **Model Evaluation** - Evaluates the model and computes metrics (MSE, R²)

All components are integrated with MLflow for experiment tracking and can be compiled into Kubeflow Pipeline components.

## Project Structure

```
mlops-kubeflow-assignment/
├── components/              # Kubeflow Pipeline component YAML files
│   ├── extract_data.yaml
│   ├── preprocess_data.yaml
│   ├── train_model.yaml
│   └── evaluate_model.yaml
├── data/
│   ├── raw/                # Raw data files
│   │   └── boston.csv
│   └── processed/          # Processed data files
│       ├── boston_raw.csv
│       └── processed.pkl
├── models/                 # Trained models and metrics
│   ├── model.pkl
│   └── metrics.txt
├── src/                    # Source code
│   ├── pipeline_components.py  # Main component functions
│   ├── data_extraction.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── compile_components.py   # Script to compile components to YAML
├── pipeline.py             # Main pipeline orchestration
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

## Features

- **MLflow Integration**: All pipeline steps log parameters, metrics, and artifacts to MLflow
- **Kubeflow Components**: Python functions compiled into reusable Kubeflow Pipeline components
- **Modular Design**: Each pipeline step is a separate, testable component
- **Experiment Tracking**: Complete tracking of data lineage, model parameters, and evaluation metrics

## Prerequisites

- Python 3.10+
- pip
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/izenWarraich/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline Locally

Run the complete pipeline:
```bash
python pipeline.py
```

This will:
1. Extract data from `data/raw/boston.csv`
2. Preprocess the data (standardize features, split train/test)
3. Train a RandomForestRegressor model
4. Evaluate the model and generate metrics

### Compiling Kubeflow Components

To compile the Python functions into Kubeflow Pipeline component YAML files:
```bash
python compile_components.py
```

This generates YAML files in the `components/` directory that can be used in Kubeflow Pipelines.

### Individual Component Functions

You can also use the components individually:

```python
from src.pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

# Extract data
extracted_path = extract_data("data/raw/boston.csv", "data/processed/boston_raw.csv")

# Preprocess data
processed_path = preprocess_data(extracted_path, "data/processed/processed.pkl")

# Train model
model_path = train_model(processed_path, "models/model.pkl", n_estimators=100)

# Evaluate model
metrics_path = evaluate_model(processed_path, model_path)
```

## Pipeline Components

### 1. Data Extraction (`extract_data`)
- **Input**: Source and destination file paths
- **Output**: Destination file path
- **Function**: Copies CSV file and logs to MLflow

### 2. Data Preprocessing (`preprocess_data`)
- **Input**: Raw CSV file path, processed output path
- **Output**: Processed pickle file path
- **Function**: 
  - Loads CSV data
  - Standardizes features using StandardScaler
  - Splits into train/test sets (80/20)
  - Saves processed data as pickle file

### 3. Model Training (`train_model`)
- **Input**: Processed data path, model output path, n_estimators (default: 100)
- **Output**: Trained model file path
- **Function**:
  - Loads processed data
  - Trains RandomForestRegressor
  - Saves model as pickle file
  - Logs training metrics

### 4. Model Evaluation (`evaluate_model`)
- **Input**: Processed data path, model path
- **Output**: Metrics file path
- **Function**:
  - Loads model and test data
  - Computes predictions
  - Calculates MSE and R² metrics
  - Saves metrics to text file

## MLflow Tracking

All pipeline runs are tracked in MLflow. The pipeline creates:
- A parent run named "full_pipeline"
- Nested runs for each component (data_extraction, preprocessing, training, evaluation)

View MLflow UI:
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Dependencies

- **mlflow**: Experiment tracking and model registry
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **joblib**: Model serialization
- **kfp**: Kubeflow Pipelines SDK
- **dvc**: Data version control (optional)

## Dataset

The project uses the Boston Housing dataset, which contains:
- 506 samples
- 13 features (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- Target variable: MEDV (median value of owner-occupied homes)

## Model

The pipeline trains a **RandomForestRegressor** with:
- Default: 100 estimators
- Random state: 42 (for reproducibility)

## Output Files

After running the pipeline, you'll find:
- `data/processed/processed.pkl`: Preprocessed data (train/test splits, scaler)
- `models/model.pkl`: Trained RandomForestRegressor model
- `models/metrics.txt`: Evaluation metrics (MSE, R²)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of an MLOps assignment and is provided as-is for educational purposes.

## Author

**Izen Warraich**
- GitHub: [@izenWarraich](https://github.com/izenWarraich)

## Acknowledgments

- Boston Housing dataset
- Kubeflow Pipelines team
- MLflow team
