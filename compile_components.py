"""
Script to compile Python functions into Kubeflow Pipeline components (YAML files).
"""
from kfp.components import create_component_from_func
from pathlib import Path
import sys

# Add src to path to import pipeline_components
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

# Create components directory if it doesn't exist
components_dir = Path("components")
components_dir.mkdir(exist_ok=True)

# Define base image with required dependencies
base_image = "python:3.10"
packages = [
    "mlflow",
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib"
]

print("Compiling components to YAML files...\n")

# Compile extract_data component
extract_data_component = create_component_from_func(
    func=extract_data,
    output_component_file=str(components_dir / "extract_data.yaml"),
    base_image=base_image,
    packages_to_install=packages
)
print(f"✓ Compiled extract_data component to {components_dir / 'extract_data.yaml'}")

# Compile preprocess_data component
preprocess_data_component = create_component_from_func(
    func=preprocess_data,
    output_component_file=str(components_dir / "preprocess_data.yaml"),
    base_image=base_image,
    packages_to_install=packages
)
print(f"✓ Compiled preprocess_data component to {components_dir / 'preprocess_data.yaml'}")

# Compile train_model component
train_model_component = create_component_from_func(
    func=train_model,
    output_component_file=str(components_dir / "train_model.yaml"),
    base_image=base_image,
    packages_to_install=packages
)
print(f"✓ Compiled train_model component to {components_dir / 'train_model.yaml'}")

# Compile evaluate_model component
evaluate_model_component = create_component_from_func(
    func=evaluate_model,
    output_component_file=str(components_dir / "evaluate_model.yaml"),
    base_image=base_image,
    packages_to_install=packages
)
print(f"✓ Compiled evaluate_model component to {components_dir / 'evaluate_model.yaml'}")

print("\n✅ All components compiled successfully!")
