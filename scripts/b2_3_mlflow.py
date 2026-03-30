# b2 - model management (mlflow) + b3 - ml results (visualizations)
# reads everything from mlflow - no local files needed
# incremental: only runs if B1 has new training since last B2 run
# - tags the best model as "production"
# - generates visualizations to discuss ml results
# images are saved only to MLflow, no local reports folder

import os
import sys
import tempfile
import yaml
import glob
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.getcwd()

with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(BASE_DIR, config['paths']['scripts_dir']))
from pipeline_state_manager import load_state, save_state, get_last_run, set_last_run

MLRUNS_PATH = os.path.join(BASE_DIR, config['mlflow']['mlruns_dir'])
EXPERIMENT_NAME = config['mlflow']['experiment_name']

# load state and check for new training from B1
state = load_state(BASE_DIR)
last_training = get_last_run(state, "training")
last_mlflow = get_last_run(state, "mlflow")

training_timestamp = last_training.get('timestamp', '')

if not training_timestamp:
    print(f"> ERROR: No training found. Run b1_training.py first.")
    exit(1)

# check if B2 already processed this training
if last_mlflow:
    last_processed = last_mlflow.get('training_processed', '')
    if last_processed == training_timestamp:
        print(f"> Mode: SKIP (training from {training_timestamp[:19]} already processed)")
        print("\n> B2 completed (nothing new to process)")
        exit(0)
    else:
        print(f"> Mode: FULL (new training detected: {training_timestamp[:19]})")
else:
    print(f"> Mode: FULL (first B2 run)")

print(f"> MLflow tracking: {MLRUNS_PATH}")

# mlflow setup
print("\n[B.2] Setting up MLflow")
mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")

client = MlflowClient(tracking_uri=f"file:{MLRUNS_PATH}")
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"> ERROR: Experiment '{EXPERIMENT_NAME}' not found. Run b1_training.py first.")
    exit(1)

print(f"> Experiment: {EXPERIMENT_NAME}")
print(f"> Experiment ID: {experiment.experiment_id}")

# find the latest runs for each model (logged by B1)
print("\n[B.2] Finding latest model runs from B1")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

model_runs = {}
for run in runs:
    run_name = run.info.run_name
    if run_name and run_name not in model_runs:
        model_runs[run_name] = run
    if len(model_runs) >= 3:
        break

print(f"> Found {len(model_runs)} recent model runs")

# find best model and tag it
best_model_name = None
best_rmse = float('inf')
best_run_id = None

for run_name, run in model_runs.items():
    rmse = run.data.metrics.get("rmse", float('inf'))
    print(f"> {run_name}: RMSE = {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = run_name
        best_run_id = run.info.run_id

# tag best model as production
print(f"\n[B.2] Tagging best model as 'Production'")
print(f"> Best model: {best_model_name} (RMSE: {best_rmse:.4f})")

client.set_tag(best_run_id, "deployment", "Production")
client.set_tag(best_run_id, "best_model", "true")

print(f"> Tagged run {best_run_id[:8]}... as Production")

# download predictions from best run artifact
print("\n[B.3] Loading predictions from MLflow")

with tempfile.TemporaryDirectory() as temp_dir:
    predictions_dir = client.download_artifacts(best_run_id, "predictions", temp_dir)
    predictions_path = os.path.join(predictions_dir, "predictions.parquet")
    
    if not os.path.exists(predictions_path):
        parquet_files = glob.glob(os.path.join(predictions_dir, "**/*.parquet"), recursive=True)
        if parquet_files:
            predictions_df = pd.read_parquet(os.path.dirname(parquet_files[0]))
        else:
            print(f"> ERROR: predictions.parquet not found in artifacts")
            exit(1)
    else:
        predictions_df = pd.read_parquet(predictions_path)
    
    print(f"> Loaded {len(predictions_df)} predictions")

    # prepare metrics dataframe from MLflow runs
    metrics_data = []
    for run_name, run in model_runs.items():
        metrics_data.append({
            'Model': run_name,
            'RMSE': run.data.metrics.get("rmse", 0),
            'MAE': run.data.metrics.get("mae", 0),
            'R2': run.data.metrics.get("r2", 0)
        })
    metrics_df = pd.DataFrame(metrics_data)

    # get graphable data from each model run
    lr_coefficients = {}
    rf_importance = {}
    gbt_importance = {}
    
    for run_name, run in model_runs.items():
        for key, value in run.data.metrics.items():
            if key.startswith("coef_"):
                feature_name = key.replace("coef_", "")
                lr_coefficients[feature_name] = value
            elif key.startswith("importance_") and "Random Forest" in run_name:
                feature_name = key.replace("importance_", "")
                rf_importance[feature_name] = value
            elif key.startswith("importance_") and "GBT" in run_name:
                feature_name = key.replace("importance_", "")
                gbt_importance[feature_name] = value

    # create temp directory for visualizations (saved only to MLflow)
    viz_temp_dir = tempfile.mkdtemp()
    
    # visualization 1: model comparison
    print("\n[B.3] Generating model comparison chart")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Model Comparison - Regression Metrics', fontsize=14, fontweight='bold')

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    ax1 = axes[0]
    bars1 = ax1.bar(metrics_df['Model'], metrics_df['RMSE'], color=colors[:len(metrics_df)])
    ax1.set_ylabel('RMSE (m2)')
    ax1.set_title('Root Mean Square Error')
    ax1.tick_params(axis='x', rotation=15)

    ax2 = axes[1]
    bars2 = ax2.bar(metrics_df['Model'], metrics_df['MAE'], color=colors[:len(metrics_df)])
    ax2.set_ylabel('MAE (m2)')
    ax2.set_title('Mean Absolute Error')
    ax2.tick_params(axis='x', rotation=15)

    ax3 = axes[2]
    bars3 = ax3.bar(metrics_df['Model'], metrics_df['R2'], color=colors[:len(metrics_df)])
    ax3.set_ylabel('R2 Score')
    ax3.set_title('Coefficient of Determination')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    comparison_path = os.path.join(viz_temp_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"> Generated: model_comparison.png")

    # visualization 2a: linear regression coefficients
    print("\n[B.3] Generating Linear Regression coefficients chart")
    lr_coef_path = None

    if lr_coefficients:
        coef_df = pd.DataFrame({
            'Feature': list(lr_coefficients.keys()),
            'Coefficient': list(lr_coefficients.values())
        }).sort_values('Coefficient', key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_coef = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_df['Coefficient']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Linear Regression Coefficients', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        lr_coef_path = os.path.join(viz_temp_dir, "lr_coefficients.png")
        plt.savefig(lr_coef_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"> Generated: lr_coefficients.png")
    else:
        print("> Skipping LR coefficients (not available)")

    # visualization 2b: random forest feature importance
    print("\n[B.3] Generating Random Forest feature importance chart")
    rf_importance_path = None

    if rf_importance:
        rf_df = pd.DataFrame({
            'Feature': list(rf_importance.keys()),
            'Importance': list(rf_importance.values())
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(rf_df['Feature'], rf_df['Importance'], color='#3498db')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        rf_importance_path = os.path.join(viz_temp_dir, "rf_feature_importance.png")
        plt.savefig(rf_importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"> Generated: rf_feature_importance.png")
    else:
        print("> Skipping RF importance (not available)")

    # visualization 2c: gbt feature importance
    print("\n[B.3] Generating GBT feature importance chart")
    gbt_importance_path = None

    if gbt_importance:
        gbt_df = pd.DataFrame({
            'Feature': list(gbt_importance.keys()),
            'Importance': list(gbt_importance.values())
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(gbt_df['Feature'], gbt_df['Importance'], color='#e74c3c')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (GBT Regressor)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        gbt_importance_path = os.path.join(viz_temp_dir, "gbt_feature_importance.png")
        plt.savefig(gbt_importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"> Generated: gbt_feature_importance.png")
    else:
        print("> Skipping GBT importance (not available)")

    # visualization 3: predictions vs actual
    print("\n[B.3] Generating predictions scatter plot")

    models = predictions_df['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle('Predictions vs Actual Values', fontsize=14, fontweight='bold')

    for i, (model, color) in enumerate(zip(models, colors)):
        ax = axes[i]
        model_data = predictions_df[predictions_df['model'] == model]
        
        ax.scatter(model_data['actual'], model_data['predicted'], 
                   alpha=0.5, s=10, color=color)
        
        min_val = min(model_data['actual'].min(), model_data['predicted'].min())
        max_val = max(model_data['actual'].max(), model_data['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='Perfect')
        
        ax.set_xlabel('Actual Size (m2)')
        ax.set_ylabel('Predicted Size (m2)')
        ax.set_title(model)
        ax.legend()

    plt.tight_layout()
    scatter_path = os.path.join(viz_temp_dir, "predictions_scatter.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"> Generated: predictions_scatter.png")

    # visualization 4: residuals distribution
    print("\n[B.3] Generating residuals histogram")

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle('Residuals Distribution (Prediction Errors)', fontsize=14, fontweight='bold')

    for i, (model, color) in enumerate(zip(models, colors)):
        ax = axes[i]
        model_data = predictions_df[predictions_df['model'] == model]
        residuals = model_data['predicted'] - model_data['actual']
        
        ax.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.axvline(x=residuals.mean(), color='black', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {residuals.mean():.2f}')
        
        ax.set_xlabel('Residual (m2)')
        ax.set_ylabel('Frequency')
        ax.set_title(model)
        ax.legend()

    plt.tight_layout()
    residuals_path = os.path.join(viz_temp_dir, "residuals_histogram.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"> Generated: residuals_histogram.png")

    # log visualizations to MLflow
    print("\n[B.3] Logging visualizations to MLflow")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(comparison_path, "visualizations")
        mlflow.log_artifact(scatter_path, "visualizations")
        mlflow.log_artifact(residuals_path, "visualizations")
    print(f"> Comparative charts logged to best model run")

    for run_name, run in model_runs.items():
        with mlflow.start_run(run_id=run.info.run_id):
            if "Linear" in run_name and lr_coef_path:
                mlflow.log_artifact(lr_coef_path, "visualizations")
                print(f"> LR coefficients logged to {run_name}")
            elif "Random" in run_name and rf_importance_path:
                mlflow.log_artifact(rf_importance_path, "visualizations")
                print(f"> RF importance logged to {run_name}")
            elif "GBT" in run_name and gbt_importance_path:
                mlflow.log_artifact(gbt_importance_path, "visualizations")
                print(f"> GBT importance logged to {run_name}")

# summary
print("\n" + "="*70)
print("B.2 + B.3 COMPLETE")
print("="*70)
print("\n> MLflow:")
print(f">   Best model tagged as 'Production': {best_model_name}")
print(f"\n>   View MLflow UI: mlflow ui --backend-store-uri file:{MLRUNS_PATH}")

print("\n> Visualizations saved to MLflow:")
print(">   - model_comparison.png")
print(">   - lr_coefficients.png")
print(">   - rf_feature_importance.png")
print(">   - gbt_feature_importance.png")
print(">   - predictions_scatter.png")
print(">   - residuals_histogram.png")
print("="*70)

# update state
set_last_run(state, "mlflow", {
    "training_processed": training_timestamp,
    "timestamp": datetime.now().isoformat(),
    "best_model": best_model_name,
    "best_rmse": best_rmse,
    "visualizations_generated": [
        "model_comparison.png",
        "lr_coefficients.png", 
        "rf_feature_importance.png",
        "gbt_feature_importance.png",
        "predictions_scatter.png",
        "residuals_histogram.png"
    ]
})

save_state(state, BASE_DIR)
print(f"\n> State saved to pipeline_state.yaml")

print("\n> B2 completed")
