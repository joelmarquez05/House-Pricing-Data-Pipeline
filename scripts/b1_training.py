# b1 - model training and validation (pyspark + mlflow)
# trains and validates 3 regression models to predict property size (m2)
# models are logged to mlflow for versioning and management
# incremental: only re-trains when A5 has new data, but uses ALL data

import os
import sys
import yaml
import json
import shutil
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import (
    LinearRegression, LinearRegressionModel,
    RandomForestRegressor, RandomForestRegressionModel,
    GBTRegressor, GBTRegressionModel
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

BASE_DIR = os.getcwd()

with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(BASE_DIR, config['paths']['scripts_dir']))
from pipeline_state_manager import load_state, save_state, get_last_run, set_last_run

EXPLOITATION_PATH = os.path.join(BASE_DIR, config['paths']['exploitation_dir'])
MLRUNS_PATH = os.path.join(BASE_DIR, config['mlflow']['mlruns_dir'])
TEMP_PATH = os.path.join(BASE_DIR, "temp")
SEED = config['training']['seed']
EXPERIMENT_NAME = config['mlflow']['experiment_name']
TRAINING_YEARS_WINDOW = config['training']['years_window']

# load state and check for new data from A5
state = load_state(BASE_DIR)
last_exploitation = get_last_run(state, "exploitation")
last_training = get_last_run(state, "training")

exploitation_timestamp = last_exploitation.get("timestamp", "")
updated_years = last_exploitation.get("years_updated", [])

# check if this exploitation was already processed
if last_training:
    last_processed = last_training.get("exploitation_processed", "")
    if last_processed == exploitation_timestamp and exploitation_timestamp:
        print(f"> Mode: SKIP (exploitation from {exploitation_timestamp[:19]} already trained)")
        print("\n> B1 completed (nothing new to train)")
        exit(0)

if len(updated_years) == 0:
    print("> Mode: SKIP (A5 had no new data)")
    print("\n> B1 completed (nothing to train)")
    exit(0)

# spark session
spark = SparkSession.builder \
    .appName("B1_Training") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .getOrCreate()

spark.sparkContext._jsc.hadoopConfiguration().set("fs.file.impl.disable.cache", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.local.block.size", "134217728")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
spark.sparkContext.setLogLevel("WARN")

# determine which years to use for training (last N available)
print("\n[B.1] Determining training years")
df_all_years = spark.read.format("delta").load(EXPLOITATION_PATH).select("year").distinct()
available_years = sorted([r.year for r in df_all_years.collect()])
years_to_train = available_years[-TRAINING_YEARS_WINDOW:]

print(f"> Available years in Delta: {available_years}")
print(f"> Training window: last {TRAINING_YEARS_WINDOW} years = {years_to_train}")
print(f"> Updated years from A5: {updated_years}")

# check if any updated year affects training data
relevant_updates = [y for y in updated_years if y in years_to_train]
if not relevant_updates:
    print(f"> Mode: SKIP (updated years {updated_years} are outside training range {years_to_train})")
    print("\n> B1 completed (no relevant updates)")
    spark.stop()
    exit(0)

print(f"> Relevant updates: {relevant_updates} → retraining required")

print(f"\n> Reading from: {EXPLOITATION_PATH}")
print(f"> Models will be logged to MLflow")

# load only training years (partition pruning for efficiency)
print("\n[B.1] Loading data from Delta Lake (partition pruning)")
df = spark.read.format("delta").load(EXPLOITATION_PATH).filter(col("year").isin(years_to_train))
df.cache()
df_row_count = df.count()
print(f"> Loaded {df_row_count} rows from years {years_to_train}")

# feature engineering
print("\n[B.1] Feature engineering")

status_indexer = StringIndexer(inputCol="status", outputCol="status_idx", handleInvalid="keep")
property_indexer = StringIndexer(inputCol="property_type", outputCol="property_type_idx", handleInvalid="keep")

numeric_features = [
    "price", "rooms", "bathrooms", "has_lift", "is_exterior", 
    "is_new_dev", "floor_clean", "income_index", "total_unemployed", "density_inh_ha"
]

all_features = numeric_features + ["status_idx", "property_type_idx"]

assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
preprocessing = Pipeline(stages=[status_indexer, property_indexer, assembler])

df_clean = df.na.drop(subset=["label_size"] + numeric_features)
preprocessing_model = preprocessing.fit(df_clean)
df_prepared = preprocessing_model.transform(df_clean).select("features", col("label_size").alias("label"))
df_prepared.cache()
prepared_count = df_prepared.count()

print(f"> Features: {all_features}")
print(f"> Prepared {prepared_count} rows")

# train/validation split
print("\n[B.1] Splitting data (80/20)")
train_df, val_df = df_prepared.randomSplit([0.8, 0.2], seed=SEED)
train_df.cache()
val_df.cache()
print(f"> Training: ~{int(prepared_count * 0.8)} rows (80%)")
print(f"> Validation: ~{int(prepared_count * 0.2)} rows (20%)")

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# model 1: linear regression
print("\n[B.1] Training Linear Regression with CrossValidator")
lr = LinearRegression(labelCol="label", featuresCol="features")

lr_param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

lr_cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=lr_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=SEED
)

lr_cv_model = lr_cv.fit(train_df)
lr_best = lr_cv_model.bestModel
lr_predictions = lr_best.transform(val_df)

lr_rmse = evaluator.evaluate(lr_predictions, {evaluator.metricName: "rmse"})
lr_mae = evaluator.evaluate(lr_predictions, {evaluator.metricName: "mae"})
lr_r2 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "r2"})

print(f"> Best regParam: {lr_best.getRegParam()}")
print(f"> Best elasticNetParam: {lr_best.getElasticNetParam()}")
print(f"> RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R2: {lr_r2:.4f}")

# model 2: random forest
print("\n[B.1] Training Random Forest with CrossValidator")
rf = RandomForestRegressor(labelCol="label", featuresCol="features", seed=SEED)

rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=SEED
)

rf_cv_model = rf_cv.fit(train_df)
rf_best = rf_cv_model.bestModel
rf_predictions = rf_best.transform(val_df)

rf_rmse = evaluator.evaluate(rf_predictions, {evaluator.metricName: "rmse"})
rf_mae = evaluator.evaluate(rf_predictions, {evaluator.metricName: "mae"})
rf_r2 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "r2"})

print(f"> Best numTrees: {rf_best.getNumTrees}")
print(f"> Best maxDepth: {rf_best.getMaxDepth()}")
print(f"> RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R2: {rf_r2:.4f}")

print("> Feature Importance:")
for i, importance in enumerate(rf_best.featureImportances.toArray()):
    if importance > 0.01:
        print(f">   {all_features[i]}: {importance:.4f}")

# model 3: gradient-boosted trees
print("\n[B.1] Training GBT Regressor with CrossValidator")
gbt = GBTRegressor(labelCol="label", featuresCol="features", seed=SEED)

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [20, 50]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .build()

gbt_cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator,
    numFolds=2,
    seed=SEED
)

gbt_cv_model = gbt_cv.fit(train_df)
gbt_best = gbt_cv_model.bestModel
gbt_predictions = gbt_best.transform(val_df)

gbt_rmse = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "rmse"})
gbt_mae = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "mae"})
gbt_r2 = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "r2"})

print(f"> Best maxIter: {gbt_best.getMaxIter()}")
print(f"> Best maxDepth: {gbt_best.getMaxDepth()}")
print(f"> RMSE: {gbt_rmse:.4f}, MAE: {gbt_mae:.4f}, R2: {gbt_r2:.4f}")

# log models to mlflow
print("\n[B.1] Logging models to MLflow")

mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"> MLflow experiment: {EXPERIMENT_NAME}")

run_ids = {}

temp_models_dir = os.path.join(BASE_DIR, "temp_models")
os.makedirs(temp_models_dir, exist_ok=True)

# log linear regression
with mlflow.start_run(run_name="Linear Regression") as run:
    mlflow.log_param("regParam", lr_best.getRegParam())
    mlflow.log_param("elasticNetParam", lr_best.getElasticNetParam())
    mlflow.log_metric("rmse", lr_rmse)
    mlflow.log_metric("mae", lr_mae)
    mlflow.log_metric("r2", lr_r2)
    coefficients = lr_best.coefficients.toArray()
    for i, coef in enumerate(coefficients):
        mlflow.log_metric(f"coef_{all_features[i]}", coef)
    mlflow.log_metric("intercept", lr_best.intercept)
    lr_temp_path = os.path.join(temp_models_dir, "linear_regression")
    lr_best.write().overwrite().save(lr_temp_path)
    mlflow.log_artifacts(lr_temp_path, "model")
    run_ids["linear_regression"] = run.info.run_id
    print(f"> Logged Linear Regression (run: {run.info.run_id[:8]}...)")

# log random forest
with mlflow.start_run(run_name="Random Forest") as run:
    mlflow.log_param("numTrees", rf_best.getNumTrees)
    mlflow.log_param("maxDepth", rf_best.getMaxDepth())
    mlflow.log_metric("rmse", rf_rmse)
    mlflow.log_metric("mae", rf_mae)
    mlflow.log_metric("r2", rf_r2)
    for i, importance in enumerate(rf_best.featureImportances.toArray()):
        if importance > 0.01:
            mlflow.log_metric(f"importance_{all_features[i]}", importance)
    rf_temp_path = os.path.join(temp_models_dir, "random_forest")
    rf_best.write().overwrite().save(rf_temp_path)
    mlflow.log_artifacts(rf_temp_path, "model")
    run_ids["random_forest"] = run.info.run_id
    print(f"> Logged Random Forest (run: {run.info.run_id[:8]}...)")

# log gbt regressor
with mlflow.start_run(run_name="GBT Regressor") as run:
    mlflow.log_param("maxIter", gbt_best.getMaxIter())
    mlflow.log_param("maxDepth", gbt_best.getMaxDepth())
    mlflow.log_metric("rmse", gbt_rmse)
    mlflow.log_metric("mae", gbt_mae)
    mlflow.log_metric("r2", gbt_r2)
    for i, importance in enumerate(gbt_best.featureImportances.toArray()):
        if importance > 0.01:
            mlflow.log_metric(f"importance_{all_features[i]}", importance)
    gbt_temp_path = os.path.join(temp_models_dir, "gbt_regressor")
    gbt_best.write().overwrite().save(gbt_temp_path)
    mlflow.log_artifacts(gbt_temp_path, "model")
    run_ids["gbt_regressor"] = run.info.run_id
    print(f"> Logged GBT Regressor (run: {run.info.run_id[:8]}...)")

shutil.rmtree(temp_models_dir, ignore_errors=True)

# save predictions as MLflow artifact
print("\n[B.1] Saving predictions to MLflow")

lr_pred_df = lr_predictions.select(
    col("label").alias("actual"),
    col("prediction").alias("predicted")
).withColumn("model", lit("Linear Regression"))

rf_pred_df = rf_predictions.select(
    col("label").alias("actual"),
    col("prediction").alias("predicted")
).withColumn("model", lit("Random Forest"))

gbt_pred_df = gbt_predictions.select(
    col("label").alias("actual"),
    col("prediction").alias("predicted")
).withColumn("model", lit("GBT Regressor"))

all_predictions = lr_pred_df.union(rf_pred_df).union(gbt_pred_df)

os.makedirs(TEMP_PATH, exist_ok=True)
predictions_temp = os.path.join(TEMP_PATH, "predictions.parquet")
all_predictions.write.mode("overwrite").parquet(predictions_temp)

best_run_id = min(
    [(run_ids["linear_regression"], lr_rmse),
     (run_ids["random_forest"], rf_rmse),
     (run_ids["gbt_regressor"], gbt_rmse)],
    key=lambda x: x[1]
)[0]

with mlflow.start_run(run_id=best_run_id):
    mlflow.log_artifact(predictions_temp, "predictions")
    mlflow.log_param("feature_names", ",".join(all_features))
    for i, importance in enumerate(rf_best.featureImportances.toArray()):
        if importance > 0.01:
            mlflow.log_metric(f"rf_importance_{all_features[i]}", importance)

print(f"> Predictions logged to run: {best_run_id[:8]}...")

shutil.rmtree(TEMP_PATH, ignore_errors=True)

# summary
print("\n" + "="*70)
print("B.1 TRAINING COMPLETE - MODEL COMPARISON")
print("="*70)
print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
print("-"*70)
print(f"{'Linear Regression':<25} {lr_rmse:>10.4f} {lr_mae:>10.4f} {lr_r2:>10.4f}")
print(f"{'Random Forest':<25} {rf_rmse:>10.4f} {rf_mae:>10.4f} {rf_r2:>10.4f}")
print(f"{'GBT Regressor':<25} {gbt_rmse:>10.4f} {gbt_mae:>10.4f} {gbt_r2:>10.4f}")
print("="*70)

best_model = min(
    [("Linear Regression", lr_rmse), ("Random Forest", rf_rmse), ("GBT Regressor", gbt_rmse)],
    key=lambda x: x[1]
)
print(f"\n> Best model: {best_model[0]} (RMSE: {best_model[1]:.4f})")

# update state for B2
set_last_run(state, "training", {
    "timestamp": datetime.now().isoformat(),
    "exploitation_processed": exploitation_timestamp,
    "models": ["linear_regression", "random_forest", "gbt_regressor"],
    "best_model": best_model[0].lower().replace(" ", "_"),
    "best_rmse": best_model[1],
    "years_trained": years_to_train,
    "triggered_by_years": relevant_updates
})

save_state(state, BASE_DIR)
print(f"> State saved to pipeline_state.yaml")

print("\n> B1 completed")
spark.stop()
