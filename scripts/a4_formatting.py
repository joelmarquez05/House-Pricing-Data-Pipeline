import os
import sys
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, sum as _sum
from pymongo import MongoClient
from datetime import datetime

BASE_DIR = os.getcwd()

with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(BASE_DIR, config['paths']['scripts_dir']))
from pipeline_state_manager import load_state, save_state, get_last_run, set_last_run, clear_last_run

LANDING_PATH = os.path.join(BASE_DIR, config['paths']['landing_dir'])
MONGO_URI = config['mongodb']['uri']
MONGO_DB = config['mongodb']['database']

# load state and check for new data from A3
state = load_state(BASE_DIR)
last_ingestion = get_last_run(state, "ingestion")
partitions_updated = last_ingestion.get("partitions_updated", {})

if len(partitions_updated) == 0:
    print("> Mode: SKIP (A3 had no new data)")
    print("\n> A4 completed (nothing to process)")
    exit(0)

print(f"> Reading from: {LANDING_PATH}")
print(f"> MongoDB: {MONGO_URI}/{MONGO_DB}")
print("> Mode: INCREMENTAL")

spark = SparkSession.builder \
    .appName("A4_Formatting") \
    .master("local[*]") \
    .config("spark.mongodb.input.uri", f"{MONGO_URI}/{MONGO_DB}") \
    .config("spark.mongodb.output.uri", f"{MONGO_URI}/{MONGO_DB}") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def save_to_mongo_incremental(df, collection_name, partition_cols):
    print(f"> Saving to '{collection_name}' (incremental by {partition_cols})")
    
    partitions = df.select(partition_cols).distinct().collect()
    
    if not partitions:
        print("> No data to save")
        return
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[collection_name]
    
    total_deleted = 0
    for partition in partitions:
        delete_filter = {c: partition[c] for c in partition_cols}
        result = collection.delete_many(delete_filter)
        total_deleted += result.deleted_count
    
    client.close()
    
    df.write.format("mongo") \
        .mode("append") \
        .option("database", MONGO_DB) \
        .option("collection", collection_name) \
        .save()
    
    print(f"> Inserted {df.count()} documents")


def save_to_mongo_overwrite(df, collection_name):
    print(f"> Saving to '{collection_name}' (full overwrite)")
    df.write.format("mongo") \
        .mode("overwrite") \
        .option("database", MONGO_DB) \
        .option("collection", collection_name) \
        .save()
    print("> Done")


def build_partition_paths(base_path, dataset_name):
    if dataset_name not in partitions_updated:
        return None
    
    partitions = partitions_updated[dataset_name]
    paths = []
    
    for p in partitions:
        if "day" in p:
            path = f"{base_path}/year={p['year']}/month={p['month']}/day={p['day']}"
        else:
            path = f"{base_path}/year={p['year']}"
        paths.append(path)
    
    print(f"> Reading {len(paths)} partition(s) directly")
    return paths


# load lookup table for neighborhood name standardization
print("\n[A.4] Loading lookup table")
df_lookup = spark.read.csv(f"{LANDING_PATH}/lookup", header=True, inferSchema=True)

lookup_clean = df_lookup.select(
    col("neighborhood").alias("original_name"),
    col("neighborhood_n_reconciled").alias("official_name"),
    col("district_n_reconciled").alias("official_district_name")
).dropDuplicates(["original_name"])
lookup_clean.cache()
print(f"> Lookup table cached ({lookup_clean.count()} neighborhoods)")


# income (annual data - partition by year)
print("\n[A.4] Processing INCOME (1/4)")
income_paths = build_partition_paths(f"{LANDING_PATH}/income", "income")
if income_paths is None:
    print("> Skipping: no new income data")
    df_income = None
else:
    df_income = spark.read.option("basePath", f"{LANDING_PATH}/income").csv(income_paths, header=True, inferSchema=True)

if df_income is not None:
    rfd_col_name = "Índex RFD Barcelona = 100"
    
    df_income_selected = df_income.select(
        col("Nom_Barri").alias("raw_neighborhood"),
        col("Nom_Districte").alias("raw_district"),
        col(rfd_col_name).cast("double").alias("income_index"),
        col("Any").cast("integer").alias("year")
    )
    
    if lookup_clean:
        df_income_formatted = df_income_selected.join(
            lookup_clean,
            df_income_selected["raw_neighborhood"] == lookup_clean["original_name"],
            "left"
        ).withColumn("neighborhood", col("official_name")) \
         .withColumn("district", col("official_district_name")) \
         .drop("raw_neighborhood", "raw_district", "original_name", "official_district_name", "official_name")
        print("> Neighborhood names standardized")
    else:
        df_income_formatted = df_income_selected.withColumnRenamed("raw_neighborhood", "neighborhood_name")
    
    save_to_mongo_incremental(df_income_formatted, "income", ["year"])


# unemployment (annual data - partition by year)
print("\n[A.4] Processing UNEMPLOYMENT (2/4)")
unemp_paths = build_partition_paths(f"{LANDING_PATH}/unemployment", "unemployment")
if unemp_paths is None:
    print("> Skipping: no new unemployment data")
    df_unemp_raw = None
else:
    df_unemp_raw = spark.read.option("basePath", f"{LANDING_PATH}/unemployment").json(unemp_paths)

if df_unemp_raw is not None:
    df_flat = df_unemp_raw.select(explode(col("result.records")).alias("data"))
    
    df_cols = df_flat.select(
        col("data.Any").cast("integer").alias("year"),
        col("data.Mes").cast("integer").alias("month"),
        col("data.Nom_Barri").alias("raw_neighborhood"),
        col("data.Nom_Districte").alias("raw_district"),
        col("data.Nombre").cast("integer").alias("unemployed_count")
    )
    
    df_unemp_grouped = df_cols.groupBy("raw_neighborhood", "year").agg(_sum("unemployed_count").alias("total_unemployed"))
    
    if lookup_clean:
        df_unemp_formatted = df_unemp_grouped.join(
            lookup_clean,
            df_unemp_grouped["raw_neighborhood"] == lookup_clean["original_name"],
            "left"
        ).withColumn("neighborhood", col("official_name")) \
         .withColumn("district", col("official_district_name")) \
         .drop("raw_neighborhood", "raw_district", "original_name", "official_name", "official_district_name")
        print("> Neighborhood names standardized")
    else:
        df_unemp_formatted = df_unemp_grouped.withColumnRenamed("raw_neighborhood", "neighborhood")
    
    save_to_mongo_incremental(df_unemp_formatted, "unemployment", ["year"])


# density (annual data - partition by year)
print("\n[A.4] Processing DENSITY (3/4)")
density_paths = build_partition_paths(f"{LANDING_PATH}/density", "density")
if density_paths is None:
    print("> Skipping: no new density data")
    df_density_raw = None
else:
    df_density_raw = spark.read.option("basePath", f"{LANDING_PATH}/density").json(density_paths)

if df_density_raw is not None:
    df_density_selected = df_density_raw.select(
        col("Nom_Barri").alias("raw_neighborhood"),
        col("Nom_Districte").alias("raw_district"),
        col("Any").cast("integer").alias("year"),
        col("Població").cast("integer").alias("population"),
        col("Densitat (hab/ha)").cast("double").alias("density_inh_ha")
    )
    
    if lookup_clean:
        df_density_formatted = df_density_selected.join(
            lookup_clean,
            df_density_selected["raw_neighborhood"] == lookup_clean["original_name"],
            "left"
        ).withColumn("neighborhood", col("official_name")) \
         .withColumn("district", col("official_district_name")) \
         .drop("raw_neighborhood", "raw_district", "original_name", "official_name", "official_district_name")
        print("> Neighborhood names standardized")
    else:
        df_density_formatted = df_density_selected.withColumnRenamed("raw_neighborhood", "neighborhood")
    
    save_to_mongo_incremental(df_density_formatted, "density", ["year"])


# idealista (daily data - partition by year/month/day)
print("\n[A.4] Processing IDEALISTA (4/4)")
idealista_paths = build_partition_paths(f"{LANDING_PATH}/idealista", "idealista")
if idealista_paths is None:
    print("> Skipping: no new idealista data")
    df_idealista = None
else:
    df_idealista = spark.read.option("basePath", f"{LANDING_PATH}/idealista").json(idealista_paths)
    print(f"> Loaded {len(idealista_paths)} partition(s)")

if df_idealista is not None:
    df_idealista_selected = df_idealista.select(
        col("propertyCode").alias("property_id"),
        col("year").cast("integer"),
        col("month").cast("integer"),
        col("day").cast("integer"),
        col("address"),
        col("neighborhood").alias("raw_neighborhood"),
        col("district"),
        col("municipality"),
        col("province"),
        col("latitude").cast("double"),
        col("longitude").cast("double"),
        col("operation"),
        col("propertyType").alias("property_type"),
        col("detailedType.typology").alias("typology"),
        col("detailedType.subTypology").alias("sub_typology"),
        col("price").cast("double"),
        col("size").cast("double"),
        col("priceByArea").cast("double").alias("price_by_area"),
        col("rooms").cast("integer"),
        col("bathrooms").cast("integer"),
        col("floor"),
        col("status"),
        col("exterior").cast("boolean"),
        col("hasLift").cast("boolean"),
        col("newDevelopment").alias("new_development")
    )
    
    if lookup_clean:
        df_idealista_formatted = df_idealista_selected.join(
            lookup_clean,
            df_idealista_selected["raw_neighborhood"] == lookup_clean["original_name"],
            "left"
        ).withColumn("neighborhood", col("official_name")) \
         .withColumn("district", col("official_district_name")) \
         .drop("raw_neighborhood", "original_name", "official_name", "official_district_name")
        print("> Neighborhood names standardized")
    else:
        df_idealista_formatted = df_idealista_selected.withColumnRenamed("raw_neighborhood", "neighborhood_name")
    
    save_to_mongo_incremental(df_idealista_formatted, "idealista", ["year", "month", "day"])

# collect updated years for A5
print("\n[A.4] Saving state for A5")
updated_years = set()

datasets_to_check = [
    (df_income, "income"),
    (df_unemp_raw, "unemployment"),
    (df_density_raw, "density"),
    (df_idealista, "idealista")
]

for df, name in datasets_to_check:
    if df is not None:
        years = [row.year for row in df.select("year").distinct().collect()]
        updated_years.update(years)
        print(f"> {name}: years {sorted(years)}")
    else:
        print(f"> {name}: skipped (no new data)")

# update state for A5
set_last_run(state, "formatting", {
    "timestamp": datetime.now().isoformat(),
    "years_updated": sorted(list(updated_years))
})

# clear ingestion data (consumed)
clear_last_run(state, "ingestion")

save_state(state, BASE_DIR)
print(f"> State saved to pipeline_state.yaml")

print("\n> A4 completed")
spark.stop()