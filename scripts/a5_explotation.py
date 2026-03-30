import os
import sys
import yaml
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, row_number, desc, when, regexp_extract, lit, avg, max as _max, last, first, lower, abs as spark_abs
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

BASE_DIR = os.getcwd()

with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(BASE_DIR, config['paths']['scripts_dir']))
from pipeline_state_manager import load_state, save_state, get_last_run, set_last_run, clear_last_run


def find_closest_year_past_priority(idealista_years_df, source_years_df):
    if source_years_df.count() == 0:
        raise ValueError("Source dataset has no years available for mapping")
    
    source_years = source_years_df.withColumnRenamed("year", "source_year")
    
    df_year_mapping = idealista_years_df.crossJoin(source_years).withColumn(
        "abs_diff", spark_abs(col("year") - col("source_year"))
    )
    
    window = Window.partitionBy("year").orderBy("abs_diff", "source_year")
    
    df_closest = df_year_mapping.withColumn("rank", row_number().over(window)) \
        .filter(col("rank") == 1) \
        .select("year", "source_year")
    
    return df_closest


def get_interpolated_income(idealista_years_df, income_df):
    if income_df.count() == 0:
        raise ValueError("Income dataset is empty, cannot interpolate")
    
    income_years = income_df.select("year", "neighborhood", "income_index") \
        .withColumnRenamed("year", "inc_year")
    
    from pyspark.sql.functions import min as _min
    year_bounds = income_df.groupBy("neighborhood").agg(
        _min("year").alias("min_inc_year"),
        _max("year").alias("max_inc_year")
    )
    
    neighborhoods = income_df.select("neighborhood").distinct()
    base = idealista_years_df.crossJoin(neighborhoods)
    base_with_bounds = base.join(year_bounds, on="neighborhood", how="left")
    
    income_prev = income_years.withColumnRenamed("inc_year", "prev_year") \
        .withColumnRenamed("income_index", "prev_income") \
        .withColumnRenamed("neighborhood", "prev_nb")
    
    income_next = income_years.withColumnRenamed("inc_year", "next_year") \
        .withColumnRenamed("income_index", "next_income") \
        .withColumnRenamed("neighborhood", "next_nb")
    
    df_with_prev = base_with_bounds.join(
        income_prev,
        (base_with_bounds["neighborhood"] == income_prev["prev_nb"]) & 
        (income_prev["prev_year"] <= base_with_bounds["year"]),
        "left"
    )
    
    window_prev = Window.partitionBy("year", "neighborhood").orderBy(desc("prev_year"))
    df_with_prev = df_with_prev.withColumn("rank_prev", row_number().over(window_prev)) \
        .filter(col("rank_prev") == 1) \
        .drop("rank_prev", "prev_nb")
    
    df_with_both = df_with_prev.join(
        income_next,
        (df_with_prev["neighborhood"] == income_next["next_nb"]) & 
        (income_next["next_year"] >= df_with_prev["year"]),
        "left"
    )
    
    window_next = Window.partitionBy("year", "neighborhood").orderBy("next_year")
    df_with_both = df_with_both.withColumn("rank_next", row_number().over(window_next)) \
        .filter(col("rank_next") == 1) \
        .drop("rank_next", "next_nb")
    
    df_interpolated = df_with_both.withColumn(
        "income_index",
        when(col("prev_year") == col("next_year"), col("prev_income"))
        .when(col("prev_year").isNull(), col("next_income"))
        .when(col("next_year").isNull(), col("prev_income"))
        .otherwise(
            (col("prev_income") * (col("next_year") - col("year")) + 
             col("next_income") * (col("year") - col("prev_year"))) / 
            (col("next_year") - col("prev_year"))
        )
    ).select("year", "neighborhood", "income_index")
    
    return df_interpolated


EXPLOITATION_PATH = os.path.join(BASE_DIR, config['paths']['exploitation_dir'])
MONGO_URI = config['mongodb']['uri']
MONGO_DB = config['mongodb']['database']

# load state and check for new data from A4
state = load_state(BASE_DIR)
last_formatting = get_last_run(state, "formatting")
updated_years = last_formatting.get("years_updated", [])

if len(updated_years) == 0:
    print("> Mode: SKIP (A4 had no new data)")
    print("\n> A5 completed (nothing to process)")
    exit(0)

print(f"> Target exploitation zone: {EXPLOITATION_PATH}")
print(f"> MongoDB: {MONGO_URI}/{MONGO_DB}")
print(f"> Mode: INCREMENTAL (years: {updated_years})")

spark = SparkSession.builder \
    .appName("A5_Exploitation") \
    .master("local[*]") \
    .config("spark.mongodb.input.uri", f"{MONGO_URI}/{MONGO_DB}") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def read_mongo(collection_name, year_filter=None):
    reader = spark.read.format("mongo") \
        .option("database", MONGO_DB) \
        .option("collection", collection_name)
    
    if year_filter:
        pipeline = json.dumps([{"$match": {"year": {"$in": year_filter}}}])
        reader = reader.option("pipeline", pipeline)
    
    return reader.load()


# load data from formatted zone
print("\n[A.5] Reading collections from MongoDB")


def load_aux_data_with_dynamic_buffer(collection_name, updated_years, initial_buffer=2, max_buffer=10):
    buffer = initial_buffer
    
    while buffer <= max_buffer:
        buffer_years = set()
        for y in updated_years:
            buffer_years.update(range(y - buffer, y + buffer + 1))
        buffer_years_list = sorted(list(buffer_years))
        
        df = read_mongo(collection_name, year_filter=buffer_years_list)
        
        if not df.isEmpty():
            print(f"> {collection_name}: found data with ±{buffer} year buffer ({min(buffer_years_list)}-{max(buffer_years_list)})")
            return df
        
        print(f"> {collection_name}: no data with ±{buffer} buffer, expanding...")
        buffer += 2
    
    raise ValueError(f"No {collection_name} data found within ±{max_buffer} years of {updated_years}")


# load auxiliary data with dynamic buffer
df_income = load_aux_data_with_dynamic_buffer("income", updated_years)
df_unemployment = load_aux_data_with_dynamic_buffer("unemployment", updated_years)
df_density = load_aux_data_with_dynamic_buffer("density", updated_years)

# for idealista, filter only to updated years
df_idealista = read_mongo("idealista", year_filter=updated_years)
print(f"> Idealista filtered to years: {updated_years} (push-down)")

if df_idealista.isEmpty():
    raise ValueError("No Idealista data found to process")


# clean idealista data
print("\n[A.5] Cleaning Idealista data")

df_clean = df_idealista.withColumn(
    "floor_clean",
    when(col("floor").rlike("(?i)bajos|bj|ground"), 0)
    .when(col("floor").rlike("(?i)entresuelo|en"), 0)
    .when(col("floor").rlike("(?i)sotano|st|ss"), -1)
    .otherwise(regexp_extract(col("floor"), r"(\d+)", 1).cast(IntegerType()))
)

df_prepared = df_clean.select(
    col("neighborhood"),
    col("district"),
    col("municipality"),
    col("year"),
    col("size").alias("label_size"),
    col("price"),
    col("rooms"),
    col("bathrooms"),
    col("hasLift").cast("integer").alias("has_lift"),
    col("exterior").cast("integer").alias("is_exterior"),
    col("new_development").cast("integer").alias("is_new_dev"),
    col("floor_clean"),
    col("status"),
    col("property_type"),
)

df_prepared = df_prepared.filter(lower(col("municipality")) == "barcelona")
df_prepared = df_prepared.filter((col("label_size") > 5) & (col("label_size") < 1000))
df_prepared = df_prepared.na.drop(subset=["label_size", "price"])
df_prepared.cache()
print(f"> After cleaning (Barcelona only, size 5-1000, no nulls): {df_prepared.count()} rows")


# prepare lookups for joins
print("\n[A.5] Preparing lookup tables")

avg_unemp = df_unemployment.agg(avg("total_unemployed")).collect()[0][0]
avg_dens = df_density.agg(avg("density_inh_ha")).collect()[0][0]

idealista_years = df_prepared.select("year").distinct()

# income: interpolation
print("> Processing income data (interpolated)")
df_income_valid = df_income.filter(col("income_index").isNotNull())
df_income_final = get_interpolated_income(idealista_years, df_income_valid)

# unemployment: closest year with past priority on tie
print("> Processing unemployment data")
window_unemp = Window.partitionBy("neighborhood").orderBy("year")

df_unemp_filled = df_unemployment.withColumn(
    "prev_val",
    last("total_unemployed", ignorenulls=True).over(window_unemp.rowsBetween(Window.unboundedPreceding, -1))
).withColumn(
    "next_val",
    first("total_unemployed", ignorenulls=True).over(window_unemp.rowsBetween(1, Window.unboundedFollowing))
)

df_unemp_interp = df_unemp_filled.withColumn(
    "total_unemployed_clean",
    when(col("total_unemployed").isNotNull(), col("total_unemployed"))
    .when(col("prev_val").isNotNull() & col("next_val").isNotNull(), (col("prev_val") + col("next_val")) / 2)
    .otherwise(coalesce(col("prev_val"), col("next_val"), lit(avg_unemp)))
).select(
    col("neighborhood").alias("unemp_neighborhood"), 
    col("year").alias("unemp_year"), 
    col("total_unemployed_clean").alias("total_unemployed")
)

unemp_year_mapping = find_closest_year_past_priority(
    idealista_years,
    df_unemployment.select("year").distinct()
)

df_unemp_final = unemp_year_mapping.join(
    df_unemp_interp,
    unemp_year_mapping["source_year"] == df_unemp_interp["unemp_year"],
    "left"
).select(
    col("year"),
    col("unemp_neighborhood").alias("neighborhood"),
    col("total_unemployed")
)

# density: closest year with past priority on tie
print("> Processing density data")
df_density_clean = df_density.select(
    col("neighborhood").alias("dens_neighborhood"), 
    col("year").alias("dens_year"), 
    col("density_inh_ha")
).na.fill({"density_inh_ha": avg_dens})

density_year_mapping = find_closest_year_past_priority(
    idealista_years,
    df_density.select("year").distinct()
)

df_density_final = density_year_mapping.join(
    df_density_clean,
    density_year_mapping["source_year"] == df_density_clean["dens_year"],
    "left"
).select(
    col("year"),
    col("dens_neighborhood").alias("neighborhood"),
    col("density_inh_ha")
)


# data enrichment through joins
print("\n[A.5] Merging datasets")

df_joined_1 = df_prepared.join(df_income_final, on=["neighborhood", "year"], how="left")
df_joined_2 = df_joined_1.join(df_unemp_final, on=["neighborhood", "year"], how="left")
df_final_enrichment = df_joined_2.join(df_density_final, on=["neighborhood", "year"], how="left")

df_final = df_final_enrichment.na.fill({"floor_clean": 1})


# write to exploitation zone
print("\n[A.5] Writing Delta table")

delta_log_path = os.path.join(EXPLOITATION_PATH, "_delta_log")
delta_table_exists = os.path.exists(delta_log_path)

rows_written = df_final.count()

if delta_table_exists:
    years_condition = " OR ".join([f"year = {y}" for y in updated_years])
    print(f"> Replacing partitions where: {years_condition}")
    df_final.write.format("delta") \
        .partitionBy("year") \
        .option("replaceWhere", years_condition) \
        .mode("overwrite") \
        .save(EXPLOITATION_PATH)
else:
    print("> Full overwrite (first run)")
    df_final.write.format("delta").partitionBy("year").mode("overwrite").save(EXPLOITATION_PATH)

print(f"> Saved to: {EXPLOITATION_PATH}")
print("> Final schema:")
df_final.printSchema()

# update state for B1
print("\n[A.5] Saving exploitation info for B1")
set_last_run(state, "exploitation", {
    "timestamp": datetime.now().isoformat(),
    "years_updated": updated_years,
    "rows_written": rows_written
})

# clear formatting data (consumed)
clear_last_run(state, "formatting")

save_state(state, BASE_DIR)
print(f"> State saved to pipeline_state.yaml")

print("\n> A5 completed")
spark.stop()