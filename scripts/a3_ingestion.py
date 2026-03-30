import os
import sys
import glob
import yaml
import json
import unicodedata
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col

BASE_DIR = os.getcwd()

with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(BASE_DIR, config['paths']['scripts_dir']))
from pipeline_state_manager import load_state, save_state, set_last_run

SOURCE_DIR = os.path.join(BASE_DIR, config['paths']['source_dir'])
LANDING_DIR = os.path.join(BASE_DIR, config['paths']['landing_dir'])

# spark session
spark = SparkSession.builder \
    .appName("A3_Ingestion") \
    .master("local[*]") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# dataset configs - loaded from config.yaml
DATASETS_CONFIG = config['datasets']

# tracks partitions updated in this run (for A4)
updated_partitions = {}

def normalize_filename(filename):
    normalized = unicodedata.normalize('NFD', filename)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

def get_files_to_process(source_path, dataset_name, extension, history, specific_file=None):
    if specific_file:
        file_path = os.path.join(source_path, specific_file)
        all_files_paths = [file_path] if os.path.exists(file_path) else []
    else:
        all_files_paths = glob.glob(os.path.join(source_path, f"*.{extension}"))
    
    dataset_history = history.get(dataset_name, {})
    files_to_process = []
    file_metadata = {}

    for file_path in all_files_paths:
        filename = os.path.basename(file_path)
        normalized_name = normalize_filename(filename)
        current_mtime = os.path.getmtime(file_path)
        stored_mtime = dataset_history.get(normalized_name)
        
        if stored_mtime is None or current_mtime > stored_mtime:
            files_to_process.append(file_path)
            file_metadata[normalized_name] = current_mtime
            
    return files_to_process, file_metadata


def parse_concatenated_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        return []

    if content.startswith("[") and content.endswith("]"):
        return json.loads(content)
    
    decoder = json.JSONDecoder()
    data_list = []
    idx = 0
    
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        obj, end_idx = decoder.raw_decode(content, idx=idx)
        data_list.append(obj)
        idx = end_idx
    
    if not data_list and content:
        fixed = "[" + content.replace("}\n{", "},\n{").replace("}{", "},{") + "]"
        return json.loads(fixed)

    return data_list


def ingest_dataset(cfg, state):
    dataset_name = cfg["name"]
    source_sub = cfg["subfolder"]
    file_fmt = cfg["format"]
    opts = cfg.get("options", {})
    date_type = cfg.get("date_type")
    regex_pattern = cfg.get("regex")
    specific_file = cfg.get("specific_file")

    input_dir = os.path.join(SOURCE_DIR, source_sub)
    output_path = os.path.join(LANDING_DIR, dataset_name)
    
    history = state.get("ingestion_history", {})

    print(f"\n[A.3] Processing: {dataset_name} ({date_type or 'snapshot'})")

    # incremental strategy (daily/yearly)
    if date_type in ["daily", "yearly"] and regex_pattern:
        files_paths, new_metadata = get_files_to_process(input_dir, dataset_name, file_fmt, history, specific_file)
        
        if not files_paths:
            print("> No new files, skipping")
            return

        print(f"> Found {len(files_paths)} file(s) to process")
        
        reader = spark.read.format(file_fmt)
        for k, v in opts.items():
            reader = reader.option(k, v)
        
        df = reader.load(files_paths)
        df = df.withColumn("origin_file", input_file_name())

        if date_type == "daily":
            df = df.withColumn("year", regexp_extract(col("origin_file"), regex_pattern, 1).cast("integer")) \
                   .withColumn("month", regexp_extract(col("origin_file"), regex_pattern, 2).cast("integer")) \
                   .withColumn("day", regexp_extract(col("origin_file"), regex_pattern, 3).cast("integer"))
            partition_cols = ["year", "month", "day"]
        else:  # yearly
            df = df.withColumn("year", regexp_extract(col("origin_file"), regex_pattern, 1).cast("integer"))
            partition_cols = ["year"]

        df = df.drop("origin_file")

        print(f"> Writing partitioned by {partition_cols}")
        df.write.format(file_fmt).options(**opts).partitionBy(*partition_cols).mode("overwrite").save(output_path)
        
        # track which partitions were updated
        partitions = df.select(partition_cols).distinct().collect()
        updated_partitions[dataset_name] = [row.asDict() for row in partitions]
        
        # update ingestion history (persistent)
        if dataset_name not in state["ingestion_history"]:
            state["ingestion_history"][dataset_name] = {}
        state["ingestion_history"][dataset_name].update(new_metadata)
        print("> History updated")

    # yearly from internal column (density)
    elif date_type == "yearly_from_column":
        year_column = cfg.get("year_column", "year")
        
        if specific_file:
            source_path = os.path.join(input_dir, specific_file)
            filename = specific_file
        else:
            source_path = input_dir
            filename = dataset_name
        
        normalized_name = normalize_filename(filename)
        current_mtime = os.path.getmtime(source_path)
        dataset_history = history.get(dataset_name, {})
        stored_mtime = dataset_history.get(normalized_name)
        
        if stored_mtime is not None and current_mtime <= stored_mtime:
            print("> No changes, skipping")
            return
        
        print(f"> Reading from {source_path}")
        
        reader = spark.read.format(file_fmt)
        for k, v in opts.items():
            reader = reader.option(k, v)
        
        df = reader.load(source_path)
        df = df.withColumn("year", col(year_column).cast("integer"))
        
        print("> Writing partitioned by year")
        df.write.format(file_fmt).partitionBy("year").mode("overwrite").save(output_path)
        
        partitions = df.select("year").distinct().collect()
        updated_partitions[dataset_name] = [row.asDict() for row in partitions]
        
        if dataset_name not in state["ingestion_history"]:
            state["ingestion_history"][dataset_name] = {}
        state["ingestion_history"][dataset_name][normalized_name] = current_mtime
        print("> History updated")

    # snapshot strategy (lookup)
    else:
        print("> Full overwrite, static dataset")
        
        source_path = os.path.join(input_dir, specific_file) if specific_file else input_dir
        
        reader = spark.read.format(file_fmt)
        for k, v in opts.items():
            reader = reader.option(k, v)
        
        df = reader.load(source_path)
        df.coalesce(1).write.format(file_fmt).options(**opts).mode("overwrite").save(output_path)
        print("> Done")


# main
print("\n[A.3] Ingestion started")

state = load_state(BASE_DIR)

for ds_config in DATASETS_CONFIG:
    ingest_dataset(ds_config, state)

# save updated partitions for A4 (in last_run)
from datetime import datetime
set_last_run(state, "ingestion", {
    "timestamp": datetime.now().isoformat(),
    "partitions_updated": updated_partitions
})

save_state(state, BASE_DIR)
print(f"\n> State saved to pipeline_state.yaml")

print("\n> A3 completed")
spark.stop()