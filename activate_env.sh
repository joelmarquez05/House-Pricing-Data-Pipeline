# Set BDA_PROJECT_PATH to the absolute directory where this script is located
export BDA_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Java & Hadoop Home (required for Spark)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export HADOOP_HOME=$BDA_PROJECT_PATH/hadoop_home
export PATH=$PATH:$HADOOP_HOME/bin
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH

# PySpark configuration
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
export SPARK_LOCAL_IP=127.0.0.1

# Airflow Home
export AIRFLOW_HOME=$BDA_PROJECT_PATH/airflow

# Virtual environment
if [ -f "$BDA_PROJECT_PATH/venv/bin/activate" ]; then
    source "$BDA_PROJECT_PATH/venv/bin/activate"
    echo "Venv environment activated successfully!"
else
    echo "Warning: Virtual environment not found. Create it with: python3 -m venv venv"
fi

# MongoDB Initialization
if pgrep -x "mongod" >/dev/null; then
    echo "MongoDB is running."
else
    echo "MongoDB is not running. Attempting to start..."
    # Attempt to start via service (most common on Linux/WSL)
    sudo service mongod start
fi

# Airflow Initialization
if pgrep -f "airflow" >/dev/null; then
    echo "Airflow is already running."
else
    echo "Initializing Airflow..."
    airflow db migrate
    echo "Starting Airflow scheduler in background..."
    nohup airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
    echo "Airflow scheduler started (PID: $!)"
fi

echo "Environment activated successfully!"
echo "Project path: $BDA_PROJECT_PATH"