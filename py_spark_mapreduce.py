from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row

# Initialize Spark session
spark = SparkSession.builder \
    .appName("us_weather_dataset") \
    .getOrCreate()

# Path to the dataset in GCS
file_path = "gs://us_weather_dataset/Weather_Data_(US).csv"

# Load the dataset into a DataFrame
df = spark.read.option("header", "true").csv(file_path)

# Data Preprocessing: Convert necessary columns to float
columns_to_cast = ["TMAX", "TMIN"]
for col in columns_to_cast:
    df = df.withColumn(col, df[col].cast("float"))

# Convert DataFrame to RDD
rdd = df.rdd

# Map function: Key-value pairs with TMAX and TMIN
def map_function(row):
    return ("min_max", (row["TMAX"], row["TMIN"]))

mapped_rdd = rdd.map(map_function)

# Reduce function: Handles None values
def reduce_function(a, b):
    tmax_values = [x for x in [a[0], b[0]] if x is not None]
    tmin_values = [x for x in [a[1], b[1]] if x is not None]

    min_tmax = min(tmax_values) if tmax_values else None
    max_tmax = max(tmax_values) if tmax_values else None
    min_tmin = min(tmin_values) if tmin_values else None
    max_tmin = max(tmin_values) if tmin_values else None

    return (min_tmax, max_tmax, min_tmin, max_tmin)

# Apply reduceByKey
reduced_rdd = mapped_rdd.reduceByKey(reduce_function)

# Collect results
min_max_values = reduced_rdd.collect()

# Convert result to DataFrame
result_rows = []
for key, values in min_max_values:
    min_tmax, max_tmax, min_tmin, max_tmin = values
    result_rows.append(Row(Metric="Min_TMAX", Value=min_tmax))
    result_rows.append(Row(Metric="Max_TMAX", Value=max_tmax))
    result_rows.append(Row(Metric="Min_TMIN", Value=min_tmin))
    result_rows.append(Row(Metric="Max_TMIN", Value=max_tmin))

# Create a DataFrame from the results
result_df = spark.createDataFrame(result_rows)

# Save the result to a CSV file in GCS
output_path = "gs://us_weather_dataset/output_min_max"
result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Stop Spark session
spark.stop()