from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("us_weather_dataset") \
    .getOrCreate()

# Path to the dataset in GCS
file_path = "gs://us_weather_dataset/Weather_Data_(US).csv"

# Load the dataset into a DataFrame
df = spark.read.option("header", "true").csv(file_path)

# Data Preprocessing: Convert necessary columns to float
columns_to_cast = ["TMAX", "TMIN", "EVAP", "PRCP", "Latitude", "Longitude", "Elevation"]
for col in columns_to_cast:
    df = df.withColumn(col, df[col].cast("float"))

# Handle missing Elevation values (-999.9)
df = df.withColumn("Elevation", F.when(df["Elevation"] == -999.9, None).otherwise(df["Elevation"]))

# Calculate overall averages across the entire dataset
avg_weather = df.agg(
    F.avg("TMAX").alias("Avg_Max_Temp"),
    F.avg("TMIN").alias("Avg_Min_Temp")
)

# Output path in GCS
output_path = "gs://us_weather_dataset/output"

# Writing the result to a single CSV file in GCS
avg_weather.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

# Stop the Spark session
spark.stop()