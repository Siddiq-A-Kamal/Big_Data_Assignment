from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("us_weather_ml_model") \
    .config("spark.hadoop.fs.gs.outputstream.upload.chunk.size", "8388608") \
    .config("spark.hadoop.fs.gs.outputstream.upload.buffer.size", "8388608") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

# Load dataset from GCS
file_path = "gs://us_weather_dataset/Weather_Data_(US).csv"
df = spark.read.option("header", "true").csv(file_path)

# Convert necessary columns to float
df = df.withColumn("TMAX", df["TMAX"].cast("float")) \
       .withColumn("TMIN", df["TMIN"].cast("float")) \
       .withColumn("Elevation", F.when(df["Elevation"] == -999.9, None).otherwise(df["Elevation"].cast("float")))

# Drop missing values
df = df.dropna()

# Feature Engineering
assembler = VectorAssembler(inputCols=["Elevation"], outputCol="features")
df = assembler.transform(df)

# Split data into training and testing sets (80% train, 20% test)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train a Linear Regression Model to predict TMAX
lr = LinearRegression(featuresCol="features", labelCol="TMAX")
model = lr.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Convert to Pandas for Visualization
pdf = predictions.select("Elevation", "TMAX", "prediction").toPandas()

# Visualization
plt.figure(figsize=(8,6))
plt.hist(pdf["Elevation"], pdf["TMAX"], color='blue', label="Actual TMAX")
plt.hist(pdf["Elevation"], pdf["prediction"], color='blue', label="Predicted TMAX")
plt.xlabel("Elevation (m)")
plt.ylabel("Temperature Max (Â°C)")
plt.legend()
plt.title("Predicted vs Actual TMAX Based on Elevation")
plt.show()

# Save model output
df_result = predictions.select("Elevation", "TMAX", "prediction").coalesce(1)
df_result.write.mode("overwrite").option("header", "true").csv("gs://us_weather_dataset/output")

# Stop Spark session
spark.stop()