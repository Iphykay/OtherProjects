from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Read from Parquet").getOrCreate()