# File: glue/transform_financial_data.py
# Purpose: Normalize raw FMP and MySQL CSV data, output as typed Parquet to /transformation/

import sys
from awsglue.transforms    import *
from awsglue.utils         import getResolvedOptions
from pyspark               import SparkContext
from awsglue.context       import GlueContext
from awsglue.job           import Job
from pyspark.sql.functions import col, to_date, upper, year

args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ---- Load FMP JSON Income Statement ----
fmp_df = spark.read.json("s3://your-financial-data-lake/raw/fmp_income_statement_2025-05-29.json")
fmp_df = fmp_df.withColumnRenamed("Year", "calendar_year") \
             .withColumn("symbol", upper(col("symbol"))) \
             .withColumn("date", to_date("date"))

fmp_df.write.mode("overwrite").parquet("s3://your-financial-data-lake/transformation/cleaned_fmp_income/")

# ---- Load MySQL Income Statement CSV ----
mysql_income_df = spark.read.option("header", True).csv("s3://your-financial-data-lake/raw/mysql_income_statements_data_2025-05-27.csv")

mysql_income_df.write.mode("overwrite").parquet("s3://your-financial-data-lake/transformation/mysql_income/")

# ---- Load Stock Prices ----
stock_prices_df = spark.read.option("header", True).csv("s3://your-financial-data-lake/raw/mysql_stock_prices_data_2025-05-27.csv")

# Drop the year column since its not complete
stock_prices_df = stock_prices_df.drop('Year')

# Add a new year column
stockPrices_df = stock_prices_df.join(mysql_income_df.select("Year"),how="outer")

# Save the file
stockPrices_df.write.mode("overwrite").parquet("s3://your-financial-data-lake/transformation/cleaned_stock_prices/")

job.commit()

