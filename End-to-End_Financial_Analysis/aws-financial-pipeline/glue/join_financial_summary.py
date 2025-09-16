# File: glue/join_financial_summary.py
# Purpose: Join transformed income statements with stock prices and derive metrics

import sys
from awsglue.transforms    import *
from awsglue.utils         import getResolvedOptions
from pyspark               import SparkContext
from awsglue.context       import GlueContext
from awsglue.job           import Job
from pyspark.sql.functions import col, explode, expr
from pyspark.sql.types     import StructType, ArrayType
import pandas              as pd

args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Load Parquet-transformed data
fmp_income   = spark.read.parquet("s3://your-financial-data-lake/transformation/fmp_income/")
mysql_income = spark.read.parquet("s3://your-financial-data-lake/transformation/mysql_income/")
stock_prices = spark.read.parquet("s3://your-financial-data-lake/transformation/stock_prices/")

# Recursive function to flatten the DataFrame
def flatten(df):
    complex_fields = [
        (field.name, field.dataType)
        for field in df.schema.fields
        if isinstance(field.dataType, (StructType, ArrayType))]

    while complex_fields:
        col_name, col_type = complex_fields.pop(0)

        # Flatten StructType columns
        if isinstance(col_type, StructType):
            expanded = [col(f"{col_name}.{nested.name}").alias(f"{col_name}_{nested.name}") for nested in col_type.fields]
            df = df.select("*", *expanded).drop(col_name)
        # if

        # Explode ArrayType columns
        elif isinstance(col_type, ArrayType):
            df = df.withColumn(col_name, explode(col_name))

        # Recalculate complex fields
        complex_fields = [
            (field.name, field.dataType)
            for field in df.schema.fields
            if isinstance(field.dataType, (StructType, ArrayType))
        ]

    return df

# Apply flattening
fmpIncome = flatten(fmp_income)

# Join with stock prices on symbol and date
combined_income = fmpIncome.join(stock_prices.select("Company","Open","Close","Volume", "Year"),how="left")
combined_income = combined_income.withColumn("PE_ratio", expr("close / eps"))

# Save as Parquet to production zone
combined_income.write.mode("overwrite").parquet("s3://your-financial-data-lake/production/financial_summary/")

job.commit()