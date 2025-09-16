module_name = 'Wrangler'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 09/04/2025
Date Last Updated: 09/09/2025

Doc:
    <***>

Notes:
    <***>

ToDo:

'''

# CUSTOM IMPORTS


# OTHER IMPORTS
import pandas              as pd
from numpy                 import divide, random, round
import random
from datetime              import datetime, timedelta
import boto3
import io
import sys
from awsglue.transforms    import *
from awsglue.utils         import getResolvedOptions
from pyspark               import SparkContext
from awsglue.context       import GlueContext
from awsglue.job           import Job
from pyspark.sql           import functions as F
from pyspark.sql.types     import StringType
from pyspark.sql           import SparkSession
from numpy                 import round, random
from pyspark.sql.types     import DecimalType
from pyspark.sql.window    import Window
from pyspark.sql.functions import row_number


# USER INTERFACE
# args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
# sc          = SparkContext()
# glueContext = GlueContext(sc)
# job         = Job(glueContext)
# job.init(args['JOB_NAME'], args)
sector_name = {
    "GOOG": "Technology",
    "MSFT": "Technology",
    "AMZN": "Consumer Discretionary",
    "AAPL": "Technology",
    "META": "Communication Service",
    "NVDA": "Technology",
    "NFLX": "Entertainment",
    "BABA": "Consumer Discretionary",
    "ORCL": "Technology",
    "TSLA": "Automotive Technology"
}
s3_boto3     = boto3.client("s3")
bucket       = "your-financial-data-lake"
spark        = (SparkSession.builder.master("local[1]")
                .appName("Read from Parquet")
                .config("spark.files.overwrite", "true")
                .config("spark.jars.packages","org.apache.hadoop:hadoop-aws:3.4.1,com.amazonaws:aws-java-sdk-bundle:1.12.770")
                .config("spark.jars.excludes", "com.google.guava:guava")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") 
                .getOrCreate())
hadoop_conf  = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", "YOUR_KEY")
hadoop_conf.set("fs.s3a.secret.key", "YOUR_SECRET")
hadoop_conf.set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")

mapping_expr = F.create_map([F.lit(x) for x in sum(sector_name.items(), ())])
save_dates   = []


# FUNCTIONS
def upload_to_s3(df, key):
    """
    Uploads a pandas dataframe to s3 as a csv file

    Args:
        df  : DataFrame
        name: name of the file
        key : s3 path to save the file
    """
    csv_buffer = df.to_csv(index=False)
    s3_boto3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)
#

def verify_files_in_s3(bucket, prefix):
    """
    Verifies files inside the s3 bucket
    """

    response = s3_boto3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("No files found in the specified bucket and prefix.")
    # if
    return obj['Key'].split("/")[-1]
#

def random_dates(df):
    """
    Generate n random dates between start and end.
    
    Args:
        year (int): tear of the event
    
    Returns:
        list[datetime]: list of random datetime objects
    """
    # Get the year list fromt the dataframe
    year_list = df.select("Year").toPandas()["Year"].tolist()

    for id, yearid in enumerate(year_list):
        year = int(yearid)

        start = datetime(year, 1, 1)

        # Figure out if it's a leap year (366 days) or not (365 days)
        days_in_year = 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 365

        # Pick a random number of days to add
        random_days = random.randint(0, days_in_year - 1)

        # save the entries
        save_dates.append(start + timedelta(days=random_days))
    # for

    return save_dates
#

def data_wrangler():
    """
    This function does data wrangling on the main data.
    This includes data cleaning, transformation and preparation.

    Output:
    -------
    Returns pandas DataFrame
    """

    # Verify
    get_file = verify_files_in_s3(bucket, "transformation/mysql_income/")

    # Read the file (good for csv)
    # s3_data  = s3_boto3.get_object(Bucket=bucket, Key='transformation/mysql_income/main-data.xlsx')
    # contents = io.BytesIO(s3_data['Body'].read()) 
    # data     = pd.read_excel(contents)

    # Read file
    data = spark.read.parquet(f"s3a://your-financial-data-lake/transformation/mysql_income/{get_file}")

    # Dropping columns
    data = data.drop("CreatedAt")

    # Lets rename the columns
    df_renamed = (data
                  .withColumnRenamed("revenue","revenue_old")
                  .withColumnRenamed("NetIncome","NetIncome_old"))
    
    # Lets create new columns
    df_new = (df_renamed
              .withColumn("revenue", (df_renamed['revenue_old'].cast(DecimalType(12,2)) * 100000))
              .withColumn("netincome", (df_renamed['NetIncome_old'].cast(DecimalType(12,2)) * 1000000)))
    
    df_new = (df_new
              .withColumn("netprofitmargin", ((df_new['netincome']/df_new['revenue']).cast(DecimalType(12,2)) * 100))
              .withColumn("costofgoodsandservices", (df_new['revenue'] * (random.random()*0.03)).cast(DecimalType(12,2))))

    df_new = (df_new
              .withColumn("grossprofit", (df_new['revenue'] - df_new['costofgoodsandservices']).cast(DecimalType(12,2)))
              .withColumn("costandexpenses", (df_new['revenue'] * (random.random() * 0.05)).cast(DecimalType(12,2))))

    df_new = (df_new
              .withColumn("ebit", (df_new['grossprofit'] - df_new['costandexpenses']).cast(DecimalType(12,2)))
              .withColumn("interest", ((df_new['revenue'] * 0.01) * (random.random() * 0.26)).cast(DecimalType(12,2))))

    df_new = (df_new
              .withColumn("ebt", (df_new['revenue'] - 
                                  (df_new['costofgoodsandservices'] + df_new['costandexpenses'] 
                                   + df_new['interest'])).cast(DecimalType(12,2)))
              .withColumn("grossprofitmargin", (((df_new['revenue'] - df_new['costofgoodsandservices']) 
                                                 / df_new['revenue']).cast(DecimalType(12,2)) * 100)))

    df_new = (df_new
              .withColumn("prftaftertax", (df_new['ebt'] * 0.15).cast(DecimalType(12,2))))

    df_new = (df_new
              .withColumn("netproft", (df_new['ebit'] - df_new['interest'] - df_new['prftaftertax']).cast(DecimalType(10,2))))
    
    df_new = df_new.withColumn("Sector", mapping_expr[df_new["Company"]])

    # Generate random dates
    dates = random_dates(df_new)

    # Get the window
    w      = Window.orderBy(F.monotonically_increasing_id())
    df_new = df_new.withColumn("id", F.row_number().over(w))

    W_         = Window.orderBy("id")
    df_indexed = df_new.withColumn("row_idx", row_number().over(W_)-1)

    # Create a dataframe and save the ids
    list_df = spark.createDataFrame(
        list(enumerate(dates)), ["row_idx","date"]
    )

    # Join DataFrames on index
    df_new = df_indexed.join(list_df, "row_idx").drop("row_idx")
    df_new = df_new.drop("id")

    # Convert to pandas
    df_data = df_new.toPandas()

    # Save as csv file and upload to s3
    upload_to_s3(df_data, "production/financial_summary/main_data.csv")

#

# Run the function
data_wrangler()