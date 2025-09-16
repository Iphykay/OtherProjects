module_name = 'Wrangler'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 09/04/2025
Date Last Updated: 09/04/2025

Doc:
    <***>

Notes:
    <***>

ToDo:

'''


# CUSTOM IMPORTS


# OTHER IMPORTS
import pandas as pd
from numpy    import divide, random, round
import random
from datetime import datetime, timedelta
import boto3
import io
import sys
from awsglue.transforms    import *
from awsglue.utils         import getResolvedOptions
from pyspark               import SparkContext
from awsglue.context       import GlueContext
from awsglue.job           import Job


# USER INTERFACE
args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)
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
    "TSLA": "Automotive Technology"}

s3_boto3   = boto3.client("s3")
bucket     = "your-financial-data-lake"


# FUNCTIONS
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
#

def random_dates(year):
    """
    Generate n random dates between start and end.
    
    Args:
        year (int): tear of the event
    
    Returns:
        list[datetime]: list of random datetime objects
    """
    start = datetime(year, 1, 1)

    # Figure out if it's a leap year (366 days) or not (365 days)
    days_in_year = 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 365

    # Pick a random number of days to add
    random_days = random.randint(0, days_in_year - 1)

    return start + timedelta(days=random_days)
#

def get_company_category(company):
    return sector_name[company]
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

    # Not yet
    data = spark.read.parquet(f"s3://your-financial-data-lake/transformation/mysql_income/{get_file}")
    
    # Drop the data columns
    data = data.drop(['Date', 'Time'], axis=1)

    # Lets rename the columns
    data.rename(columns={'revenue':'revenue_old', 'NetIncome':'NetIncome_old'}, 
                inplace=True)
    
    # Lets create new columns
    data['revenue']                = round(data['revenue_old'] * 100000,2)
    data['netincome']              = round(data['NetIncome_old'] * 1000000,2)
    data['netprofitmargin']        = round(divide(data['netincome'],data['revenue']) * 100,2)
    data['costofgoodsandservices'] = round(data['revenue'] * (random.random() * 0.03),2)
    data['grossprofit']            = round(data['revenue'] - data['costofgoodsandservices'],2)
    data['costandexpenses']        = round(data['revenue'] * (random.random() * 0.05),2)
    data['ebit']                   = round(data['grossprofit'] - data['costandexpenses'],2)
    data['interest']               = round((data['revenue'] * 0.01) * (random.random() * 0.26),2)
    data['ebt']                    = round(data['revenue'] - (data['costofgoodsandservices'] + data['costandexpenses'] + data['interest']),2)
    data['proftaftertax']          = round(data['ebt'] * 0.15,2)
    data['netproft']               = round(data['ebit'] - data['interest'] - data['proftaftertax'],2)
    data['grossprofitmargin']      = round(divide((data['revenue']-data['costofgoodsandservices']),data['revenue']) * 100,2)
    data['Sector']                 = data['company'].apply(lambda x: get_company_category(x))
    data['date']                   = data['year'].apply(lambda x: random_dates(x))

    return data
#