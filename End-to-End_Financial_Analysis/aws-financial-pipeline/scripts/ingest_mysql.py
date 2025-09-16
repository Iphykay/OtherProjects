import pandas as pd
import boto3
import pyodbc
from datetime import datetime
from runpy    import run_path

finConfig = run_path('/home/codezeus/project_airflow/aws-financial-pipeline/config/fin_config.py')

# Variables
S3_BUCKET         = finConfig['S3_BUCKET']
connection_string = finConfig['connection_string']
data              = ['Income_Statements_Data','Stock_Prices_Data','ETL_Logs_Data']

def get_data(name_data):
    conn = pyodbc.connect(connection_string)
    df   = pd.read_sql(f"SELECT * from {name_data}", conn)
    conn.close()
    return df

def upload_to_s3(df, key):
    csv_buffer = df.to_csv(index=False)
    s3         = boto3.client('s3')
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_buffer)

def main():
    for dataID in data:
        df  = get_data(dataID)
        key = f"raw/mysql_{dataID.lower()}_{datetime.now().strftime('%Y-%m-%d')}.csv"
        upload_to_s3(df, key)

if __name__ == '__main__':
    main()