import os
import sys
import requests
import json
import boto3
from runpy    import run_path
from datetime import datetime
import itertools

# Ensure we can import from settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

finConfig = run_path('/home/codezeus/project_airflow/aws-financial-pipeline/config/fin_config.py')

# Variables
SYMBOL                = finConfig['LISTOFCOMP']
FMP_API_KEY           = finConfig['FMP_API_KEY']
AWS_ACCESS_KEY_ID     = finConfig['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = finConfig['AWS_SECRET_ACCESS_KEY']
REGION                = finConfig['REGION']
S3_BUCKET             = finConfig['S3_BUCKET']
ALLCOMP               = []

def fetch_fmp_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=5&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return response.json()

def upload_to_s3(data, key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION
    )
    
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data),
        ContentType='application/json'
    )
    print(f"Uploaded to s3://{S3_BUCKET}/{key}")

def main():
    for symID in SYMBOL:
        data = fetch_fmp_data(symID)
        ALLCOMP.append(data)
    # for
    main_data = list(itertools.chain.from_iterable(ALLCOMP))
    
    key = f"raw/fmp_income_statement_{datetime.now().strftime('%Y-%m-%d')}.json"
    upload_to_s3(main_data, key)

if __name__ == '__main__':
    main()