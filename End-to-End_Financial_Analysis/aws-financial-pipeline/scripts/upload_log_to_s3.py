import boto3
import json
import requests
from fin_config import AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID
from datetime import datetime

# Replace with your actual credentials
REGION = 'us-east-1'
S3_BUCKET = 'your-financial-data-lake'
FMP_API_KEY = '36kXMg8KWlvPl3TyhEXxnqEyda2yMa1C'

# Local file path and S3 key
local_file = r'C:\Users\Starboy\aws-financial-pipeline\config\ETL_Logs_Data.csv'
s3_key = 'raw/ETL_Logs_Data.csv'

def fetch_fmp_data(symbol='AAPL'):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=5&apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    print(f"📡 Status Code: {response.status_code}")
    return data

fetch_data = fetch_fmp_data()

# Upload
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)
s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"raw/fmp_income_statement_{datetime.now().strftime('%Y-%m-%d')}.json",
        Body=json.dumps(fetch_data),
        ContentType='application/json')

try:
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    print(f"✅ File uploaded to s3://{S3_BUCKET}/{s3_key}")
except Exception as e:
    print(f"❌ Upload failed: {e}")