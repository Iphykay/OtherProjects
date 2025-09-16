# OTHER IMPORTS
import sagemaker
import boto3
import pandas as pd
import io
from sklearn.model_selection import train_test_split

# USER INTERFAE
sector_cat = {
    'Entertainment': 0,
    'Automotive Technology': 1,
    'Technology':2,
    'Consumer Discretionary': 3,
    'Communication Service':4
}

train_sk_prefix = "ML components/training-data"
test_sk_prefix  = "ML components/testing-data"


# FUNCTIONS
def verify_files_in_s3(bucket, prefix):
    response = s3_boto3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("No files found in the specified bucket and prefix.")
    # if
#

def get_sector_category(sector):
    return sector_cat[sector]
#

def get_YoY_revenue_change_df(company_names):

    # Get the current year
    years = sum_revenue['year'].unique()
    for id_yr in range(len(years)-1):
        # get the current year
        current_revenue = sum_revenue.loc[(sum_revenue['company'] == company_names) & 
                                          (sum_revenue['year'] == years[id_yr]), 'total_revenue'].values[0]
        prev_revenue    = sum_revenue.loc[(sum_revenue['company'] == company_names) & 
                                          (sum_revenue['year'] == years[id_yr] - 1), 'total_revenue'].values[0]

        revenue_change = ((current_revenue - prev_revenue) / prev_revenue) * 100
                
        if id_yr == 0:
            sum_revenue.loc[(sum_revenue['company'] == company_names) & 
                            (sum_revenue['year'] == years[id_yr]), 'YoY Revenue Change %'] = 0.0
            sum_revenue.loc[(sum_revenue['company'] == company_names) & 
                            (sum_revenue['year'] == years[id_yr]-1), 'YoY Revenue Change %'] = revenue_change
        else:
            sum_revenue.loc[(sum_revenue['company'] == company_names) & 
                            (sum_revenue['year'] == years[id_yr]-1), 'YoY Revenue Change %'] = revenue_change
        # if
    # for

    return sum_revenue
#

def upload_data_to_s3(data, bucket):
    for dataType in list(data.keys()):
        if dataType.endswith('train'):
            train_path = sgm_sess.upload_data('financial_train_data.csv', 
                                              bucket=bucket, 
                                              key_prefix=train_sk_prefix)
        else:
            test_path  =  sgm_sess.upload_data('financial_test_data.csv', 
                                               bucket=bucket, 
                                               key_prefix=test_sk_prefix)
        # if
    # for
    print("Data uploaded to S3 successfully.")

    return train_path, test_path
#

# Initialize clients
sm_boto3   = boto3.client("sagemaker")
s3_boto3   = boto3.client("s3")
sgm_sess   = sagemaker.Session()
get_region = sgm_sess.boto_session.region_name
bucket     = "your-financial-data-lake"

# Get the data loation from the s3 bucket
s3_data_path = "s3://your-financial-data-lak/production/financial_summary/main-data.xlsx"
ml_path      = "s3://your-financial-data-lake/ML components/"

# Verify
verify_files_in_s3(bucket, "production/financial_summary/")

# Read the data using pandas
s3_data  = s3_boto3.get_object(Bucket=bucket, Key='production/financial_summary/main-data.xlsx')
contents = io.BytesIO(s3_data['Body'].read()) 
data_df  = pd.read_excel(contents)

# Remove the unnecessary columns
data_df  = data_df.drop(columns=['revenue_old', 'netincome_old'])

# Getting unique values
sector_names  = data_df.Sector.unique()
company_names = data_df.company.unique()
years         = data_df.year.unique()
sector_names  = data_df.Sector.unique()

data_df['Sector'] = data_df['Sector'].apply(lambda x: get_sector_category(x))

sum_revenue = data_df.groupby(by=['company', 'year'])['revenue'].sum().reset_index()
sum_revenue = sum_revenue.rename(columns={'revenue': 'total_revenue'})
sum_revenue = sum_revenue.sort_values(by=['company','year'], ascending=False).reset_index().drop(columns=['index'])

# Get the YoY revenue change for each company
for id_cmp in sum_revenue['company'].unique():
    print("Processing company:", id_cmp)

    see_result = get_YoY_revenue_change_df(id_cmp)
# for

# Sector Prediction
data_df = data_df.drop(columns=['company', 'date'])

# get columns
columns = list(data_df.columns)

y_data = data_df["Sector"]

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_df, test_size=0.2, 
                                         random_state=42, 
                                         stratify=y_data)

# See the shape of the test data
print("Shape of Xtest:", train_data.shape)
print("Shape of ytest:", test_data.shape)

# Save the data into a csv file
train_data.to_csv('financial_train_data.csv', index=False)
test_data.to_csv('financial_test_data.csv', index=False)

Alldata = {'train': train_data, 'test': test_data}

train_path, test_path = upload_data_to_s3(Alldata, bucket)

# The MLsmodel_script goes here
