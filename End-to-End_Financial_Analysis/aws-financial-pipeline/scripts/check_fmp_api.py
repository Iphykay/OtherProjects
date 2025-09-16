import requests
import json

# Replace with your actual API key
FMP_API_KEY = '36kXMg8KWlvPl3TyhEXxnqEyda2yMa1C'

def fetch_fmp_data(symbol='AAPL'):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=5&apikey={FMP_API_KEY}"
    response = requests.get(url)
    print(f"📡 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"Received {len(data)} records for {symbol}")
                print(json.dumps(data[0], indent=2))  # Print first record as sample
            else:
                print("Response received, but data list is empty.")
        except json.JSONDecodeError:
            print("Failed to decode JSON. Raw response:")
            print(response.text)
    else:
        print("Failed to fetch data from API.")
        print("Response:", response.text)
    # if
#

if __name__ == '__main__':
    fetch_fmp_data()

