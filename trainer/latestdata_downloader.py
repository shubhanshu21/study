import yfinance as yf
from datetime import datetime
import pandas as pd

def getLatestData(symbol):
    start_date = "1950-01-01"
    # end_date = "2025-05-25"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    data = yf.download(symbol+'.NS', start=start_date, end=end_date, interval="1d")

    # 1. Reset index to move 'Date' from index to column
    data.reset_index(inplace=True)

    # 2. Save to CSV
    data.to_csv(f"./datasets/{symbol}.csv", index=False)

    # Load the saved CSV to clean it
    df = pd.read_csv(f"./datasets/{symbol}.csv")

    # Remove the first row that contains the ticker symbol repetition
    df = df.drop(index=0)

    # Save the cleaned DataFrame back to CSV
    df.to_csv(f"./datasets/{symbol}.csv", index=False)

    print("✅ Data saved to:", f"datasets/{symbol}.csv")


SYMBOLS = [
    'ADANIENT', 'ADANIPORTS', 'AMBUJACEM', 'APOLLOHOSP', 'ASIANPAINT', 'AUROPHARMA', 
    'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BANDHANBNK', 'BANKBARODA', 
    'BEL', 'BERGEPAINT', 'BHARTIARTL', 'BIOCON', 'BOSCHLTD', 'BPCL', 'BRITANNIA', 
    'CHOLAFIN', 'CIPLA', 'COALINDIA', 'COLPAL', 'DABUR', 'DIVISLAB', 'DLF', 'DRREDDY', 
    'EICHERMOT', 'GAIL', 'GODREJCP', 'GRASIM', 'HAVELLS', 'HCLTECH', 'HDFCBANK', 
    'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'ICICIBANK', 'ICICIPRULI', 
    'IDEA', 'IDFCFIRSTB', 'INDIGO', 'INDUSINDBK', 'INFY', 'IOC', 'ITC', 'JINDALSTEL', 
    'JSWSTEEL', 'KOTAKBANK', 'LT', 'LTTS', 'M&M', 'MARICO', 'MARUTI', 
    'MOTHERSON', 'MPHASIS', 'MRF', 'NESTLEIND', 'NMDC', 'NTPC', 'ONGC', 'PAGEIND', 'PETRONET', 
    'PIDILITIND', 'PIIND', 'PNB', 'POWERGRID', 'RELIANCE', 'SAIL', 'SBILIFE', 'SBIN', 
    'SHREECEM', 'SIEMENS', 'SRF', 'SUNPHARMA', 'TATACHEM', 'TATACONSUM', 'TATAMOTORS', 
    'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TORNTPHARM', 'TORNTPOWER', 'TRENT', 
    'TVSMOTOR', 'UBL', 'ULTRACEMCO', 'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'ZEEL'
]

for symbol in SYMBOLS:
    print(f"Processing {symbol}...")
    getLatestData(symbol)