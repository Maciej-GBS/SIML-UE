import yfinance as yf
import re

START = '2010-07-18'

def fetch_ticker(symbol: str):
    ticker = yf.Ticker(symbol)

    if 'shortName' in ticker.info:
        print(ticker.info['shortName'])
    else:
        print(symbol)

    name = re.search(r'[A-Za-z0-9]+', ticker.info['symbol'])
    if name and name.group(0):
        df = ticker.history(start=START, period='max')
        df.to_csv(f'data/{name.group(0)}.csv')
    else:
        raise Exception(f"bad ticker name: {ticker.info}")

if __name__ != '__main__':
    print('not a library')
    exit(1)

symbols = ['^IXIC', 'GC=F', 'CL=F', '^GSPC', 'BLK', 'TNX', 'GXC', 'MCHI', '0386.HK', '0857.HK', 'HKD=X', 'CNY=X']
for s in symbols:
    fetch_ticker(s)
