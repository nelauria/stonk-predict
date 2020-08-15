import pandas as pd
from datetime import datetime as dt
from datetime import timezone
import requests
from io import StringIO


# function to get stonk data
def get_stonk(stonk_code, hist_date='01-01-2010'):
    today = dt.today()

    # converting dates to Unix timestamps
    try:
        unix_hist = dt.strptime(hist_date, '%m-%d-%Y').replace(tzinfo=timezone.utc).timestamp()
    except:
        raise Exception('The date inputted could not be interpreted. The date should be in MM-DD-YYYY format.')
    unix_today = today.replace(tzinfo=timezone.utc).timestamp()

    # getting stonk history from Yahoo! Finance
    link = (f'https://query1.finance.yahoo.com/v7/finance/download/'
            f'{stonk_code}?period1={int(unix_hist)}&period2={int(unix_today)}&interval=1d&events=history')
    try:
        b = requests.get(link)
    except:
        raise Exception('The stock code entered could not be found on Yahoo! Finance.')
    s = str(b.content, 'utf-8')
    s_io = StringIO(s)
    stonks = pd.read_csv(s_io, encoding='utf-8', parse_dates=['Date'])

    # converting Date column to numeric data
    stonks['Date'] = stonks['Date'].map(dt.toordinal)

    return stonks

