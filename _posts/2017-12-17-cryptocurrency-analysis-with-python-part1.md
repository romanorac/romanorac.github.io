---
layout: post
title:  "Cryptocurrency Analysis with Python - MACD"
categories: cryptocurrency analysis
---

Cryptocurrencies are becoming mainstream so I've decided to spend the weekend learning about it. 
I've hacked together the 
[code]({{site.url}}/assets/notebooks/2017-12-17-cryptocurrency-analysis-with-python-part1.ipynb)
to download daily Bitcoin prices and apply a simple trading strategy to it. 

Note that there already exists tools for performing this kind of analysis, eg. 
[tradeview](https://www.tradingview.com/), but this way enables more in-depth analysis.

Check out my [next blog post]({{site.url}}/cryptocurrency/analysis/2017/12/25/cryptocurrency-analysis-with-python-part2.html), 
where I describe buy and hold strategy and follow me on [twitter](https://twitter.com/romanorac) to get latest updates.


<div align="middle">
<img src="{{site.url}}/assets/2017-12-10-visualizing_trading_strategy.gif" alt="Visualizing Trading Strategy Animation">
</div>

## Disclaimer
I am not a trader and this blog post is not a financial advice. This is purely introductory knowledge.

## Requirements

- Python 3

- [Jupyter Notebook](http://jupyter.org/)

- [Pandas Data Analysis Library](https://pandas.pydata.org/) 

- [Bokeh interactive visualization library](https://bokeh.pydata.org/en/latest/)

- [stock Statistics/Indicators Calculation Helper](https://github.com/jealous/stockstats)

## Getting cryptocurrency data

We download daily Bitcoin data in USD on Bitstamp exchange. [Other exchanges](https://www.cryptocompare.com/api/#introduction) are also supported.  


```python
from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'
```

The [cryptocompare api](https://www.cryptocompare.com/api/#introduction) returns following columns:
 - **open**, the price at which the period opened,
 - **high**, the highest price reached during the period,
 - **low**, the lowest price reached during the period,
 - **close**, the price at which the period closed,
 - **volumefrom**, the volume in the base currency that things are traded into,
 - **volumeto**, the volume in the currency that is being traded.
 
We download the data and store it to a file.


```python
import requests
from datetime import datetime


def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,\
        'datetime_interval should be one of %s' % supported_intervals

    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)

    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 2000, 'aggregate': 1,
              'e': exchange}
    request = requests.get(url, params=params)
    data = request.json()
    return data


def convert_to_dataframe(data):
    df = pd.io.json.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df


def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df


data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
df = convert_to_dataframe(data)
df = filter_empty_datapoints(df)

current_datetime = datetime.now().date().isoformat()
filename = get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime)
print('Saving data to %s' % filename)
df.to_csv(filename, index=False)
```

    Downloading day trading data for BTC USD from Bitstamp
    Filtering 877 empty datapoints
    Saving data to BTC_USD_Bitstamp_day_2017-12-25.csv


## Read the data

We read the data from a file so we don't need to download it again.


```python
import pandas as pd

def read_dataset(filename):
    print('Reading data from %s' % filename)
    df = pd.read_csv(filename)
    df.datetime = pd.to_datetime(df.datetime) # change type from object to datetime
    df = df.set_index('datetime') 
    df = df.sort_index() # sort by datetime
    print(df.shape)
    return df

df = read_dataset(filename)
```

    Reading data from BTC_USD_Bitstamp_day_2017-12-25.csv
    (1124, 6)


##  Trading strategy

A trading strategy is a set of objective rules defining the conditions that must be met for a trade entry and exit to occur. 

We are going to apply Moving Average Convergence Divergence (MACD) trading strategy, which is a popular indicator used in technical analysis. 
MACD calculates two moving averages of varying lengths to identify trend direction and duration.
Then, it takes the difference in values between those two moving averages (MACD line) 
and an exponential moving average (signal line) of those moving averages.
Tradeview has a great blog post about [MACD](https://www.tradingview.com/wiki/MACD_).

As we can see in the example below:
- exit trade (sell) when MACD line crosses below the MACD signal line,
- enter trade (buy) when MACD line crosses above the MACD signal line. 

<div align="middle">
<img src="http://www.onlinetradingconcepts.com/images/technicalanalysis/MACDbuysellaltNQ.gif" alt="MACD">
</div>

## Calculate the trading strategy
We use [stockstats](https://github.com/jealous/stockstats) package to calculate MACD.


```python
from stockstats import StockDataFrame
df = StockDataFrame.retype(df)
df['macd'] = df.get('macd') # calculate MACD
```

stockstats adds 5 columns to dataset:
- **close_12_ema** is fast 12 days exponential moving average,
- **close_26_ema** is slow 26 days exponential moving average,
- **macd** is MACD line,
- **macds** is signal line,
- **macdh** is MACD histogram.


```python
df.head()
```




<div style="overflow-x:scroll;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low</th>
      <th>high</th>
      <th>open</th>
      <th>close</th>
      <th>volumefrom</th>
      <th>volumeto</th>
      <th>close_12_ema</th>
      <th>close_26_ema</th>
      <th>macd</th>
      <th>macds</th>
      <th>macdh</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-11-28</th>
      <td>360.57</td>
      <td>381.34</td>
      <td>363.59</td>
      <td>376.28</td>
      <td>8617.15</td>
      <td>3220878.18</td>
      <td>376.280000</td>
      <td>376.280000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2014-11-29</th>
      <td>372.25</td>
      <td>386.60</td>
      <td>376.42</td>
      <td>376.72</td>
      <td>7245.19</td>
      <td>2746157.05</td>
      <td>376.518333</td>
      <td>376.508462</td>
      <td>0.009872</td>
      <td>0.005484</td>
      <td>0.008775</td>
    </tr>
    <tr>
      <th>2014-11-30</th>
      <td>373.32</td>
      <td>381.99</td>
      <td>376.57</td>
      <td>373.34</td>
      <td>3046.33</td>
      <td>1145566.61</td>
      <td>375.277829</td>
      <td>375.370064</td>
      <td>-0.092235</td>
      <td>-0.034565</td>
      <td>-0.115341</td>
    </tr>
    <tr>
      <th>2014-12-01</th>
      <td>373.03</td>
      <td>382.31</td>
      <td>376.40</td>
      <td>378.39</td>
      <td>6660.56</td>
      <td>2520662.37</td>
      <td>376.260220</td>
      <td>376.214306</td>
      <td>0.045914</td>
      <td>-0.007302</td>
      <td>0.106432</td>
    </tr>
    <tr>
      <th>2014-12-02</th>
      <td>375.23</td>
      <td>382.86</td>
      <td>378.39</td>
      <td>379.25</td>
      <td>6832.53</td>
      <td>2593576.46</td>
      <td>377.072532</td>
      <td>376.918296</td>
      <td>0.154236</td>
      <td>0.040752</td>
      <td>0.226969</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing trading strategy 

We use bokeh interactive charts to plot the data.

The line graph shows daily closing prices with candlesticks (zoom in).
A candlestick displays the high, low, opening  and closing prices 
for a specific period. Tradeview has a great blogpost about 
[candlestick graph](https://www.investopedia.com/terms/c/candlestick.asp).

Below the line graph we plot the MACD strategy with MACD line (blue), signal line (orange) and histogram (purple).


```python
from math import pi

from bokeh.plotting import figure, show, output_notebook, output_file
output_notebook()

datetime_from = '2016-01-01 00:00'
datetime_to = '2017-12-10 00:00'


def get_candlestick_width(datetime_interval):
    if datetime_interval == 'minute':
        return 30 * 60 * 1000  # half minute in ms
    elif datetime_interval == 'hour':
        return 0.5 * 60 * 60 * 1000  # half hour in ms
    elif datetime_interval == 'day':
        return 12 * 60 * 60 * 1000  # half day in ms


df_limit = df[datetime_from: datetime_to].copy()
inc = df_limit.close > df_limit.open
dec = df_limit.open > df_limit.close

title = '%s datapoints from %s to %s for %s and %s from %s with MACD strategy' % (
    datetime_interval, datetime_from, datetime_to, from_symbol, to_symbol, exchange)
p = figure(x_axis_type="datetime",  plot_width=1000, title=title)

p.line(df_limit.index, df_limit.close, color='black')

# plot macd strategy
p.line(df_limit.index, 0, color='black')
p.line(df_limit.index, df_limit.macd, color='blue')
p.line(df_limit.index, df_limit.macds, color='orange')
p.vbar(x=df_limit.index, bottom=[
       0 for _ in df_limit.index], top=df_limit.macdh, width=4, color="purple")

# plot candlesticks
candlestick_width = get_candlestick_width(datetime_interval)
p.segment(df_limit.index, df_limit.high,
          df_limit.index, df_limit.low, color="black")
p.vbar(df_limit.index[inc], candlestick_width, df_limit.open[inc],
       df_limit.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df_limit.index[dec], candlestick_width, df_limit.open[dec],
       df_limit.close[dec], fill_color="#F2583E", line_color="black")

output_file("visualizing_trading_strategy.html", title="visualizing trading strategy")
show(p)
```

{% include 2017-12-10-visualizing_trading_strategy.html %}
