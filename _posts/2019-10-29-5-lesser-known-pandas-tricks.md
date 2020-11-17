---
layout: post
title:  "5 lesser-known pandas tricks"
use_math: true
categories: machine learning
---

<div style="font-size:80%; text-align:center;">
<div align="middle">
<img src="{{site.url}}/assets/images/2019-10-29-5-lesser-known-pandas-tricks/pandas-logo.png
" alt="Pandas">
</div>
Pandas provides high-performance, easy-to-use data structures and data analysis tools for the Python
</div>

pandas needs no introduction as it became the de facto tool for data analysis in Python. 
As a Data Scientist, I use pandas daily and I am always amazed by how many functionalities it has.
In this post, I am going to show you 5 pandas tricks that I learned recently and using them helps me to be more productive.

For pandas newbies - [Pandas](https://pandas.pydata.org/) provides high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.

To run the examples download this [Jupyter notebook]({{site.url}}/assets/notebooks/2019-10-29-5-lesser-known-pandas-tricks.ipynb).


```python
from platform import python_version

import pandas as pd
import xlsxwriter
```

<b>Here are a few links you might be interested in</b>:

- [Intro to Machine Learning](https://imp.i115008.net/c/2402645/788201/11298)
- [Intro to Programming](https://imp.i115008.net/c/2402645/788200/11298)
- [Data Science for Business Leaders](https://imp.i115008.net/c/2402645/880006/11298)
- [AI for Healthcare](https://imp.i115008.net/c/2402645/824078/11298)
- [Autonomous Systems](https://imp.i115008.net/c/2402645/829912/11298)
- [Learn SQL](https://imp.i115008.net/c/2402645/828338/11298)

Disclosure: Bear in mind that some of the links above are affiliate links and if you go through them to make a purchase I will earn a commission. Keep in mind that I link courses because of their quality and not because of the commission I receive from your purchases. The decision is yours, and whether or not you decide to buy something is completely up to you.

## Setup


```python
print("python version==%s" % python_version())
print("pandas==%s" % pd.__version__)
print("xlsxwriter==%s" % xlsxwriter.__version__)
```

    python version==3.7.3
    pandas==0.25.0
    xlsxwriter==1.2.1


## 1. Date Ranges

When fetching the data from an external API or a database, we many times need to specify a date range.
Pandas got us covered. 
There is a [data_range](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html) function, which returns dates incremented by days, months or years, etc.

Let's say we need a date range incremented by days. 


```python
date_from = "2019-01-01"
date_to = "2019-01-12"
date_range = pd.date_range(date_from, date_to, freq="D")
date_range
```




    DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                   '2019-01-05', '2019-01-06', '2019-01-07', '2019-01-08',
                   '2019-01-09', '2019-01-10', '2019-01-11', '2019-01-12'],
                  dtype='datetime64[ns]', freq='D')



Let's transform the generated date_range to start and end dates, which can be passed to a subsequent function.


```python
for i, (date_from, date_to) in enumerate(zip(date_range[:-1], date_range[1:]), 1):
    date_from = date_from.date().isoformat()
    date_to = date_to.date().isoformat()
    print("%d. date_from: %s, date_to: %s" % (i, date_from, date_to))
```

    1. date_from: 2019-01-01, date_to: 2019-01-02
    2. date_from: 2019-01-02, date_to: 2019-01-03
    3. date_from: 2019-01-03, date_to: 2019-01-04
    4. date_from: 2019-01-04, date_to: 2019-01-05
    5. date_from: 2019-01-05, date_to: 2019-01-06
    6. date_from: 2019-01-06, date_to: 2019-01-07
    7. date_from: 2019-01-07, date_to: 2019-01-08
    8. date_from: 2019-01-08, date_to: 2019-01-09
    9. date_from: 2019-01-09, date_to: 2019-01-10
    10. date_from: 2019-01-10, date_to: 2019-01-11
    11. date_from: 2019-01-11, date_to: 2019-01-12


## 2. Merge with indicator

Merging two datasets is the process of bringing two datasets together into one, and aligning the rows from each based on common attributes or columns.

One of the arguments of the merge function that I've missed is the `indicator` argument.
Indicator argument adds a `_merge` column to a DataFrame, which tells you "where the row came from", left, right or both DataFrames.
The `_merge` column can be very useful when working with bigger datasets to check the correctness of a merge operation.


```python
left = pd.DataFrame({"key": ["key1", "key2", "key3", "key4"], "value_l": [1, 2, 3, 4]})
left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>key1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>key2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>key4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({"key": ["key3", "key2", "key1", "key6"], "value_r": [3, 2, 1, 6]})
right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>key3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>key2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>key6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge = left.merge(right, on='key', how='left', indicator=True)
df_merge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>value_l</th>
      <th>value_r</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>key1</td>
      <td>1</td>
      <td>1.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>1</th>
      <td>key2</td>
      <td>2</td>
      <td>2.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key3</td>
      <td>3</td>
      <td>3.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>key4</td>
      <td>4</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
  </tbody>
</table>
</div>



The `_merge` column can be used to check if there is an expected number of rows with values from both DataFrames.


```python
df_merge._merge.value_counts()
```




    both          3
    left_only     1
    right_only    0
    Name: _merge, dtype: int64



## 3. Nearest merge

When working with financial data, like stocks or cryptocurrencies, we may need to combine quotes (price changes) with actual trades.
Let's say that we would like to merge each trade with a quote that occurred a few milliseconds before it.
Pandas has a function merge_asof, which enables merging DataFrames by the nearest key (timestamp in our example).
The datasets quotes and trades are taken from [pandas example](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)

The quotes DataFrame contains price changes for different stocks. 
Usually, there are many more quotes than trades.


```python
quotes = pd.DataFrame(
    [
        ["2016-05-25 13:30:00.023", "GOOG", 720.50, 720.93],
        ["2016-05-25 13:30:00.023", "MSFT", 51.95, 51.96],
        ["2016-05-25 13:30:00.030", "MSFT", 51.97, 51.98],
        ["2016-05-25 13:30:00.041", "MSFT", 51.99, 52.00],
        ["2016-05-25 13:30:00.048", "GOOG", 720.50, 720.93],
        ["2016-05-25 13:30:00.049", "AAPL", 97.99, 98.01],
        ["2016-05-25 13:30:00.072", "GOOG", 720.50, 720.88],
        ["2016-05-25 13:30:00.075", "MSFT", 52.01, 52.03],
    ],
    columns=["timestamp", "ticker", "bid", "ask"],
)
quotes['timestamp'] = pd.to_datetime(quotes['timestamp'])
quotes.shape
```




    (8, 4)




```python
quotes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>ticker</th>
      <th>bid</th>
      <th>ask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-05-25 13:30:00.023</td>
      <td>GOOG</td>
      <td>720.50</td>
      <td>720.93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-05-25 13:30:00.023</td>
      <td>MSFT</td>
      <td>51.95</td>
      <td>51.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-05-25 13:30:00.030</td>
      <td>MSFT</td>
      <td>51.97</td>
      <td>51.98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-05-25 13:30:00.041</td>
      <td>MSFT</td>
      <td>51.99</td>
      <td>52.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>GOOG</td>
      <td>720.50</td>
      <td>720.93</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-05-25 13:30:00.049</td>
      <td>AAPL</td>
      <td>97.99</td>
      <td>98.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-05-25 13:30:00.072</td>
      <td>GOOG</td>
      <td>720.50</td>
      <td>720.88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-05-25 13:30:00.075</td>
      <td>MSFT</td>
      <td>52.01</td>
      <td>52.03</td>
    </tr>
  </tbody>
</table>
</div>



The trades DataFrame contains trades of different stocks.


```python
trades = pd.DataFrame(
    [
        ["2016-05-25 13:30:00.023", "MSFT", 51.95, 75],
        ["2016-05-25 13:30:00.038", "MSFT", 51.95, 155],
        ["2016-05-25 13:30:00.048", "GOOG", 720.77, 100],
        ["2016-05-25 13:30:00.048", "GOOG", 720.92, 100],
        ["2016-05-25 13:30:00.048", "AAPL", 98.00, 100],
    ],
    columns=["timestamp", "ticker", "price", "quantity"],
)
trades['timestamp'] = pd.to_datetime(trades['timestamp'])
trades.shape
```




    (5, 4)




```python
trades.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>ticker</th>
      <th>price</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-05-25 13:30:00.023</td>
      <td>MSFT</td>
      <td>51.95</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-05-25 13:30:00.038</td>
      <td>MSFT</td>
      <td>51.95</td>
      <td>155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>GOOG</td>
      <td>720.77</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>GOOG</td>
      <td>720.92</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>AAPL</td>
      <td>98.00</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>


We merge trades and quotes by tickers, where the latest quote can be 10 ms behind the trade.
If a quote is more than 10 ms behind the trade or there isn't any quote, the bid and ask for that quote will be null (AAPL ticker in this example). 

```python
pd.merge_asof(trades, quotes, on="timestamp", by='ticker', tolerance=pd.Timedelta('10ms'), direction='backward')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>ticker</th>
      <th>price</th>
      <th>quantity</th>
      <th>bid</th>
      <th>ask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-05-25 13:30:00.023</td>
      <td>MSFT</td>
      <td>51.95</td>
      <td>75</td>
      <td>51.95</td>
      <td>51.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-05-25 13:30:00.038</td>
      <td>MSFT</td>
      <td>51.95</td>
      <td>155</td>
      <td>51.97</td>
      <td>51.98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>GOOG</td>
      <td>720.77</td>
      <td>100</td>
      <td>720.50</td>
      <td>720.93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>GOOG</td>
      <td>720.92</td>
      <td>100</td>
      <td>720.50</td>
      <td>720.93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-05-25 13:30:00.048</td>
      <td>AAPL</td>
      <td>98.00</td>
      <td>100</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Create an Excel report

Pandas (with XlsxWriter library) enables us to create an Excel report from the DataFrame. 
This is a major time saver - no more saving a DataFrame to CSV and then formatting it in Excel.
We can also add all kinds of [charts](https://pandas-xlsxwriter-charts.readthedocs.io/), etc.


```python
df = pd.DataFrame(pd.np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"])
df.shape
```




    (3, 3)



The code snippet below creates an Excel report. To save a DataFrame to the Excel file, uncomment the `writer.save()` line.


```python
report_name = 'example_report.xlsx'
sheet_name = 'Sheet1'

writer = pd.ExcelWriter(report_name, engine='xlsxwriter')
df.to_excel(writer, sheet_name=sheet_name, index=False)
# writer.save() 
```

As mentioned before, the library also supports adding charts to the Excel report.
We need to define the type of the chart (line chart in our example) and the data series for the chart (the data series needs to be in the Excel spreadsheet).


```python
# define the workbook
workbook = writer.book
worksheet = writer.sheets[sheet_name]

# create a chart line object
chart = workbook.add_chart({'type': 'line'})

# configure the series of the chart from the spreadsheet
# using a list of values instead of category/value formulas:
#     [sheetname, first_row, first_col, last_row, last_col]
chart.add_series({
    'categories': [sheet_name, 1, 0, 3, 0],
    'values':     [sheet_name, 1, 1, 3, 1],
})

# configure the chart axes
chart.set_x_axis({'name': 'Index', 'position_axis': 'on_tick'})
chart.set_y_axis({'name': 'Value', 'major_gridlines': {'visible': False}})

# place the chart on the worksheet
worksheet.insert_chart('E2', chart)

# output the excel file
writer.save()
```

## 5. Save the disk space

When working on multiple Data Science projects, you usually end up with many preprocessed datasets from different experiments.
Your SSD on a laptop can get cluttered quickly.
Pandas enables you to compress the dataset when saving it and then reading back in compressed format.

Let's create a big pandas DataFrame with random numbers.


```python
df = pd.DataFrame(pd.np.random.randn(50000,300))
df.shape
```




    (50000, 300)




```python
df.head()
```




<div style="overflow-x:scroll;">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>290</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.034521</td>
      <td>-1.480572</td>
      <td>1.095903</td>
      <td>0.164909</td>
      <td>0.145135</td>
      <td>1.708804</td>
      <td>0.535697</td>
      <td>-0.227051</td>
      <td>0.422574</td>
      <td>0.899798</td>
      <td>...</td>
      <td>0.743022</td>
      <td>2.616465</td>
      <td>0.541790</td>
      <td>0.370654</td>
      <td>1.253279</td>
      <td>-2.299622</td>
      <td>0.923463</td>
      <td>0.653043</td>
      <td>-1.985603</td>
      <td>1.913356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.767697</td>
      <td>-0.987742</td>
      <td>0.215685</td>
      <td>-0.955054</td>
      <td>0.028924</td>
      <td>-0.087211</td>
      <td>-1.516443</td>
      <td>-1.362196</td>
      <td>-0.773938</td>
      <td>0.964846</td>
      <td>...</td>
      <td>1.246892</td>
      <td>1.105367</td>
      <td>-0.651634</td>
      <td>-2.175714</td>
      <td>-1.026294</td>
      <td>0.447006</td>
      <td>-0.303998</td>
      <td>-0.630391</td>
      <td>-0.031626</td>
      <td>0.905474</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.103714</td>
      <td>0.314054</td>
      <td>0.286481</td>
      <td>-0.097979</td>
      <td>0.262260</td>
      <td>0.390012</td>
      <td>-0.659802</td>
      <td>0.028104</td>
      <td>-0.286029</td>
      <td>0.435272</td>
      <td>...</td>
      <td>-0.610004</td>
      <td>-0.914822</td>
      <td>-0.555851</td>
      <td>-0.455562</td>
      <td>-0.218939</td>
      <td>-0.035112</td>
      <td>1.299518</td>
      <td>0.655536</td>
      <td>0.504187</td>
      <td>-0.049588</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.691572</td>
      <td>2.525289</td>
      <td>-1.598500</td>
      <td>0.630398</td>
      <td>-0.025554</td>
      <td>1.300467</td>
      <td>0.528646</td>
      <td>-0.632779</td>
      <td>0.781360</td>
      <td>-0.177085</td>
      <td>...</td>
      <td>-0.946025</td>
      <td>0.278085</td>
      <td>-1.978881</td>
      <td>-0.057186</td>
      <td>-0.123851</td>
      <td>-0.729205</td>
      <td>0.347192</td>
      <td>0.363281</td>
      <td>1.500823</td>
      <td>0.026872</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.278685</td>
      <td>-1.258590</td>
      <td>0.328225</td>
      <td>-0.371242</td>
      <td>1.255827</td>
      <td>0.272875</td>
      <td>-1.263065</td>
      <td>-1.180428</td>
      <td>1.453985</td>
      <td>0.373956</td>
      <td>...</td>
      <td>-0.892563</td>
      <td>0.601878</td>
      <td>-0.849996</td>
      <td>2.799809</td>
      <td>1.303018</td>
      <td>0.071240</td>
      <td>0.677262</td>
      <td>0.984029</td>
      <td>-1.361368</td>
      <td>0.320774</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 300 columns</p>
</div>
</div>



When we save this file as CSV, it takes almost 300 MB on the hard drive.


```python
df.to_csv('random_data.csv', index=False)
df.shape
```




    (50000, 300)



With a single argument `compression='gzip'`, we can reduce the file size to 136 MB.


```python
df.to_csv('random_data.gz', compression='gzip', index=False)
df.shape
```




    (50000, 300)



It is also easy to read the gzipped data to the DataFrame, so we don't lose any functionality.


```python
df = pd.read_csv('random_data.gz')
df.shape
```




    (50000, 300)



## Conclusion

These tricks help me daily to be more productive with pandas.
Hopefully, this blog post showed you a new pandas function, that will help you to be more productive. 

What's your favorite pandas trick?
